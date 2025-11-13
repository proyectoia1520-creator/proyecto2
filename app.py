import io
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
import streamlit as st


# ----------------------------
# app config
# ----------------------------
st.set_page_config(page_title="Pulmo-ML Viewer", page_icon="🫁", layout="wide")
st.title("🫁 Pulmo-ML Viewer - clasificacion pulmonar")


# ----------------------------
# utils
# ----------------------------
@st.cache_resource
def load_ckpt(ckpt_path: Path, force_classes=None):
    import re
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # --- clases ---
    if force_classes:
        classes = force_classes
    else:
        classes = ckpt.get("classes")
        if classes is None:
            raise ValueError("el checkpoint no trae 'classes'; define el orden manual en la barra lateral")
    num_classes = len(classes)

    # --- state_dict bruto (acepta ckpt['model'] o el dict entero) ---
    raw = ckpt.get("model", ckpt)
    if not isinstance(raw, dict):
        raise ValueError("el checkpoint no contiene un dict de pesos en 'model'")

    # --- desanidar y quitar prefijos comunes ---
    def unwrap(sd):
        for k in ("state_dict","model_state_dict","weights","params","model"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
        return sd
    sd = unwrap(raw)

    def strip_prefixes(sd, prefixes=("module.","model.","backbone.")):
        out = {}
        for k,v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            out[nk] = v
        return out
    sd = strip_prefixes(sd)

    # --- auto-detect de resnet18 vs resnet50 por llaves ---
    # Heuristica: ResNet50 (bottleneck) tiene conv3 en los bloques: layer1.0.conv3, layer2.*.conv3, etc.
    keys = list(sd.keys())
    has_conv3 = any(".conv3.weight" in k for k in keys)  # tipico de bottleneck
    # Si el ckpt trae args, se respeta; si no, heuristica
    model_name = (ckpt.get("args", {}) or {}).get("model")
    if model_name is None:
        model_name = "resnet50" if has_conv3 else "resnet18"
    model_name = model_name.lower()

    # --- construir el modelo correcto ---
    from models.cnn_basica_def import CNNSimple

    # si las llaves contienen 'f.' o 'c.', se asume CNN personalizada
    if any(k.startswith(("f.","c.")) for k in sd.keys()):
        model = CNNSimple(num_classes=num_classes)
        model_name = "cnn_basica"
    else:
        if model_name == "resnet50":
            model = tvm.resnet50(weights=None)
        else:
            model = tvm.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)


    # --- cargar pesos: primero estricto; si falla, no estricto y reporta ---
    warn = None
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        warn = model.load_state_dict(sd, strict=False)
        model.__load_warnings__ = warn

    model.eval()
    temp = ckpt.get("temperature", None)

    # --- adjunta debug corto de llaves para UI ---
    if warn:
        miss = list(warn.missing_keys)[:10]
        unexp = list(warn.unexpected_keys)[:10]
        model.__load_debug__ = {"missing_sample": miss, "unexpected_sample": unexp}

    return model, classes, temp, model_name


def preprocess(img_pil, size=384, use_imagenet_norm=True):
    ops = [T.Grayscale(num_output_channels=3),
           T.Resize(size),
           T.CenterCrop(size),
           T.ToTensor()]
    if use_imagenet_norm:
        ops.append(T.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225]))
    return T.Compose(ops)(img_pil).unsqueeze(0)


def last_conv_layer(model: nn.Module):
    for _, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Conv2d):
            return m
    return None


def gradcam(model: nn.Module, x: torch.Tensor):
    """
    simple grad-cam sobre la ultima conv detectada.
    retorna mapa CAM normalizado (H,W) y pred_idx.
    """
    layer = last_conv_layer(model)
    if layer is None:
        raise RuntimeError("no se encontro una capa conv en el modelo (grad-cam no disponible)")

    activations = []
    gradients = []

    def fwd_hook(_, __, out):
        activations.append(out.detach())

    def bwd_hook(_, gin, gout):
        gradients.append(gout[0].detach())

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    out = model(x)
    pred_idx = int(out.argmax(1).item())
    loss = out[0, pred_idx]
    loss.backward()

    a = activations[-1][0]           # (C,H,W)
    g = gradients[-1][0]             # (C,H,W)
    w = g.mean(dim=(1, 2))           # (C,)
    cam = (w[:, None, None] * a).sum(0).clamp(min=0)  # (H,W)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()
    return cam.cpu().numpy(), pred_idx


# ----------------------------
# auto-discover local models
# ----------------------------
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
available_models = sorted([p.name for p in models_dir.glob("*.pt")])


# ----------------------------
# sidebar - config
# ----------------------------
st.sidebar.header("config")
if not available_models:
    st.sidebar.warning("no hay archivos .pt en carpeta 'models/'")
model_file = st.sidebar.selectbox("modelo local (.pt)", available_models) if available_models else None
img_size = st.sidebar.number_input("tamano de entrada (px)", min_value=128, max_value=1024, value=384, step=32)
use_imagenet = st.sidebar.checkbox("normalizacion imagenet", value=True)
top_k = st.sidebar.slider("top-k", min_value=1, max_value=10, value=5)
raw_classes = st.sidebar.text_input("orden de clases manual (coma-separado, opcional)", "")
force_classes = [c.strip() for c in raw_classes.split(",")] if raw_classes.strip() else None
show_cam = st.sidebar.checkbox("mostrar grad-cam", value=True)


# ----------------------------
# body - ui
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("imagen")
    uploaded = st.file_uploader("sube jpg/png", type=["jpg", "jpeg", "png"])
    img_pil = None
    if uploaded is not None:
        img_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(img_pil, caption="imagen cargada", use_container_width=True)

with col2:
    st.subheader("modelo")
    model = None
    classes = None
    temp = None
    model_name = None
    if model_file:
        try:
            model, classes, temp, model_name = load_ckpt(models_dir / model_file, force_classes=force_classes)
            st.success(f"modelo: {model_name} | clases: {classes}")
            warn = getattr(model, "__load_warnings__", None)
            if warn:
                st.warning(f"carga no estricta: missing={len(warn.missing_keys)}, unexpected={len(warn.unexpected_keys)}")
                dbg = getattr(model, "__load_debug__", {})
                if dbg:
                    st.caption(f"missing_sample: {dbg.get('missing_sample')}")
                    st.caption(f"unexpected_sample: {dbg.get('unexpected_sample')}")
            st.caption(f"fc.out_features = {model.fc.out_features}")
        except Exception as e:
            st.error(f"error al cargar: {e}")

st.markdown("---")

if (img_pil is not None) and (model is not None):
    # prediccion
    x = preprocess(img_pil, size=img_size, use_imagenet_norm=use_imagenet)
    with torch.no_grad():
        logits = model(x)
        if temp:
            logits = logits / temp
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    pred_label = classes[idx] if classes else f"clase_{idx}"
    st.subheader("resultado")
    st.success(f"clase predicha: {pred_label}")
    # top-k
    order = np.argsort(probs)[::-1][:top_k]
    st.write({(classes[i] if classes else f'clase_{i}'): float(probs[i]) for i in order})
    st.bar_chart({(classes[i] if classes else f'clase_{i}'): float(probs[i]) for i in order})

    # grad-cam
    if show_cam:
        try:
            cam, pred_idx = gradcam(model, x)
            # overlay simple con matplotlib-free: usar numpy + PIL
            cam_uint8 = (cam * 255).astype("uint8")
            cam_img = Image.fromarray(cam_uint8).resize(img_pil.size, resample=Image.BILINEAR).convert("L")
            # heatmap basico: mezclar cam en canal R
            img_arr = np.array(img_pil).astype("float32")
            heat = np.stack([np.array(cam_img), np.zeros_like(cam_uint8), np.zeros_like(cam_uint8)], axis=-1).astype("float32")
            heat = (heat / 255.0) * 255.0
            overlap = (0.5 * img_arr + 0.5 * heat).clip(0, 255).astype("uint8")
            st.image(overlap, caption="grad-cam (overlay R)", use_column_width=True)
        except Exception as e:
            st.info(f"grad-cam no disponible: {e}")
else:
    st.info("sube una imagen y selecciona un modelo para predecir")

