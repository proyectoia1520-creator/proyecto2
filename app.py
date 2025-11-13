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

    # clases
    classes = force_classes if force_classes else ckpt.get("classes")
    if not classes:
        raise ValueError("El checkpoint no trae 'classes' y no se ingreso un orden manual.")
    num_classes = len(classes)

    # arquitectura
    model_name = (ckpt.get("arch") or (ckpt.get("args",{}) or {}).get("model") or "cnn_basica").lower()

    # state_dict crudo
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    if not isinstance(sd, dict):
        raise ValueError("No se encontro un dict de pesos en el checkpoint.")

    # limpiar prefijos
    def strip_prefixes(d, prefixes=("module.","model.","backbone.")):
        out={}
        for k,v in d.items():
            nk=k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            out[nk]=v
        return out
    sd = strip_prefixes(sd)

    # decidir arquitectura y construir modelo
    from models.cnn_basica_def import CNNSimple
    # si aparecen llaves con 'f.' o 'c.' asumimos tu CNN
    is_custom = (model_name == "cnn_basica") or any(k.startswith(("f.","c.")) for k in sd.keys())
    if is_custom:
        model = CNNSimple(num_classes=num_classes)
        # remap f.<num>.* -> f<num>.*  y c.<num>.* -> c<num>.*
        fixed={}
        pat = re.compile(r'^(f|c)\.(\d+)\.(.+)$')  # ej: f.3.weight -> f3.weight
        for k,v in sd.items():
            m = pat.match(k)
            fixed[(f"{m.group(1)}{m.group(2)}.{m.group(3)}" if m else k)] = v
        sd = fixed
        model_name = "cnn_basica"
    else:
        import torchvision.models as tvm
        import torch.nn as nn
        if model_name == "resnet50":
            model = tvm.resnet50(weights=None)
        else:
            model = tvm.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # cargar pesos (estricto; si falla, no estricto + warning)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        warn = model.load_state_dict(sd, strict=False)
        model.__load_warnings__ = warn
        miss = list(warn.missing_keys)[:10]
        unexp = list(warn.unexpected_keys)[:10]
        model.__load_debug__ = {"missing_sample": miss, "unexpected_sample": unexp}

    model.eval()
    temp = ckpt.get("temperature", None)
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
            if hasattr(model, "fc"):
                st.caption(f"fc.out_features = {model.fc.out_features}")
            else:
                st.caption("modelo sin capa fc (CNN personalizada)")

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

