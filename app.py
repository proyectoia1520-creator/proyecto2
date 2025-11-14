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
# modelos para los checkpoints nuevos
# ----------------------------
class CNNBasicaSeq(nn.Module):
    """
    Misma arquitectura que la usada en el script de entrenamiento:
    - 3 conv + maxpool
    - 2 capas FC (256 -> nc)
    """
    def __init__(self, nc: int, im_size: int = 224):
        super().__init__()
        self.im_size = im_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * (im_size//8) * (im_size//8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, nc)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ----------------------------
# utils
# ----------------------------
@st.cache_resource
def load_ckpt(ckpt_path: Path, force_classes=None):
    import re

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # -------- clases --------
    DEFAULT_CLASSES = [
        "bacterial_pneumonia",
        "covid",
        "normal_lung",
        "tuberculosis",
        "viral_pneumonia",
    ]

    def load_classes():
        # 1) sidebar (override manual)
        if force_classes:
            return list(force_classes)
        # 2) del checkpoint (formato viejo)
        if isinstance(ckpt, dict) and "classes" in ckpt:
            return list(ckpt["classes"])
        # 3) classes.txt al costado del modelo
        classes_txt = ckpt_path.parent / "classes.txt"
        if classes_txt.exists():
            with open(classes_txt, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        # 4) fallback por defecto
        return DEFAULT_CLASSES

    classes = load_classes()
    num_classes = len(classes)

    # -------- sacar state_dict crudo --------
    if isinstance(ckpt, dict) and any(
        k in ckpt for k in ("model_state_dict", "state_dict", "model")
    ):
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError("No se encontro un dict de pesos en el checkpoint.")

    # limpiar prefijos comunes
    def strip_prefixes(d, prefixes=("module.", "model.", "backbone.")):
        out = {}
        for k, v in d.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            out[nk] = v
        return out

    sd = strip_prefixes(sd)

    # normalizar fc.* para resnets viejos (fc.1.* -> fc.*)
    def remap_resnet_fc_keys(sd_in: dict):
        if "fc.weight" not in sd_in and any(k.startswith("fc.1.") for k in sd_in.keys()):
            fixed = {}
            for k, v in sd_in.items():
                if k.startswith("fc.1."):
                    fixed["fc." + k[len("fc.1."):]] = v
                elif not k.startswith("fc.0."):
                    fixed[k] = v
            return fixed
        return sd_in

    sd = remap_resnet_fc_keys(sd)

    keys = list(sd.keys())

    # -------- detectar tipo de modelo por llaves --------
    has_old_cnn = any(
        k.startswith(("f0.", "f3.", "f6.", "f9.", "c3.", "f.", "c."))
        for k in keys
    )
    has_seq_cnn = any(k.startswith("conv.") for k in keys) and any(
        k.startswith("fc.") for k in keys
    ) and not any(k.startswith("layer1.") for k in keys)
    has_resnet = any(k.startswith("layer1.") for k in keys) or "fc.weight" in sd

    # -------- construir modelo segun tipo --------
    model = None
    model_name = "desconocido"

    if has_old_cnn:
        # CNN antigua (CNNSimple con f0,f3,f6,f9,c3)
        from models.cnn_basica_def import CNNSimple

        model = CNNSimple(num_classes=num_classes)

        # remap f.<num>.* -> f<num>.*  y c.<num>.* -> c<num>.*
        fixed = {}
        pat = re.compile(r"^(f|c)\.(\d+)\.(.+)$")
        for k, v in sd.items():
            m = pat.match(k)
            fixed[(f"{m.group(1)}{m.group(2)}.{m.group(3)}" if m else k)] = v
        sd = fixed
        model_name = "cnn_basica_old"

    elif has_seq_cnn:
        # CNNBasica nueva (conv.0 / conv.3 / conv.6 + fc.*)
        model = CNNBasicaSeq(nc=num_classes, im_size=224)
        model_name = "cnn_basica"

    elif has_resnet:
        # ResNet (18 o 50) segun dimension de fc.weight
        fc_in = None
        fc_out = num_classes
        if "fc.weight" in sd:
            fc_weight = sd["fc.weight"]
            if fc_weight.ndim == 2:
                fc_out, fc_in = fc_weight.shape

        if fc_in == 2048:
            base = tvm.resnet50(weights=None)
            model_name = "resnet50"
        elif fc_in == 512:
            base = tvm.resnet18(weights=None)
            model_name = "resnet18"
        else:
            # si no sabemos, asumimos resnet50 porque es lo que usaste en el script
            base = tvm.resnet50(weights=None)
            model_name = "resnet50?"

        base.fc = nn.Linear(base.fc.in_features, fc_out)
        model = base

    else:
        raise RuntimeError(
            f"No pude inferir el tipo de modelo a partir de las llaves: ejemplo {keys[:5]}"
        )

    # -------- cargar pesos --------
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        warn = model.load_state_dict(sd, strict=False)
        model.__load_warnings__ = warn
        miss = list(warn.missing_keys)[:10]
        unexp = list(warn.unexpected_keys)[:10]
        model.__load_debug__ = {"missing_sample": miss, "unexpected_sample": unexp}

    model.eval()
    temp = ckpt.get("temperature", None) if isinstance(ckpt, dict) else None
    return model, classes, temp, model_name


def preprocess(img_pil, size=224, use_imagenet_norm=True):
    # IMPORTANTE: que coincida con eval_tf del entrenamiento
    # eval_tf = Resize((IM_SIZE, IM_SIZE)) + ToTensor + Normalize(...)
    ops = [
        T.Resize((size, size)),
        T.ToTensor(),
    ]
    if use_imagenet_norm:
        ops.append(
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        )
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
        raise RuntimeError(
            "no se encontro una capa conv en el modelo (grad-cam no disponible)"
        )

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

    a = activations[-1][0]  # (C,H,W)
    g = gradients[-1][0]  # (C,H,W)
    w = g.mean(dim=(1, 2))  # (C,)
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

model_file = (
    st.sidebar.selectbox("modelo local (.pt)", available_models)
    if available_models
    else None
)
img_size = st.sidebar.number_input(
    "tamano de entrada (px)", min_value=128, max_value=1024, value=224, step=32
)
use_imagenet = st.sidebar.checkbox("normalizacion imagenet", value=True)
top_k = st.sidebar.slider("top-k", min_value=1, max_value=10, value=5)
raw_classes = st.sidebar.text_input(
    "orden de clases manual (coma-separado, opcional)", ""
)
force_classes = (
    [c.strip() for c in raw_classes.split(",")] if raw_classes.strip() else None
)
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
        st.image(img_pil, caption="imagen cargada", use_column_width=True)

with col2:
    st.subheader("modelo")
    model = None
    classes = None
    temp = None
    model_name = None
    if model_file:
        try:
            model, classes, temp, model_name = load_ckpt(
                models_dir / model_file, force_classes=force_classes
            )
            st.success(f"modelo: {model_name} | clases: {classes}")
            warn = getattr(model, "__load_warnings__", None)
            if warn:
                st.warning(
                    f"carga no estricta: missing={len(warn.missing_keys)}, unexpected={len(warn.unexpected_keys)}"
                )
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

    order = np.argsort(probs)[::-1][:top_k]
    topk_dict = {
        (classes[i] if classes else f"clase_{i}"): float(probs[i]) for i in order
    }
    st.write(topk_dict)
    st.bar_chart(topk_dict)

    if show_cam:
        try:
            cam, pred_idx = gradcam(model, x)
            cam_uint8 = (cam * 255).astype("uint8")
            cam_img = (
                Image.fromarray(cam_uint8)
                .resize(img_pil.size)
                .convert("L")
            )
            img_arr = np.array(img_pil).astype("float32")
            heat = np.stack(
                [np.array(cam_img), np.zeros_like(cam_uint8), np.zeros_like(cam_uint8)],
                axis=-1,
            ).astype("float32")
            overlap = (0.5 * img_arr + 0.5 * heat).clip(0, 255).astype("uint8")
            st.image(overlap, caption="grad-cam (overlay R)", use_column_width=True)
        except Exception as e:
            st.info(f"grad-cam no disponible: {e}")
else:
    st.info("sube una imagen y selecciona un modelo para predecir")
