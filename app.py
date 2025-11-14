import io
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
import streamlit as st


# ----------------------------------------------------------
# MODELO 1: CNNSimple (el tuyo clásico con f0, f3, f6, f9)
# ----------------------------------------------------------
class CNNSimple(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.f0 = nn.Conv2d(3, 32, 3, padding=1)
        self.f3 = nn.Conv2d(32, 64, 3, padding=1)
        self.f6 = nn.Conv2d(64, 128, 3, padding=1)
        self.f9 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.c3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.f0(x))
        x = torch.relu(self.f3(x))
        x = torch.relu(self.f6(x))
        x = torch.relu(self.f9(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.c3(x)
        return x


# ----------------------------------------------------------
# MODELO 2: CNNSequential (el que coincide con TUS MODELOS NUEVOS)
# ----------------------------------------------------------
class CNNSequential(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # conv.0
            nn.ReLU(),                        # conv.1
            nn.Conv2d(32, 64, 3, padding=1),  # conv.2
            nn.ReLU(),                        # conv.3
            nn.Conv2d(64, 128, 3, padding=1), # conv.4
            nn.ReLU(),                        # conv.5
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(128, 64),               # fc.0
            nn.ReLU(),                        # fc.1
            nn.Linear(64, num_classes),       # fc.2
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ----------------------------------------------------------
# APP CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Pulmo-ML Viewer", page_icon="🫁", layout="wide")
st.title("🫁 Pulmo-ML Viewer - clasificacion pulmonar")


# ----------------------------------------------------------
# LOAD CHECKPOINT WITH AUTO-DETECT ARCHITECTURE
# ----------------------------------------------------------
@st.cache_resource
def load_ckpt(ckpt_path: Path):
    import re
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # -----------------------------
    # Clases del checkpoint o fallback
    # -----------------------------
    DEFAULT_CLASSES = [
        "bacterial_pneumonia",
        "covid",
        "normal_lung",
        "tuberculosis",
        "viral_pneumonia",
    ]

    classes = ckpt.get("classes")

    if classes is None:
        cl_path = ckpt_path.parent / "classes.txt"
        if cl_path.exists():
            with open(cl_path, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            classes = DEFAULT_CLASSES

    num_classes = len(classes)

    # -------------------------------------------
    # Obtener state_dict real del checkpoint
    # -------------------------------------------
    sd = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt.get("model")
        or ckpt
    )
    if not isinstance(sd, dict):
        raise ValueError("No se encontró un dict de pesos.")

    # -------------------------------------------
    # Limpieza de prefijos
    # -------------------------------------------
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

    # -------------------------------------------
    # AUTO-DETECTAR ARQUITECTURA
    # -------------------------------------------
    is_sequential = any(k.startswith("conv.") or k.startswith("fc.") for k in sd.keys())
    is_custom = any(k.startswith(("f.", "c.")) for k in sd.keys())

    # --- MODELO NUEVO: CNNSequential ---
    if is_sequential:
        model = CNNSequential(num_classes=num_classes)
        model_name = "cnn_sequential"

    # --- MODELO ANTIGUO: CNNSimple ---
    elif is_custom:
        # remap f.X → fX
        fixed = {}
        pat = re.compile(r"^(f|c)\.(\d+)\.(.+)$")
        for k, v in sd.items():
            m = pat.match(k)
            fixed[(f"{m.group(1)}{m.group(2)}.{m.group(3)}" if m else k)] = v
        sd = fixed
        model = CNNSimple(num_classes=num_classes)
        model_name = "cnn_basica"

    # --- RESNET ---
    else:
        arch = (ckpt.get("arch") or "").lower()
        if arch == "resnet50":
            model = tvm.resnet50(weights=None)
        else:
            model = tvm.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_name = arch or "resnet"

    # -------------------------------------------
    # Cargar pesos
    # -------------------------------------------
    try:
        model.load_state_dict(sd, strict=True)
    except:
        warn = model.load_state_dict(sd, strict=False)
        model.__load_warnings__ = warn

    model.eval()
    return model, classes, model_name


# ----------------------------------------------------------
# PREPROCESS
# ----------------------------------------------------------
def preprocess(img_pil, size=384, use_imagenet_norm=True):
    ops = [
        T.Grayscale(num_output_channels=3),
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
    ]
    if use_imagenet_norm:
        ops.append(
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        )
    return T.Compose(ops)(img_pil).unsqueeze(0)


# ----------------------------------------------------------
# GRAD-CAM
# ----------------------------------------------------------
def last_conv_layer(model):
    for _, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Conv2d):
            return m
    return None


def gradcam(model, x):
    layer = last_conv_layer(model)
    if layer is None:
        raise RuntimeError("No conv layer for grad-cam.")

    activations = []
    gradients = []

    def fwd(_, __, out):
        activations.append(out.detach())

    def bwd(_, gin, gout):
        gradients.append(gout[0].detach())

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)

    out = model(x)
    pred_idx = int(out.argmax(1).item())
    out[0, pred_idx].backward()

    a = activations[-1][0]
    g = gradients[-1][0]
    w = g.mean(dim=(1, 2))

    cam = (w[:, None, None] * a).sum(0).clamp(min=0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()
    return cam.cpu().numpy(), pred_idx


# ----------------------------------------------------------
# DISCOVER MODELS
# ----------------------------------------------------------
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
available_models = sorted([p.name for p in models_dir.glob("*.pt")])


# ----------------------------------------------------------
# SIDEBAR CONFIG
# ----------------------------------------------------------
st.sidebar.header("config")
model_file = st.sidebar.selectbox("modelo (.pt)", available_models)
img_size = st.sidebar.slider("input size", 128, 1024, 384, 32)
use_imagenet = st.sidebar.checkbox("imagenet norm", True)
top_k = st.sidebar.slider("top-k", 1, 10, 5)
show_cam = st.sidebar.checkbox("grad-cam", True)


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("imagen")
    uploaded = st.file_uploader("sube una imagen", type=["jpg", "png", "jpeg"])
    img_pil = None
    if uploaded:
        img_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(img_pil, caption="imagen cargada", use_column_width=True)

with col2:
    st.subheader("modelo")
    if model_file:
        model, classes, model_name = load_ckpt(models_dir / model_file)
        st.success(f"modelo detectado: {model_name}")
        st.write("clases:", classes)


st.markdown("---")

if img_pil is not None:
    x = preprocess(img_pil, img_size, use_imagenet)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, 1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    pred_label = classes[idx]

    st.subheader("resultado")
    st.success(f"predicción: {pred_label} (indice {idx})")

    order = np.argsort(probs)[::-1][:top_k]
    topk = {classes[i]: float(probs[i]) for i in order}
    st.write(topk)
    st.bar_chart(topk)

    if show_cam:
        try:
            cam, _ = gradcam(model, x)
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
            )
            overlay = (0.5 * img_arr + 0.5 * heat).clip(0, 255).astype("uint8")
            st.image(overlay, caption="grad-cam", use_column_width=True)

        except Exception as e:
            st.error(str(e))
