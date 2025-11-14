import io
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
import streamlit as st


# ==========================================================
#  MODELOS CNN
# ==========================================================

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


class CNNSequential(nn.Module):
    """
    Usado para checkpoints con capas conv.* y sin definir claramente la cabeza.
    Construimos la parte conv con pesos del checkpoint y la fc nueva con 5 clases.
    """
    def __init__(self, out_channels, num_classes=5):
        super().__init__()
        self.conv = nn.Identity()  # se rellena luego con un nn.Sequential
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ==========================================================
#  APP CONFIG
# ==========================================================
st.set_page_config(page_title="Pulmo-ML Viewer", page_icon="🫁", layout="wide")
st.title("🫁 Pulmo-ML Viewer - Clasificacion pulmonar")


# ==========================================================
#  CARGA DE CHECKPOINT (RESNET18/50 + CNNs) 
# ==========================================================
@st.cache_resource
def load_ckpt(ckpt_path: Path):
    import re

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # -----------------------------
    # CLASES: siempre 5 de pulmon
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
        # intentar classes.txt
        cl_path = ckpt_path.parent / "classes.txt"
        if cl_path.exists():
            with open(cl_path, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            classes = DEFAULT_CLASSES
    else:
        classes = list(classes)
        if len(classes) < 5:
            classes = DEFAULT_CLASSES
        else:
            classes = classes[:5]

    num_classes = 5
    classes = classes[:5]

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
        raise ValueError("No se encontro un dict de pesos en el checkpoint.")

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
    has_resnet_layers = any(
        k.startswith("layer1.") or k.startswith("conv1.") for k in sd.keys()
    )
    has_sequential_keys = any(k.startswith("conv.") or k.startswith("fc.") for k in sd.keys())
    has_custom_keys = any(
        k.startswith(("f0.", "f3.", "f6.", "f9.", "c3."))
        or k.startswith(("f.", "c."))
        for k in sd.keys()
    )
    arch_meta = (ckpt.get("arch") or (ckpt.get("args", {}) or {}).get("model") or "").lower()

    # =====================================================
    #  CASO 1: RESNET (18 o 50 segun fc.weight de checkpoint)
    # =====================================================
    if has_resnet_layers or arch_meta.startswith("resnet"):
        fc_in = None
        if "fc.weight" in sd:
            fc_in = sd["fc.weight"].shape[1]  # columnas = in_features de la fc original

        if fc_in == 2048:
            backbone = tvm.resnet50(weights=None)
            model_name = "resnet50"
        else:
            backbone = tvm.resnet18(weights=None)
            model_name = "resnet18"

        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        backbone.load_state_dict(sd, strict=False)
        backbone.eval()
        return backbone, classes, model_name

    # =====================================================
    #  CASO 2: CNNSimple (f0,f3,f6,f9,c3)
    # =====================================================
    if has_custom_keys or arch_meta.startswith("cnn"):
        fixed = {}
        pat = re.compile(r"^(f|c)\.(\d+)\.(.+)$")
        for k, v in sd.items():
            m = pat.match(k)
            if m:
                nk = f"{m.group(1)}{m.group(2)}.{m.group(3)}"
            else:
                nk = k
            fixed[nk] = v
        sd_simple = fixed

        model = CNNSimple(num_classes=num_classes)
        model.load_state_dict(sd_simple, strict=False)
        model.eval()
        return model, classes, "cnn_basica"

    # =====================================================
    #  CASO 3: CNNSequential (conv.* y fc.*)
    # =====================================================
    if has_sequential_keys:
        conv_items = [(k, v) for k, v in sd.items() if k.startswith("conv.")]
        conv_layers = {}

        for full_key, tensor in conv_items:
            # conv.0.weight -> capa 0, param weight
            _, lid, param = full_key.split(".", 2)
            lid = int(lid)
            conv_layers.setdefault(lid, {})[param] = tensor

        conv_seq = []
        sorted_ids = sorted(conv_layers.keys())
        out_channels = None

        for lid in sorted_ids:
            params = conv_layers[lid]
            if "weight" in params:
                w = params["weight"]
                kh, kw = w.shape[2], w.shape[3]
                layer = nn.Conv2d(
                    in_channels=w.shape[1],
                    out_channels=w.shape[0],
                    kernel_size=(kh, kw),
                    padding=0 if (kh == 1 and kw == 1) else 1,
                )
                layer.weight.data = w
                if "bias" in params:
                    layer.bias.data = params["bias"]
                else:
                    nn.init.zeros_(layer.bias)
                out_channels = w.shape[0]
                conv_seq.append(layer)
            else:
                conv_seq.append(nn.ReLU())

        backbone = nn.Sequential(*conv_seq)
        if out_channels is None:
            out_channels = 128  # fallback

        model = CNNSequential(out_channels=out_channels, num_classes=num_classes)
        model.conv = backbone
        model.eval()
        return model, classes, "cnn_sequential"

    # =====================================================
    #  Fallback: resnet18 sin pesos
    # =====================================================
    fallback = tvm.resnet18(weights=None)
    fallback.fc = nn.Linear(fallback.fc.in_features, num_classes)
    fallback.eval()
    return fallback, classes, "resnet18_fallback"


# ==========================================================
#  PREPROCESS
# ==========================================================
def preprocess(img_pil, size=384):
    ops = [
        T.Grayscale(num_output_channels=3),
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
    ]
    return T.Compose(ops)(img_pil).unsqueeze(0)


# ==========================================================
#  (Opcional) GRAD-CAM - si luego quieres agregar, se puede
# ==========================================================
def last_conv_layer(model):
    for _, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Conv2d):
            return m
    return None


def gradcam(model, x):
    layer = last_conv_layer(model)
    if layer is None:
        raise RuntimeError("No se encontro una capa conv para grad-cam.")

    activations = []
    gradients = []

    def fwd(_, __, out):
        activations.append(out.detach())

    def bwd(_, gin, gout):
        gradients.append(gout[0].detach())

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)

    model.zero_grad()
    out = model(x)
    pred_idx = int(out.argmax(1).item())
    out[0, pred_idx].backward()

    a = activations[-1][0]      # (C,H,W)
    g = gradients[-1][0]        # (C,H,W)
    w = g.mean(dim=(1, 2))      # (C,)

    cam = (w[:, None, None] * a).sum(0).clamp(min=0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()
    return cam.cpu().numpy(), pred_idx


# ==========================================================
#  DISCOVER MODELS
# ==========================================================
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
available_models = sorted([p.name for p in models_dir.glob("*.pt")])


# ==========================================================
#  SIDEBAR
# ==========================================================
st.sidebar.header("Config")

if not available_models:
    st.sidebar.warning("No hay modelos en carpeta 'models/'")

model_file = st.sidebar.selectbox("Modelo (.pt)", available_models) if available_models else None
img_size = st.sidebar.slider("Tamaño de entrada (px)", 128, 1024, 384, 32)
top_k = st.sidebar.slider("top-k", 1, 5, 5)
show_cam = st.sidebar.checkbox("Mostrar grad-cam", False)


# ==========================================================
#  UI PRINCIPAL
# ==========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagen")
    uploaded = st.file_uploader("Sube una imagen pulmonar", type=["jpg", "png", "jpeg"])
    img_pil = None
    if uploaded:
        img_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(img_pil, caption="Imagen cargada", use_column_width=True)

with col2:
    st.subheader("Modelo")
    model = None
    classes = None
    model_name = None

    if model_file:
        try:
            model, classes, model_name = load_ckpt(models_dir / model_file)
            st.success(f"Modelo detectado: {model_name}")
            st.write("Clases:", classes)
        except Exception as e:
            st.error(f"Error al cargar modelo: {e}")

st.markdown("---")

# ==========================================================
#  PREDICCION
# ==========================================================
if img_pil is not None and model is not None:
    x = preprocess(img_pil, img_size)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    pred = classes[idx]

    st.subheader("Resultado")
    st.success(f"Prediccion: {pred} (indice {idx})")

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
            st.error(f"Error en grad-cam: {e}")
else:
    if model is None:
        st.info("Selecciona un modelo .pt en la barra lateral.")
    if img_pil is None:
        st.info("Sube una imagen para predecir.")
