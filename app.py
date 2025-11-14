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


# EL MODELO QUE USAN TUS CHECKPOINTS NUEVOS
class CNNSequential(nn.Module):
    def __init__(self, out_channels, num_classes=5):
        super().__init__()
        self.conv = nn.Identity()  # se va a reemplazar con conv.* del checkpoint
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
st.title("🫁 Pulmo-ML Viewer - Clasificación Pulmonar")


# ==========================================================
#  CARGA DE CHECKPOINT ULTRA-ROBUSTA
# ==========================================================
@st.cache_resource
def load_ckpt(ckpt_path: Path):

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # CLASES FIJAS
    classes = [
        "bacterial_pneumonia",
        "covid",
        "normal_lung",
        "tuberculosis",
        "viral_pneumonia",
    ]

    # EXTRAER STATE_DICT
    sd = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt.get("model")
        or ckpt
    )
    if not isinstance(sd, dict):
        raise ValueError("checkpoint sin state_dict válido")

    # QUITAR PREFIJOS RAROS
    clean = {}
    for k, v in sd.items():
        for p in ["module.", "model.", "backbone."]:
            if k.startswith(p):
                k = k[len(p):]
        clean[k] = v
    sd = clean

    # ¿ES RESNET?
    is_resnet = any(k.startswith("layer1.") or k.startswith("conv1.") for k in sd)

    # ¿ES CNNSimple?
    is_simple = any(k.startswith(("f0.", "f3.", "f6.", "f9.", "c3.")) for k in sd)

    # ¿ES CNNSequential?
    is_seq = any(k.startswith(("conv.", "fc.")) for k in sd)

    # =======================================================
    #  CASO 1: RESNET
    # =======================================================
    if is_resnet:
        model = tvm.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        model.load_state_dict(sd, strict=False)
        return model.eval(), classes, "resnet18"

    # =======================================================
    #  CASO 2: CNNSimple
    # =======================================================
    if is_simple:
        # remap f.X → fX si viene así
        fixed = {}
        import re
        pat = re.compile(r"^(f|c)\.(\d+)\.(.+)$")
        for k, v in sd.items():
            m = pat.match(k)
            fixed[(f"{m.group(1)}{m.group(2)}.{m.group(3)}" if m else k)] = v
        sd = fixed

        model = CNNSimple()
        model.load_state_dict(sd, strict=False)
        return model.eval(), classes, "cnn_basica"

    # =======================================================
    #  CASO 3: CNNSequential (tu caso nuevo)
    # =======================================================
    if is_seq:
        # 1) reconstruir conv.* en un nn.Sequential
        conv_items = [(k, v) for k, v in sd.items() if k.startswith("conv.")]
        conv_layers = {}

        for full_key, tensor in conv_items:
            # ejemplo: conv.0.weight → capa=0, parámetro=weight
            _, layer_id, param = full_key.split('.', 2)
            layer_id = int(layer_id)
            conv_layers.setdefault(layer_id, {})[param] = tensor

        # Reconstruimos capas reales
        conv_seq = []
        sorted_ids = sorted(conv_layers.keys())

        out_channels = None
        for lid in sorted_ids:
            params = conv_layers[lid]
            if "weight" in params:
                w = params["weight"]
                layer = nn.Conv2d(
                    in_channels=w.shape[1],
                    out_channels=w.shape[0],
                    kernel_size=3,
                    padding=1,
                )
                layer.weight.data = w
                layer.bias.data = params.get("bias", torch.zeros_like(layer.bias))
                out_channels = w.shape[0]
                conv_seq.append(layer)
            else:
                conv_seq.append(nn.ReLU())

        backbone = nn.Sequential(*conv_seq)

        # 2) crear modelo final con esta backbone
        model = CNNSequential(out_channels, num_classes=5)
        model.conv = backbone

        # ignorar pesos de fc.* porque no sirven
        return model.eval(), classes, "cnn_sequential"

    # =======================================================
    #  Fallback: resnet18
    # =======================================================
    model = tvm.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(sd, strict=False)
    return model.eval(), classes, "fallback_resnet"


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
#  DESCUBRIR MODELOS
# ==========================================================
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
available_models = sorted([p.name for p in models_dir.glob("*.pt")])


# ==========================================================
#  SIDEBAR
# ==========================================================
st.sidebar.header("Config")

if not available_models:
    st.sidebar.warning("No hay modelos en carpeta /models")

model_file = st.sidebar.selectbox("Modelo (.pt)", available_models)
img_size = st.sidebar.slider("Tamaño", 128, 1024, 384)
top_k = st.sidebar.slider("top-k", 1, 5, 5)


# ==========================================================
#  UI
# ==========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagen")
    uploaded = st.file_uploader("Sube una imagen pulmonar", type=["jpg", "png", "jpeg"])
    img_pil = None
    if uploaded:
        img_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(img_pil, caption="Imagen cargada")

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
#  PREDICCIÓN
# ==========================================================
if img_pil is not None and model is not None:

    x = preprocess(img_pil, img_size)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    pred = classes[idx]

    st.success(f"Predicción: {pred} (índice {idx})")

    # top-k
    order = np.argsort(probs)[::-1][:top_k]
    topk = {classes[i]: float(probs[i]) for i in order}

    st.write(topk)
    st.bar_chart(topk)
