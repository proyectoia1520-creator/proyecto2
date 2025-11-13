import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import streamlit as st
from pathlib import Path
import io

st.set_page_config(page_title="Pulmo-ML Viewer", page_icon="🫁", layout="wide")
st.title("🫁 Pulmo-ML Viewer — Detección de Enfermedad Pulmonar")

# ---------- UTILS ----------
@st.cache_resource
def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("args", {}).get("model", "resnet18").lower()
    classes = ckpt.get("classes", ["NORMAL","PNEUMONIA","COVID","TUBERCULOSIS","VIRAL"])
    temp = ckpt.get("temperature", None)

    # Crear modelo segun tipo
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    # Cargar pesos
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, classes, temp, model_name

def preprocess(img, size=384):
    tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0)

# ---------- AUTO-LOAD MODELS ----------
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
available_models = sorted([f.name for f in models_dir.glob("*.pt")])

# ---------- UI ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagen de rayos X")
    img_file = st.file_uploader("Sube imagen JPG/PNG", type=["jpg","jpeg","png"])
    img_size = st.slider("Tamaño de entrada", 128, 512, 384, 32)
    if img_file:
        image = Image.open(io.BytesIO(img_file.read())).convert("RGB")
        st.image(image, caption="Imagen cargada", use_container_width=True)

with col2:
    st.subheader("Modelo entrenado")
    if not available_models:
        st.warning("⚠️ No se encontraron archivos .pt en la carpeta 'models/'.")
        st.stop()

    model_choice = st.selectbox("Selecciona modelo local", available_models)
    model_path = models_dir / model_choice
    model, classes, temp, model_name = load_ckpt(model_path)
    st.success(f"Modelo: {model_name} · Clases: {classes}")

if img_file and model is not None:
    st.markdown("---")
    st.subheader("Resultados de predicción")
    x = preprocess(image, img_size)
    with torch.no_grad():
        logits = model(x)
        if temp: logits /= temp
        probs = torch.softmax(logits, dim=1)[0].numpy()
    pred_idx = np.argmax(probs)
    pred_class = classes[pred_idx]
    st.success(f"Diagnóstico estimado: **{pred_class}**")
    st.bar_chart(dict(zip(classes, probs)))
