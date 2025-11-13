# 🫁 Streamlit Lung X-ray Classifier (CNN Básica / ResNet)

App de Streamlit para comparar dos modelos PyTorch (.pt) de clasificación de radiografías de tórax:
- `cnn_basica.pt`
- `resnet.pt`

## 📁 Estructura del repo

```
streamlit-lung-app/
├── app.py
├── requirements.txt
├── models/
│   ├── cnn_basica.pt      # coloca aquí tu modelo
│   └── resnet50.pt        # coloca aquí tu modelo
└── README.md
```

> Ajusta la lista de clases en `app.py` si tus etiquetas difieren:
```
CLASS_NAMES = [
    "Neumonía bacteriana",
    "Coronavirus",
    "Pulmón normal",
    "Tuberculosis",
    "Neumonía viral",
]
```

## ▶️ Ejecución local (Windows/Mac/Linux)

1) Clona o copia esta carpeta y coloca tus modelos `.pt` en `models/`  
2) Crea un entorno e instala dependencias:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```
3) Inicia la app:
```bash
streamlit run app.py
```
4) Abre el navegador en la URL que imprime Streamlit (usualmente `http://localhost:8501`).

### Notas sobre los modelos
- Si tu `.pt` es un **TorchScript**, se carga con `torch.jit.load`.
- Si tu `.pt` es un **modelo completo** guardado con `torch.save(model)`, se carga con `torch.load`.
- Si tu `.pt` es solo **state_dict**, verás un error. Debes reconstruir la arquitectura y cargar el state_dict, o exportar a TorchScript.

## 🚀 Deploy desde GitHub (Streamlit Community Cloud)

1) Sube todo el contenido a un repositorio público de GitHub (por ejemplo, `usatin/streamlit-lung-app`).  
2) Ve a [streamlit.io/cloud](https://streamlit.io/cloud) ➜ **New app** ➜ conecta tu GitHub.  
3) Selecciona el repo y rama, y define el **Main file path** como `app.py`.  
4) En **Advanced settings**, puedes dejar por defecto.  
5) Sube tus modelos a la carpeta `models/` del repo (o usa **Secrets** + descarga desde un storage propio si los .pt son pesados).  
6) Lanza el deploy. Si el archivo .pt es muy grande, considera Git LFS o alojarlo externamente y descargarlo en tiempo de ejecución.

## 🔧 Ajustes recomendados
- Si entrenaste con otro tamaño o normalización, ajusta en la barra lateral:
  - **Tamaño de entrada (px)**
  - **Normalización ImageNet**
- Si tus clases están en otro orden, edita `CLASS_NAMES` en `app.py`.

## 🛟 Problemas comunes
- **El .pt no carga**: probablemente es un state_dict. Exporta un TorchScript o guarda el modelo completo.
- **Predicciones raras**: confirma que el preprocesamiento (tamaño/normalización) coincida con el del entrenamiento.
- **CUDA/MPS**: la app usa automáticamente GPU si está disponible; si no, CPU.

¡Listo! 🎉