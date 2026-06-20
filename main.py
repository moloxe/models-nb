import torch

print(f"Versión de PyTorch: {torch.__version__}")

# Verificar si el chip Apple Silicon (MPS) está disponible
if torch.backends.mps.is_available():
    print("¡Éxito! Tu chip Apple Silicon (MPS) está disponible para aceleración.")
else:
    print("MPS no disponible. Se ejecutará solo en CPU.")

# --- MLP
# --- MLP
# --- MLP

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask import request, jsonify

# ──────────────────────────────────────────────────────────────────────────────
# 1. INICIALIZACIÓN Y CACHÉ DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────
with open("mlp_metadata.json", encoding="utf-8") as f:
    META = json.load(f)

CACHE_MODELOS = {}
KEYS_MODELOS = ["superhost", "instant", "disp90", "precio", "calificacion"]

def cargar_modelo_y_prep(key):
    if key not in CACHE_MODELOS:
        m = next(x for x in META["modelos"] if x["key"] == key)
        modelo = tf.keras.models.load_model(m["archivo_keras"])
        prep = joblib.load(m["archivo_preprocessor"])
        CACHE_MODELOS[key] = (modelo, prep, m)
    return CACHE_MODELOS[key]

# Pre-cargar todos los modelos en memoria
for k in KEYS_MODELOS:
    cargar_modelo_y_prep(k)

# ──────────────────────────────────────────────────────────────────────────────
# 2. FUNCIÓN DE PREDICCIÓN MULTITARGET
# ──────────────────────────────────────────────────────────────────────────────
def predictMlp(data):
    """
    Recibe un diccionario con los datos crudos de una propiedad.
    Devuelve un diccionario con las predicciones de los 5 modelos.
    """
    try:
        df_input = pd.DataFrame([data])
        resultados = {}

        for key in KEYS_MODELOS:
            modelo, prep, m = CACHE_MODELOS[key]

            # 1. Filtrar solo las columnas que conoce el preprocesador
            cols_modelo = m["cols_num"] + m["cols_cat"]
            df_local = df_input[cols_modelo].copy()

            # 2. Forzar explícitamente a numérico las columnas correspondientes
            for col in m["cols_num"]:
                df_local[col] = pd.to_numeric(df_local[col], errors='coerce')

            # 3. Transformar y predecir
            X = np.asarray(prep.transform(df_local), dtype="float32")
            out = modelo.predict(X, verbose=0).ravel()

            if m["tipo"] == "binario":
                prob = float(out[0])
                clase = m["clases"][1] if prob > m["umbral_decision"] else m["clases"][0]
                resultados[key] = {"prediccion": clase, "score": round(prob, 4)}

            elif m["tipo"] == "multiclase":
                idx = int(out.argmax())
                resultados[key] = {"prediccion": m["clases"][idx], "score": round(float(out[idx]), 4)}

            elif m["tipo"] == "regresion":
                resultados[key] = {"prediccion": round(float(out[0]), 2), "score": None}

        return resultados
    except Exception as e:
        return {"error": str(e)}

# Ejemplo de uso:
# datos_propiedad = {
#     "Capacidad": 4,
#     "Baños": 1.5,
#     "Camas": 3,
#     "Precio por noche": 120.0,
#     "Tipo de habitación": "Espacio entero",
#     "Política de cancelación": "Flexible",
#     "Número de cuartos o habitaciones": 2,
#     "Tiempo como host en años": 3.0,
#     "Instant bookable": "NO",
#     "Disponibilidad mayor a 90 días": "SI",
#     "Ubicación exacta": "SI",
#     "Número de servicios o amenidades": 12,
#     "Tipo de alojamiento": "Apartamento",
#     "¿Identidad verificada del host?": "SI",
#     "Tipo de cama": "Cama real",
#     "Promedio de reviews o calificaión": 4.8,
#     "Número de reviews o reseñas": 25,
#     "¿Host tiene foto de perfil?": "SI",
#     "Verificar teléfono de huésped": "NO",
#     "¿Es superhost?": "NO"
# }
# resultado = predictMlp(datos_propiedad)
# print(resultado)

# --- CNN
# --- CNN
# --- CNN

import torch
import torch.nn as nn
from PIL import Image as PILImage
import torchvision.transforms as T
import torchvision.models as models

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN Y RECONSTRUCCIÓN DE LA ARQUITECTURA
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(DEVICE)

CNN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def construir_cnn():
    # Carga la arquitectura base (sin pesos preentrenados por eficiencia, ya que cargarás los tuyos)
    bb = models.efficientnet_b0()
    bb.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    return bb

# ──────────────────────────────────────────────────────────────────────────────
# 2. CARGA DEL MODELO
# ──────────────────────────────────────────────────────────────────────────────
# Asegúrate de que 'cnn_model.pth' esté en el mismo directorio o proporciona la ruta completa
modelo = construir_cnn()
modelo.load_state_dict(torch.load('cnn_model.pth', map_location=DEVICE))
modelo.to(DEVICE)
modelo.eval()

# ──────────────────────────────────────────────────────────────────────────────
# 3. FUNCIÓN DE PREDICCIÓN
# ──────────────────────────────────────────────────────────────────────────────
def predictCnn(image):
    """
    Recibe la ruta de una imagen (str) o un objeto PIL Image.
    Devuelve la probabilidad (0 a 1) calculada por el modelo.
    """
    try:
        if isinstance(image, str):
            img = PILImage.open(image).convert('RGB')
        else:
            img = image.convert('RGB')

        tensor = CNN_TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = modelo(tensor).squeeze()
            proba = torch.sigmoid(out).item()

        return round(proba, 4)

    except Exception as e:
        return f"Error al procesar la imagen: {e}"

# Ejemplo de uso:
# resultado = predictCnn("/content/interior.webp")
# print(resultado)

# --- SERVER
# --- SERVER
# --- SERVER

from flask import Flask, jsonify, request
import threading
from flask_cors import CORS
import io
from PIL import Image as PILImage
app = Flask(__name__)
CORS(app)
@app.route('/', methods=['GET'])
def alive():
    return jsonify({"alive": True})
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

@app.route('/mlp', methods=['POST'])
def mlp_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Body vacío o no es JSON válido'}), 400
    ans = predictMlp(data)
    if "error" in ans:
        return jsonify(ans), 500
    return jsonify({'ans': ans})

@app.route('/cnn', methods=['POST'])
def cnn_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen bajo la clave "image"'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Archivo sin nombre o vacío'}), 400
    try:
        image_bytes = file.read()
        img = PILImage.open(io.BytesIO(image_bytes))
        ans = predictCnn(img)
        return jsonify({'ans': ans})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def run_app():
    app.run(port=5000, host='0.0.0.0', debug=True, use_reloader=False)
threading.Thread(target=run_app).start()
