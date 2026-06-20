import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Resolver paths relativos a la ubicación de este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
METADATA_PATH = os.path.join(ASSETS_DIR, "mlp_metadata.json")

with open(METADATA_PATH, encoding="utf-8") as f:
    META = json.load(f)

CACHE_MODELOS = {}
KEYS_MODELOS = ["superhost", "instant", "disp90", "precio", "calificacion"]

def cargar_modelo_y_prep(key):
    if key not in CACHE_MODELOS:
        m = next(x for x in META["modelos"] if x["key"] == key)
        archivo_keras_path = os.path.join(ASSETS_DIR, m["archivo_keras"])
        archivo_preprocessor_path = os.path.join(ASSETS_DIR, m["archivo_preprocessor"])
        
        modelo = tf.keras.models.load_model(archivo_keras_path)
        prep = joblib.load(archivo_preprocessor_path)
        CACHE_MODELOS[key] = (modelo, prep, m)
    return CACHE_MODELOS[key]

# Pre-cargar todos los modelos en memoria
for k in KEYS_MODELOS:
    cargar_modelo_y_prep(k)

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
if __name__ == '__main__':
    datos_propiedad = {
        "Capacidad": 4,
        "Baños": 1.5,
        "Camas": 3,
        "Precio por noche": 120.0,
        "Tipo de habitación": "Espacio entero",
        "Política de cancelación": "Flexible",
        "Número de cuartos o habitaciones": 2,
        "Tiempo como host en años": 3.0,
        "Instant bookable": "NO",
        "Disponibilidad mayor a 90 días": "SI",
        "Ubicación exacta": "SI",
        "Número de servicios o amenidades": 12,
        "Tipo de alojamiento": "Apartamento",
        "¿Identidad verificada del host?": "SI",
        "Tipo de cama": "Cama real",
        "Promedio de reviews o calificaión": 4.8,
        "Número de reviews o reseñas": 25,
        "¿Host tiene foto de perfil?": "SI",
        "Verificar teléfono de huésped": "NO",
        "¿Es superhost?": "NO"
    }
    resultado = predictMlp(datos_propiedad)
    print("Resultado del ejemplo de uso de MLP:")
    print(resultado)
