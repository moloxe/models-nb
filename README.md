# Models NB Server

Este proyecto expone un servidor en Flask con modelos de Machine Learning (MLP y CNN).

## Requisitos Previos
- Python 3.9 o superior.
- Make (opcional, pero recomendado para automatización).

## Instalación y Ejecución con Conda (Recomendado)

Debido a que las librerías de Machine Learning (TensorFlow, PyTorch) pueden ser sensibles a las versiones de Python, se recomienda usar **Conda** para crear un entorno aislado con Python 3.11.

1. **Crear y activar el entorno:**
   ```bash
   conda create -n models-nb python=3.11 -y
   conda activate models-nb
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el servidor:**
   ```bash
   python server.py
   ```

## Instalación con Make (Entorno Virtual)

Si no cuentas con `make`, puedes correr los siguientes comandos:

```bash
# 1. Crear entorno virtual
python3 -m venv venv

# 2. Activar entorno virtual
# En Mac/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Iniciar el servidor
python3 server.py
```

El servidor estará escuchando en `http://0.0.0.0:5000` con los siguientes endpoints:
- `GET /` - Verifica que el servicio está activo.
- `POST /mlp` - Predicción usando Multi-Layer Perceptron.
- `POST /cnn` - Predicción de imágenes usando Convolutional Neural Network.
