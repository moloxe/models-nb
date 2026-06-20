# Models NB Server

Este proyecto expone un servidor en Flask con modelos de Machine Learning (MLP y CNN).

## Requisitos Previos
- Python 3.9 o superior.
- Make (opcional, pero recomendado para automatización).

## Instalación y Ejecución

Puedes usar el comando `make` para automatizar todo:

1. **Instalar dependencias y crear entorno virtual:**
   ```bash
   make install
   ```

2. **Ejecutar el servidor:**
   ```bash
   make run
   ```

## Forma Manual (Sin Make)

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
