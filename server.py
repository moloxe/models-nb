from flask import Flask, jsonify, request
from flask_cors import CORS
import io
import threading
from PIL import Image as PILImage

# Importar funciones de predicción
from mlp import predictMlp
from cnn import predictCnn

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def alive():
    return jsonify({"alive": True})

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

def run_app():
    app.run(port=5000, host='0.0.0.0', debug=True, use_reloader=False)

if __name__ == '__main__':
    # Si se ejecuta como script principal, se corre en el hilo principal
    run_app()
else:
    # Si se importa (por ejemplo, en un notebook), se inicia en un hilo secundario
    threading.Thread(target=run_app).start()
