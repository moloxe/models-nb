import os
import torch
import torch.nn as nn
from PIL import Image as PILImage
import torchvision.transforms as T
import torchvision.models as models

# Resolver paths relativos a la ubicación de este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "cnn_model.pth")
EXAMPLE_IMAGE_PATH = os.path.join(ASSETS_DIR, "interior.webp")

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Dispositivo para CNN: {DEVICE}")

CNN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def construir_cnn():
    bb = models.efficientnet_b0()
    bb.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    return bb

# Carga de modelo
modelo = construir_cnn()
modelo.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
modelo.to(DEVICE)
modelo.eval()

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
if __name__ == '__main__':
    resultado = predictCnn(EXAMPLE_IMAGE_PATH)
    print("Resultado del ejemplo de uso de CNN:")
    print(resultado)
