import torch
from config import YOLOV3_ANCHORS
from utils import read_classes
from model import YOLOv3
from weights import carregar_pesos_yolov3
from inference import executar_predicao, executar_predicao_ultralytics


def main():
    """Run YOLOv3 predictions on test images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_names = read_classes("data/coco.names")
    print(f"Loaded {len(class_names)} classes")

    modelo = YOLOv3(num_classes=len(class_names)).to(device)

    # Convert weights on first run
    try:
        arquivo_weights = "weights/yolov3.weights"
        modelo = carregar_pesos_yolov3(arquivo_weights, modelo)
        torch.save(modelo.state_dict(), "yolov3_convertido.pth")
        print("Weights converted and saved")
    except Exception as e:
        print(f"Conversion warning: {e}")

    try:
        modelo.load_state_dict(torch.load("yolov3_convertido.pth", map_location=device))
        print("Weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    print("\nRunning predictions...\n")
    
    executar_predicao("images/dog.jpg", modelo, class_names, YOLOV3_ANCHORS, device, score_threshold=0.5)
    executar_predicao("images/food.jpg", modelo, class_names, YOLOV3_ANCHORS, device)


def main_ultralytics():
    """Run YOLOv8 predictions on test images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = 'yolov8n.pt'

    images = [
        "images/dog.jpg",
        "images/food.jpg",
        "images/cat.jpg",
        "images/dog2.jpg",
        "images/wine.jpg",
        "images/city_scene.jpg",
        "images/eagle.jpg",
        "images/giraffe.jpg",
        "images/horses.jpg",
        "images/motorbike.jpg",
        "images/person.jpg",
        "images/surf.jpg"
    ]
    
    for img in images:
        executar_predicao_ultralytics(
            img,
            model_path=model_path,
            score_threshold=0.25,
            iou_threshold=0.45,
            device=device
        )


if __name__ == "__main__":
    main_ultralytics()
