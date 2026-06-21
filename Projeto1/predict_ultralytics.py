"""
Script de predição simplificado usando ultralytics YOLOv26

Este script demonstra como fazer predições usando o modelo ultralytics
de forma simples e direta.
"""

import torch
from pathlib import Path
from weights import carregar_pesos_ultralytics_v26, listar_camadas_ultralytics


def predict_with_ultralytics(image_path, model_path='yolov8n.pt', conf_threshold=0.25, iou_threshold=0.45, save_results=True):
    

    modelo = carregar_pesos_ultralytics_v26(model_path, device=device)
    
    modelo.conf = conf_threshold
    modelo.iou = iou_threshold
    

    results = modelo.predict(
        source=image_path,
        save=save_results,
        conf=conf_threshold,
        iou=iou_threshold,
        show=False,  # Não mostra automaticamente
        verbose=True
    )

    
    for r in results:
        boxes = r.boxes
        
        if len(boxes) == 0:
            print("Nenhum objeto")
            continue
        
        print(f"Detectados {len(boxes)} objetos:\n")
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = modelo.names[cls_id]
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            print(f"  {i+1}. {class_name}")
            print(f"     Confiança: {conf:.2%}")
            print(f"     Caixa: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            print()
    
    return results


def predict_multiple_images(image_paths, model_path='yolov8n.pt', conf_threshold=0.25):

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo = carregar_pesos_ultralytics_v26(model_path, device=device)
    modelo.conf = conf_threshold
    
    for img_path in image_paths:
        if not Path(img_path).exists():
            print(f"⚠️  Pulando: {img_path} (não encontrado)")
            continue
        
        results = modelo.predict(source=img_path, save=True, verbose=False)
        
        for r in results:
            boxes = r.boxes
            print(f"   ✓ {len(boxes)} objetos detectados")



def predict_with_visualization(image_path, model_path='yolov8n.pt', conf_threshold=0.25):
    """
    Faz predição e mostra a imagem com as detecções.
    
    Args:
        image_path: Caminho para a imagem
        model_path: Caminho para os pesos do modelo
        conf_threshold: Threshold de confiança
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    print("\n" + "="*60)
    print("PREDIÇÃO COM VISUALIZAÇÃO")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo = carregar_pesos_ultralytics_v26(model_path, device=device)
    modelo.conf = conf_threshold
    
    # Faz predição
    results = modelo.predict(source=image_path, save=False, verbose=True)
    
    # Plota resultados
    for r in results:
        # Obtém a imagem com as caixas desenhadas
        im_array = r.plot()  # BGR numpy array
        
        # Converte BGR para RGB
        im_rgb = im_array[:, :, ::-1]
        
        # Mostra com matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.title(f'Detecções: {len(r.boxes)} objetos')
        plt.tight_layout()
        plt.show()
    
    print("="*60 + "\n")


def list_available_models():
    """Lista modelos YOLO disponíveis para download."""
    print("\n" + "="*60)
    print("MODELOS YOLO DISPONÍVEIS")
    print("="*60 + "\n")
    
    models = {
        'YOLOv8 Nano': 'yolov8n.pt',
        'YOLOv8 Small': 'yolov8s.pt',
        'YOLOv8 Medium': 'yolov8m.pt',
        'YOLOv8 Large': 'yolov8l.pt',
        'YOLOv8 XLarge': 'yolov8x.pt',
        'YOLOv11 Nano': 'yolo11n.pt',
        'YOLOv11 Small': 'yolo11s.pt',
        'YOLOv11 Medium': 'yolo11m.pt',
        'YOLOv11 Large': 'yolo11l.pt',
        'YOLOv11 XLarge': 'yolo11x.pt',
    }
    
    print("Modelos disponíveis para download automático:\n")
    for name, filename in models.items():
        print(f"  • {name:20s} → {filename}")
    
    print("\n" + "="*60)
    print("Use qualquer um desses nomes como model_path")
    print("O ultralytics fará o download automaticamente se necessário")
    print("="*60 + "\n")


def main():
    """Função principal com exemplos de uso."""
    print("\n" + "="*60)
    print("SCRIPT DE PREDIÇÃO - ULTRALYTICS YOLOv26")
    print("="*60 + "\n")
    
    # Lista modelos disponíveis
    list_available_models()
    
    # Exemplo 1: Predição simples
    print("\n📌 EXEMPLO 1: Predição simples")
    print("-" * 60)
    
    # Descomente e ajuste o caminho da imagem:
    # predict_with_ultralytics('images/dog.jpg', model_path='yolov8n.pt', conf_threshold=0.25)
    
    # Exemplo 2: Predição com visualização
    print("\n📌 EXEMPLO 2: Predição com visualização")
    print("-" * 60)
    
    # Descomente e ajuste o caminho da imagem:
    # predict_with_visualization('images/dog.jpg', model_path='yolov8n.pt', conf_threshold=0.25)
    
    # Exemplo 3: Predição em múltiplas imagens
    print("\n📌 EXEMPLO 3: Predição em lote")
    print("-" * 60)
    
    # Descomente e ajuste os caminhos:
    # images = ['images/dog.jpg', 'images/food.jpg', 'images/street.jpg']
    # predict_multiple_images(images, model_path='yolov8n.pt', conf_threshold=0.25)
    
    # Exemplo 4: Listar arquitetura do modelo
    print("\n📌 EXEMPLO 4: Inspecionar arquitetura do modelo")
    print("-" * 60)
    
    # Descomente para ver a estrutura:
    # listar_camadas_ultralytics('yolov8n.pt')
    
    print("\n" + "="*60)
    print("Para executar os exemplos, descomente as linhas correspondentes")
    print("e ajuste os caminhos das imagens conforme necessário.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

   
