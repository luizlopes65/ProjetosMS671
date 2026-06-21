"""
Script para executar experimentos com diferentes parâmetros de confiança e IoU
Testa YOLOv3 e YOLOv26 com diferentes configurações
"""

import torch
import os
from pathlib import Path
from glob import glob
from weights import carregar_pesos_ultralytics_v26, carregar_pesos_yolov3
from model import YOLOv3
from config import YOLOV3_ANCHORS
from utils import read_classes
import matplotlib.pyplot as plt


def criar_estrutura_pastas():
    """Cria a estrutura de pastas para os experimentos."""
    # Experimentos de confiança
    for conf in [0.25, 0.5, 0.75]:
        Path(f"exps/exp_v3/conf_{conf}").mkdir(parents=True, exist_ok=True)
        Path(f"exps/exp_v26/conf_{conf}").mkdir(parents=True, exist_ok=True)
    
    # Experimentos de IoU
    for iou in [0.25, 0.5, 0.75]:
        Path(f"exps/exp_v3/iou_{iou}").mkdir(parents=True, exist_ok=True)
        Path(f"exps/exp_v26/iou_{iou}").mkdir(parents=True, exist_ok=True)
    
    print("✓ Estrutura de pastas criada")


def listar_imagens(pasta="images"):
    """Lista todas as imagens na pasta especificada."""
    extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    imagens = []
    for ext in extensoes:
        imagens.extend(glob(f"{pasta}/{ext}"))
        imagens.extend(glob(f"{pasta}/{ext.upper()}"))
    return sorted(imagens)


def executar_v26_conf_sweep(imagens, device='cpu'):
    """Testa YOLOv26 com diferentes valores de confiança."""
    print("\n" + "="*60)
    print("EXPERIMENTO: YOLOv26 - Variação de Confiança")
    print("="*60 + "\n")
    
    conf_values = [0.25, 0.5, 0.75]
    iou_fixed = 0.45
    
    for conf in conf_values:
        print(f"\n--- Testando conf={conf}, iou={iou_fixed} ---\n")
        
        modelo = carregar_pesos_ultralytics_v26('yolov8n.pt', device=device)
        modelo.conf = conf
        modelo.iou = iou_fixed
        
        for i, img_path in enumerate(imagens, 1):
            print(f"[{i}/{len(imagens)}] {Path(img_path).name}", end=" ")
            
            try:
                results = modelo.predict(
                    source=img_path,
                    save=False,
                    conf=conf,
                    iou=iou_fixed,
                    verbose=False
                )
                
                for r in results:
                    im_array = r.plot()
                    im_rgb = im_array[:, :, ::-1]
                    
                    img_name = Path(img_path).name
                    output_path = f"exps/exp_v26/conf_{conf}/{img_name}"
                    
                    plt.figure(figsize=(12, 8))
                    plt.imshow(im_rgb)
                    plt.axis('off')
                    plt.title(f'YOLOv26 - conf={conf}, iou={iou_fixed} - {len(r.boxes)} detecções')
                    plt.tight_layout()
                    plt.savefig(output_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    print(f"→ {len(r.boxes)} detecções")
            
            except Exception as e:
                print(f"→ Erro: {e}")
    
    print(f"\n✅ Experimento YOLOv26 (conf sweep) concluído!")


def executar_v26_iou_sweep(imagens, device='cpu'):
    """Testa YOLOv26 com diferentes valores de IoU."""
    print("\n" + "="*60)
    print("EXPERIMENTO: YOLOv26 - Variação de IoU")
    print("="*60 + "\n")
    
    conf_fixed = 0.25
    iou_values = [0.25, 0.5, 0.75]
    
    for iou in iou_values:
        print(f"\n--- Testando conf={conf_fixed}, iou={iou} ---\n")
        
        modelo = carregar_pesos_ultralytics_v26('yolov8n.pt', device=device)
        modelo.conf = conf_fixed
        modelo.iou = iou
        
        for i, img_path in enumerate(imagens, 1):
            print(f"[{i}/{len(imagens)}] {Path(img_path).name}", end=" ")
            
            try:
                results = modelo.predict(
                    source=img_path,
                    save=False,
                    conf=conf_fixed,
                    iou=iou,
                    verbose=False
                )
                
                for r in results:
                    im_array = r.plot()
                    im_rgb = im_array[:, :, ::-1]
                    
                    img_name = Path(img_path).name
                    output_path = f"exps/exp_v26/iou_{iou}/{img_name}"
                    
                    plt.figure(figsize=(12, 8))
                    plt.imshow(im_rgb)
                    plt.axis('off')
                    plt.title(f'YOLOv26 - conf={conf_fixed}, iou={iou} - {len(r.boxes)} detecções')
                    plt.tight_layout()
                    plt.savefig(output_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    print(f"→ {len(r.boxes)} detecções")
            
            except Exception as e:
                print(f"→ Erro: {e}")
    
    print(f"\n✅ Experimento YOLOv26 (iou sweep) concluído!")


def executar_v3_conf_sweep(imagens, device='cpu'):
    """Testa YOLOv3 com diferentes valores de confiança."""
    print("\n" + "="*60)
    print("EXPERIMENTO: YOLOv3 - Variação de Confiança")
    print("="*60 + "\n")
    
    # Carrega modelo uma vez
    if not Path("yolov3_convertido.pth").exists():
        print("⚠️  yolov3_convertido.pth não encontrado. Pulando YOLOv3...")
        return
    
    try:
        class_names = read_classes("data/coco.names")
        modelo = YOLOv3(num_classes=len(class_names)).to(device)
        modelo.load_state_dict(torch.load("yolov3_convertido.pth", map_location=device))
        modelo.eval()
    except Exception as e:
        print(f"❌ Erro ao carregar YOLOv3: {e}")
        return
    
    conf_values = [0.25, 0.5, 0.75]
    iou_fixed = 0.45
    
    for conf in conf_values:
        print(f"\n--- Testando conf={conf}, iou={iou_fixed} ---\n")
        
        for i, img_path in enumerate(imagens, 1):
            print(f"[{i}/{len(imagens)}] {Path(img_path).name}", end=" ")
            
            try:
                num_detections = processar_v3_imagem(
                    img_path, modelo, class_names, conf, iou_fixed, 
                    f"exps/exp_v3/conf_{conf}", device
                )
                print(f"→ {num_detections} detecções")
            except Exception as e:
                print(f"→ Erro: {e}")
    
    print(f"\n✅ Experimento YOLOv3 (conf sweep) concluído!")


def executar_v3_iou_sweep(imagens, device='cpu'):
    """Testa YOLOv3 com diferentes valores de IoU."""
    print("\n" + "="*60)
    print("EXPERIMENTO: YOLOv3 - Variação de IoU")
    print("="*60 + "\n")
    
    # Carrega modelo uma vez
    if not Path("yolov3_convertido.pth").exists():
        print("⚠️  yolov3_convertido.pth não encontrado. Pulando YOLOv3...")
        return
    
    try:
        class_names = read_classes("data/coco.names")
        modelo = YOLOv3(num_classes=len(class_names)).to(device)
        modelo.load_state_dict(torch.load("yolov3_convertido.pth", map_location=device))
        modelo.eval()
    except Exception as e:
        print(f"❌ Erro ao carregar YOLOv3: {e}")
        return
    
    conf_fixed = 0.25
    iou_values = [0.25, 0.5, 0.75]
    
    for iou in iou_values:
        print(f"\n--- Testando conf={conf_fixed}, iou={iou} ---\n")
        
        for i, img_path in enumerate(imagens, 1):
            print(f"[{i}/{len(imagens)}] {Path(img_path).name}", end=" ")
            
            try:
                num_detections = processar_v3_imagem(
                    img_path, modelo, class_names, conf_fixed, iou, 
                    f"exps/exp_v3/iou_{iou}", device
                )
                print(f"→ {num_detections} detecções")
            except Exception as e:
                print(f"→ Erro: {e}")
    
    print(f"\n✅ Experimento YOLOv3 (iou sweep) concluído!")


def processar_v3_imagem(img_path, modelo, class_names, conf_threshold, iou_threshold, output_dir, device):
    """Processa uma imagem com YOLOv3 e salva o resultado."""
    from utils import preprocess_image, reverter_escala_caixas, generate_colors, manual_nms
    from PIL import Image, ImageDraw, ImageFont
    from inference import decode_yolo
    import numpy as np
    
    image, image_data = preprocess_image(img_path, (416, 416))
    image_data = image_data.to(device)
    
    with torch.no_grad():
        out1, out2, out3 = modelo(image_data)
        
        b1, s1 = decode_yolo(out1, YOLOV3_ANCHORS[0], len(class_names), 416)
        b2, s2 = decode_yolo(out2, YOLOV3_ANCHORS[1], len(class_names), 416)
        b3, s3 = decode_yolo(out3, YOLOV3_ANCHORS[2], len(class_names), 416)
        
        all_boxes = torch.cat([b1, b2, b3], dim=0)
        all_scores = torch.cat([s1, s2, s3], dim=0)
        
        valid_mask = torch.isfinite(all_boxes).all(dim=-1) & torch.isfinite(all_scores).all(dim=-1)
        all_boxes = all_boxes[valid_mask]
        all_scores = all_scores[valid_mask]
        
        box_class_scores, box_classes = torch.max(all_scores, dim=-1)
        mask = box_class_scores >= conf_threshold
        
        boxes = all_boxes[mask]
        scores = box_class_scores[mask]
        classes = box_classes[mask]
        
        if boxes.size(0) > 0:
            boxes = reverter_escala_caixas(boxes, (416, 416), image.size)
            keep = manual_nms(boxes, scores, classes, iou_threshold)
            keep = keep[:10]
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
            
            # Desenha detecções
            colors = generate_colors(class_names)
            font = ImageFont.load_default()
            thickness = (image.size[0] + image.size[1]) // 300
            
            for i_box, c in enumerate(classes.cpu().numpy()):
                predicted_class = class_names[c]
                box = boxes[i_box].cpu().numpy()
                score = scores[i_box].cpu().item()
                
                label = f'{predicted_class} {score:.2f}'
                draw = ImageDraw.Draw(image)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                
                top, left, bottom, right = box
                top = max(0, int(top))
                left = max(0, int(left))
                bottom = min(image.size[1], int(bottom))
                right = min(image.size[0], int(right))
                
                if left < right and top < bottom:
                    text_origin = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array([left, top + 1])
                    
                    for j in range(thickness):
                        if left+j < right-j and top+j < bottom-j:
                            draw.rectangle([left+j, top+j, right-j, bottom-j], outline=colors[c])
                    
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                    draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
                del draw
            
            num_detections = len(boxes)
        else:
            num_detections = 0
    
    # Salva imagem
    img_name = Path(img_path).name
    output_path = f"{output_dir}/{img_name}"
    image.save(output_path)
    
    return num_detections


def main():
    """Função principal."""
    print("\n" + "="*60)
    print("EXPERIMENTOS DE PARÂMETROS")
    print("Testando diferentes valores de confiança e IoU")
    print("="*60 + "\n")
    
    # Cria estrutura de pastas
    criar_estrutura_pastas()
    
    # Lista imagens
    imagens = listar_imagens("images")
    
    if not imagens:
        print("❌ Nenhuma imagem encontrada na pasta 'images/'")
        return
    
    print(f"\n📸 Encontradas {len(imagens)} imagens")
    
    # Detecta dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Dispositivo: {device}\n")
    
    # Executa experimentos
    print("="*60)
    print("Iniciando experimentos...")
    print("="*60)
    
    # YOLOv26 experiments
    executar_v26_conf_sweep(imagens, device)
    executar_v26_iou_sweep(imagens, device)
    
    # YOLOv3 experiments
    executar_v3_conf_sweep(imagens, device)
    executar_v3_iou_sweep(imagens, device)
    
    print("\n" + "="*60)
    print("🎉 TODOS OS EXPERIMENTOS CONCLUÍDOS!")
    print("="*60)
    print("\n📁 Resultados salvos em:")
    print("  YOLOv3:")
    print("    - exps/exp_v3/conf_0.25/")
    print("    - exps/exp_v3/conf_0.5/")
    print("    - exps/exp_v3/conf_0.75/")
    print("    - exps/exp_v3/iou_0.25/")
    print("    - exps/exp_v3/iou_0.5/")
    print("    - exps/exp_v3/iou_0.75/")
    print("\n  YOLOv26:")
    print("    - exps/exp_v26/conf_0.25/")
    print("    - exps/exp_v26/conf_0.5/")
    print("    - exps/exp_v26/conf_0.75/")
    print("    - exps/exp_v26/iou_0.25/")
    print("    - exps/exp_v26/iou_0.5/")
    print("    - exps/exp_v26/iou_0.75/")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

   
