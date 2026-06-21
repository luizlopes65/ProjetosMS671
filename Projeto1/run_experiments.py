"""Run comparative experiments between YOLOv3 and YOLOv8."""

import torch
import os
from pathlib import Path
from glob import glob
from weights import carregar_pesos_ultralytics_v26, carregar_pesos_yolov3
from model import YOLOv3
from config import YOLOV3_ANCHORS
from utils import read_classes, preprocess_image, reverter_escala_caixas, manual_nms
from inference import decode_yolo, desenhar_deteccoes
import matplotlib.pyplot as plt


def criar_estrutura_pastas():
    """Create folder structure for experiments."""
    Path("exps/exp_v3").mkdir(parents=True, exist_ok=True)
    Path("exps/exp_v26").mkdir(parents=True, exist_ok=True)
    print("✓ Estrutura de pastas criada:")
    print("  - exps/exp_v3/")
    print("  - exps/exp_v26/")


def listar_imagens(pasta="images"):
    """List all images in specified folder."""
    extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    imagens = []
    for ext in extensoes:
        imagens.extend(glob(f"{pasta}/{ext}"))
        imagens.extend(glob(f"{pasta}/{ext.upper()}"))
    return sorted(imagens)


def executar_exp_v26(imagens, device='cpu'):
    """Run YOLOv8 experiment and save results."""
    print("\n" + "="*60)
    print("EXPERIMENTO: YOLOv26 (Ultralytics)")
    print("="*60 + "\n")
    
    # Carrega modelo
    modelo = carregar_pesos_ultralytics_v26('yolov8n.pt', device=device)
    modelo.conf = 0.25
    modelo.iou = 0.45
    
    print(f"Processando {len(imagens)} imagens...\n")
    
    for i, img_path in enumerate(imagens, 1):
        print(f"[{i}/{len(imagens)}] Processando: {img_path}")
        
        try:
            # Faz predição
            results = modelo.predict(
                source=img_path,
                save=False,
                conf=0.25,
                iou=0.45,
                verbose=False
            )
            
            # Salva resultado
            for r in results:
                # Obtém imagem com anotações
                im_array = r.plot()
                im_rgb = im_array[:, :, ::-1]  # BGR para RGB
                
                # Nome do arquivo de saída
                img_name = Path(img_path).name
                output_path = f"exps/exp_v26_hard/{img_name}"
                
                # Salva imagem
                plt.figure(figsize=(12, 8))
                plt.imshow(im_rgb)
                plt.axis('off')
                plt.title(f'YOLOv26 - {len(r.boxes)} detecções')
                plt.tight_layout()
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"  ✓ {len(r.boxes)} objetos detectados")
                print(f"  💾 Salvo em: {output_path}")
        
        except Exception as e:
            print(f"  ❌ Erro: {e}")
        
        print()
    
    print("="*60)
    print("✅ Experimento YOLOv26 concluído!")
    print(f"📁 Resultados em: exps/exp_v26/")
    print("="*60 + "\n")


def executar_exp_v3(imagens, device='cpu'):
    """Run YOLOv3 experiment and save results."""
    print("\n" + "="*60)
    print("EXPERIMENTO: YOLOv3 (Original)")
    print("="*60 + "\n")
    
    # Verifica se os pesos existem
    if not Path("yolov3_convertido.pth").exists():
        print("⚠️  Arquivo yolov3_convertido.pth não encontrado!")
        print("   Tentando converter de weights/yolov3.weights...")
        
        try:
            class_names = read_classes("data/coco.names")
            modelo = YOLOv3(num_classes=len(class_names)).to(device)
            modelo = carregar_pesos_yolov3("weights/yolov3.weights", modelo)
            torch.save(modelo.state_dict(), "yolov3_convertido.pth")
            print("✓ Pesos convertidos com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao converter pesos: {e}")
            print("   Pulando experimento YOLOv3...")
            return
    
    # Carrega modelo
    try:
        class_names = read_classes("data/coco.names")
        modelo = YOLOv3(num_classes=len(class_names)).to(device)
        modelo.load_state_dict(torch.load("yolov3_convertido.pth", map_location=device))
        modelo.eval()
        print("✓ Modelo YOLOv3 carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("   Pulando experimento YOLOv3...")
        return
    
    print(f"Processando {len(imagens)} imagens...\n")
    
    for i, img_path in enumerate(imagens, 1):
        print(f"[{i}/{len(imagens)}] Processando: {img_path}")
        
        try:
            plt.ioff()
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
                mask = box_class_scores >= 0.5
                
                boxes = all_boxes[mask]
                scores = box_class_scores[mask]
                classes = box_classes[mask]
                
                if boxes.size(0) > 0:
                    boxes = reverter_escala_caixas(boxes, (416, 416), image.size)
                    keep = manual_nms(boxes, scores, classes, 0.4)
                    keep = keep[:10]
                    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
                    image = desenhar_deteccoes(image, boxes, scores, classes, class_names)
                    num_detections = len(boxes)
                else:
                    num_detections = 0
            img_name = Path(img_path).name
            output_path = f"exps/exp_v3_hard/{img_name}"
            image.save(output_path)
            
            print(f"  ✓ {num_detections} objetos detectados")
            print(f"  💾 Salvo em: {output_path}")
        
        except Exception as e:
            print(f"  ❌ Erro: {e}")
        
        print()
    
    print("="*60)
    print("✅ Experimento YOLOv3 concluído!")
    print(f"📁 Resultados em: exps/exp_v3/")
    print("="*60 + "\n")


def main():
    """Main function to run experiments."""
    print("\n" + "="*60)
    print("EXECUTANDO EXPERIMENTOS COMPARATIVOS")
    print("YOLOv3 vs YOLOv26 (Ultralytics)")
    print("="*60 + "\n")
    
    # Cria estrutura de pastas
    criar_estrutura_pastas()
    
    # Lista imagens
    imagens = listar_imagens("images/hard_images")
    
    if not imagens:
        print("❌ Nenhuma imagem encontrada na pasta 'images/'")
        print("   Adicione imagens (.jpg, .png, etc.) na pasta 'images/' e tente novamente.")
        return
    
    print(f"\n📸 Encontradas {len(imagens)} imagens:")
    for img in imagens:
        print(f"  - {img}")
    print()
    
    # Detecta dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Dispositivo: {device}\n")
    
    # Executa experimentos
    executar_exp_v26(imagens, device)
    executar_exp_v3(imagens, device)
    
    print("\n" + "="*60)
    print("🎉 TODOS OS EXPERIMENTOS CONCLUÍDOS!")
    print("="*60)
    print("\n📁 Resultados salvos em:")
    print("  - exps/exp_v3/  (YOLOv3 original)")
    print("  - exps/exp_v26/ (YOLOv26 ultralytics)")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

   
