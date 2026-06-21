import torch
import os
import time
import numpy as np
from pathlib import Path
from glob import glob

# Importações do seu projeto
from weights import carregar_pesos_ultralytics_v26, carregar_pesos_yolov3
from model import YOLOv3
from config import YOLOV3_ANCHORS
from utils import read_classes, preprocess_image, reverter_escala_caixas, manual_nms
from inference import decode_yolo

# --- CONFIGURAÇÕES DOS EXPERIMENTOS ---
GRID_PESQUISA = [
    {'conf': 0.25, 'iou': 0.45},
    {'conf': 0.50, 'iou': 0.45},
    {'conf': 0.50, 'iou': 0.50},
    {'conf': 0.25, 'iou': 0.25},
    {'conf': 0.50, 'iou': 0.75},
    {'conf': 0.75, 'iou': 0.50},
    {'conf': 0.75, 'iou': 0.75},
]

def listar_imagens(pasta="images"):
    extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    imagens = []
    for ext in extensoes:
        imagens.extend(glob(f"{pasta}/{ext}"))
    return sorted(imagens)

def executar_exp_v26(imagens, device, conf, iou):
    """Executa YOLOv26 e retorna (tempo_médio, detecções_médias, confiança_média)."""
    modelo = carregar_pesos_ultralytics_v26('yolov8n.pt', device=device)
    tempos, detecoes, confiancas = [], [], []
    
    for img_path in imagens:
        start = time.time()
        results = modelo.predict(source=img_path, conf=conf, iou=iou, verbose=False)
        tempos.append(time.time() - start)
        
        # Extrai scores de confiança
        scores = results[0].boxes.conf.cpu().numpy()
        detecoes.append(len(scores))
        if len(scores) > 0:
            confiancas.append(np.mean(scores))

    return np.mean(tempos), np.mean(detecoes), np.mean(confiancas) if confiancas else 0.0

def executar_exp_v3(imagens, device, conf, iou):
    """Executa YOLOv3 e retorna (tempo_médio, detecções_médias, confiança_média)."""
    class_names = read_classes("data/coco.names")
    modelo = YOLOv3(num_classes=len(class_names)).to(device)
    if Path("yolov3_convertido.pth").exists():
        modelo.load_state_dict(torch.load("yolov3_convertido.pth", map_location=device))
    modelo.eval()

    tempos, detecoes, confiancas = [], [], []

    for img_path in imagens:
        start = time.time()
        image, image_data = preprocess_image(img_path, (416, 416))
        image_data = image_data.to(device)

        with torch.no_grad():
            out1, out2, out3 = modelo(image_data)
            b1, s1 = decode_yolo(out1, YOLOV3_ANCHORS[0], len(class_names), 416)
            b2, s2 = decode_yolo(out2, YOLOV3_ANCHORS[1], len(class_names), 416)
            b3, s3 = decode_yolo(out3, YOLOV3_ANCHORS[2], len(class_names), 416)
            
            all_scores = torch.cat([s1, s2, s3], dim=0)
            box_class_scores, _ = torch.max(all_scores, dim=-1)
            
            # Filtro de confiança aplicado aqui para a métrica
            mask = box_class_scores >= conf
            scores_filtrados = box_class_scores[mask].cpu().numpy()
            
            # Nota: Para simplificar a métrica de precisão por confiança, 
            # usamos os scores antes do NMS ou após o filtro de máscara.
            detecoes.append(len(scores_filtrados))
            if len(scores_filtrados) > 0:
                confiancas.append(np.mean(scores_filtrados))
        
        tempos.append(time.time() - start)

    return np.mean(tempos), np.mean(detecoes), np.mean(confiancas) if confiancas else 0.0

def imprimir_tabela_final(resultados):
    print("\n" + "="*110)
    print("📊 COMPARATIVO DETALHADO: YOLOv3 vs YOLOv26".center(110))
    print("="*110)
    header = (f"{'Config (C/I)':<15} | {'Modelo':<8} | {'Tempo (s)':<12} | "
              f"{'Det. Médias':<12} | {'Confianca Méd.':<15}")
    print(header)
    print("-" * 110)
    
    for r in resultados:
        # Note o uso de aspas duplas por fora para aceitar aspas simples dentro do dicionário r['conf']
        config_str = f"C:{r['conf']} I:{r['iou']}"
        
        # Linha YOLOv3
        print(f"{config_str:<15} | {'V3':<8} | {r['v3_t']:<12.4f} | "
              f"{r['v3_d']:<12.1f} | {r['v3_c']:<15.2%}")
        
        # Linha YOLOv26
        print(f"{'':<15} | {'V26':<8} | {r['v26_t']:<12.4f} | "
              f"{r['v26_d']:<12.1f} | {r['v26_c']:<15.2%}")
        print("-" * 110)

def main():
    imagens = listar_imagens("images")
    if not imagens: return print("❌ Nenhuma imagem em 'images/'")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Dispositivo: {device} | Imagens: {len(imagens)}\n")
    
    resultados_finais = []

    for param in GRID_PESQUISA:
        c, i = param['conf'], param['iou']
        print(f"🔄 Processando Grid: Conf={c}, IOU={i}...")
        
        v26_t, v26_d, v26_c = executar_exp_v26(imagens, device, c, i)
        v3_t, v3_d, v3_c = executar_exp_v3(imagens, device, c, i)
        
        resultados_finais.append({
            'conf': c, 'iou': i,
            'v26_t': v26_t, 'v26_d': v26_d, 'v26_c': v26_c,
            'v3_t': v3_t, 'v3_d': v3_d, 'v3_c': v3_c
        })

    imprimir_tabela_final(resultados_finais)

if __name__ == "__main__":
    main()
