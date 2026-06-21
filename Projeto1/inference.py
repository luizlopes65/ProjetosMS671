import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils import preprocess_image, reverter_escala_caixas, generate_colors, manual_nms


def decode_yolo(feats, anchors, num_classes, img_size=416):
    """
    Decode raw YOLO outputs into boxes and scores.
    
    Args:
        feats: Raw feature map from YOLO head
        anchors: Anchor boxes for this scale
        num_classes: Number of object classes
        img_size: Input image size
    
    Returns:
        boxes: Decoded bounding boxes [N, 4]
        scores: Class scores [N, num_classes]
    """
    B, C, H, W = feats.shape
    num_anchors = len(anchors)
    feats = feats.view(B, num_anchors, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()

    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float().to(feats.device).view(1, 1, H, W, 2)

    # XY (Centro)
    box_xy = torch.sigmoid(feats[..., :2])
    box_xy = (box_xy + grid) / torch.tensor([W, H], dtype=torch.float32, device=feats.device)

    # WH (Largura e Altura)
    anchors_tensor = torch.tensor(anchors, dtype=torch.float32, device=feats.device).view(1, num_anchors, 1, 1, 2)
    box_wh = torch.exp(torch.clamp(feats[..., 2:4], max=15.0)) * anchors_tensor
    box_wh = box_wh / img_size # Normaliza para 0 a 1

    # Confiança e Classes (YOLOv3 usa sigmoid para as classes também)
    box_confidence = torch.sigmoid(feats[..., 4:5])
    box_class_probs = torch.sigmoid(feats[..., 5:])

    # Converte (x, y, w, h) para (esquerda, topo, direita, baixo)
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    boxes = torch.cat([box_mins[..., 1:2], box_mins[..., 0:1], box_maxes[..., 1:2], box_maxes[..., 0:1]], dim=-1)

    scores = box_confidence * box_class_probs
    return boxes.view(-1, 4), scores.view(-1, num_classes)


def desenhar_deteccoes(image, boxes, scores, classes, class_names):
    """
    Desenha as detecções em uma imagem.
    
    Args:
        image: PIL Image
        boxes: Tensor de caixas [N, 4] no formato [y_min, x_min, y_max, x_max]
        scores: Tensor de scores [N]
        classes: Tensor de índices de classes [N]
        class_names: Lista com nomes das classes
    
    Returns:
        image: PIL Image com as detecções desenhadas
    """
    colors = generate_colors(class_names)
    font = ImageFont.load_default()
    thickness = (image.size[0] + image.size[1]) // 300
    
    for i, c in reversed(list(enumerate(classes.cpu().numpy()))):
        predicted_class = class_names[c]
        box = boxes[i].cpu().numpy()
        score = scores[i].cpu().item()
        
        label = f'{predicted_class} {score:.2f}'
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        
        if left >= right or top >= bottom:
            continue
        
        text_origin = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array([left, top + 1])
        
        for j in range(thickness):
            if left+j >= right-j or top+j >= bottom-j:
                break
            draw.rectangle([left+j, top+j, right-j, bottom-j], outline=colors[c])
        
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
        del draw
    
    return image


def executar_predicao(image_file, model, class_names, anchors, device, score_threshold=0.5, iou_threshold=0.4):
    """
    Run complete prediction pipeline on an image.
    
    Args:
        image_file: Path to input image
        model: YOLOv3 model
        class_names: List of class names
        anchors: Anchor boxes for all scales
        device: torch device
        score_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
    """
    print(f"Processing: {image_file}")
    image, image_data = preprocess_image(image_file, (416, 416))
    image_data = image_data.to(device)

    model.eval()
    with torch.no_grad():
        out1, out2, out3 = model(image_data)

        # Decodifica as três escalas e junta os tensores
        b1, s1 = decode_yolo(out1, anchors[0], len(class_names), 416)
        b2, s2 = decode_yolo(out2, anchors[1], len(class_names), 416)
        b3, s3 = decode_yolo(out3, anchors[2], len(class_names), 416)

        all_boxes = torch.cat([b1, b2, b3], dim=0)
        all_scores = torch.cat([s1, s2, s3], dim=0)

        # Limpa tensores
        valid_mask = torch.isfinite(all_boxes).all(dim=-1) & torch.isfinite(all_scores).all(dim=-1)
        all_boxes = all_boxes[valid_mask]
        all_scores = all_scores[valid_mask]

        # Filtragem por Confiança
        box_class_scores, box_classes = torch.max(all_scores, dim=-1)
        mask = box_class_scores >= score_threshold

        boxes = all_boxes[mask]
        scores = box_class_scores[mask]
        classes = box_classes[mask]

        if boxes.size(0) == 0:
            print("No objects detected")
            return

        boxes = reverter_escala_caixas(boxes, (416, 416), image.size)
        keep = manual_nms(boxes, scores, classes, iou_threshold)
        keep = keep[:10]  # Top 10 detections
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    image = desenhar_deteccoes(image, boxes, scores, classes, class_names)

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

 

def executar_predicao_ultralytics(image_file, model_path='yolov8n.pt', score_threshold=0.25, iou_threshold=0.45, device='cpu'):
    """
    Run YOLOv8 prediction using ultralytics.
    
    Args:
        image_file: Path to input image
        model_path: Path to YOLOv8 weights
        score_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        device: torch device
    
    Returns:
        results: Ultralytics prediction results
    """
    from weights import carregar_pesos_ultralytics_v26
    
    modelo = carregar_pesos_ultralytics_v26(model_path, device=device)
    modelo.conf = score_threshold
    modelo.iou = iou_threshold
    
    results = modelo.predict(
        source=image_file,
        save=True,
        conf=score_threshold,
        iou=iou_threshold,
        show=False,
        verbose=True
    )
    
    for r in results:
        im_array = r.plot()
        im_rgb = im_array[:, :, ::-1]
        
        plt.figure(figsize=(12, 12))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.title(f'Detecções: {len(r.boxes)} objetos')
        plt.tight_layout()
        plt.show()
    
    return results
