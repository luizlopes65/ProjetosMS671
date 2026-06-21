import random
import colorsys
import numpy as np
import torch
from PIL import Image


def read_classes(classes_path):
    """Read class names from file."""
    with open(classes_path) as f:
        return [c.strip() for c in f.readlines()]


def generate_colors(class_names):
    """Generate distinct colors for each class using HSV."""
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)
    random.shuffle(colors)
    random.seed(None)
    return colors


def letterbox_image(image, size=(416, 416)):
    """Resize image maintaining aspect ratio with gray padding."""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def reverter_escala_caixas(boxes, img_size, original_shape):
    """Remove letterbox effect and scale boxes to original image size."""
    iw, ih = original_shape
    w, h = img_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    dx = (w - nw) / 2.0 / w
    dy = (h - nh) / 2.0 / h
    scale_w = nw / w
    scale_h = nh / h

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dy) / scale_h
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dx) / scale_w

    boxes[:, [0, 2]] *= ih
    boxes[:, [1, 3]] *= iw

    return boxes


def scale_boxes(boxes, image_shape):
    """Scale boxes to image dimensions."""
    height, width = image_shape
    image_dims = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes.device)
    return boxes * image_dims


def preprocess_image(img_path, model_image_size=(416, 416)):
    """Preprocess image for model input."""
    image = Image.open(img_path).convert('RGB')
    boxed_image = letterbox_image(image, model_image_size)

    image_data = np.array(boxed_image, dtype='float32') / 255.0
    image_data = image_data[:, :, ::-1].copy()

    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = torch.from_numpy(image_data).unsqueeze(0)
    return image, image_data


def manual_nms(boxes, scores, classes, iou_threshold):
    """
    Manual Non-Maximum Suppression implementation.
    
    Args:
        boxes: Bounding boxes [N, 4] in format [y_min, x_min, y_max, x_max]
        scores: Confidence scores [N]
        classes: Class indices [N]
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep: Tensor of indices to keep
    """
    keep = []
    order = scores.argsort(descending=True)
    while (len(order) > 0):
        idx = order[0]
        keep.append(idx.item())
        if (len(order) == 1):
            break
        order = order[1:]
        classe_campea = classes[idx]
        campea = boxes[idx]
        classe_restantes = classes[order]
        restantes = boxes[order]

        xx1 = torch.max(campea[0], restantes[:, 0])
        yy1 = torch.max(campea[1], restantes[:, 1])
        xx2 = torch.min(campea[2], restantes[:, 2])
        yy2 = torch.min(campea[3], restantes[:, 3])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        area_intersecao = w * h
        area_campea = (campea[2] - campea[0]) * (campea[3] - campea[1])
        area_restantes = (restantes[:, 2] - restantes[:, 0]) * (restantes[:, 3] - restantes[:, 1])
        uniao = area_campea + area_restantes - area_intersecao
        iou = area_intersecao / uniao
        mask = (iou <= iou_threshold) | (classe_campea != classe_restantes)
        order = order[mask]

    return torch.tensor(keep, dtype=torch.long)

 
