import numpy as np
import torch
import torch.nn as nn
from model import ConvBlock
from ultralytics import YOLO
from pathlib import Path


def carregar_pesos_yolov3(caminho_weights, modelo):
    """Load YOLOv3 weights from binary .weights file."""
    print(f"Loading weights from: {caminho_weights}")
    with open(caminho_weights, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    modulos = []
    for m in modelo.modules():
        if isinstance(m, ConvBlock) or (isinstance(m, nn.Conv2d) and m.bias is not None):
            modulos.append(m)

    ptr = 0
    for i, modulo in enumerate(modulos):
        if isinstance(modulo, ConvBlock):
            conv, bn = modulo.conv, modulo.bn
            num_bn = bn.bias.numel()

            bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.bias))
            ptr += num_bn

            bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.weight))
            ptr += num_bn

            bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.running_mean))
            ptr += num_bn

            bn_var = torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.running_var)
            bn.running_var.data.copy_(torch.clamp(bn_var, min=1e-5))
            ptr += num_bn
            num_w = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv.weight))
            ptr += num_w

        elif isinstance(modulo, nn.Conv2d):
            num_b = modulo.bias.numel()
            modulo.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b]).view_as(modulo.bias))
            ptr += num_b

            num_w = modulo.weight.numel()
            modulo.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_w]).view_as(modulo.weight))
            ptr += num_w

    print(f"Success! Loaded {ptr} parameters")
    return modelo


def carregar_pesos_ultralytics_v26(caminho_weights, modelo=None, device='cpu'):
    """Load YOLOv8 weights using ultralytics."""
    yolo_model = YOLO(caminho_weights)
    yolo_model.to(device)
    
    if modelo is None:
        return yolo_model
    else:
        modelo = transferir_pesos_ultralytics(yolo_model, modelo)
        return modelo


def transferir_pesos_ultralytics(yolo_ultralytics, modelo_customizado):
    """Transfer weights from ultralytics model to custom model."""
    ultralytics_state = yolo_ultralytics.model.state_dict()
    custom_state = modelo_customizado.state_dict()
    
    transferred = 0
    for name, param in custom_state.items():
        if name in ultralytics_state:
            if param.shape == ultralytics_state[name].shape:
                custom_state[name] = ultralytics_state[name]
                transferred += 1
    
    modelo_customizado.load_state_dict(custom_state)
    print(f"✓ Transferidos {transferred} parâmetros com sucesso")
    return modelo_customizado

def extrair_backbone_ultralytics(caminho_weights, device='cpu'):
    """Extract backbone from ultralytics model."""
    
    yolo_model = YOLO(caminho_weights)
    yolo_model.to(device)
    
    backbone = yolo_model.model.model[:10]
    return backbone


def salvar_pesos_compativel_ultralytics(modelo, caminho_saida):
    """Save weights in ultralytics-compatible format."""
    print(f"Saving weights to: {caminho_saida}")
    
    checkpoint = {
        'model': modelo.state_dict(),
        'epoch': 0,
        'best_fitness': 0.0,
        'date': None,
    }
    
    torch.save(checkpoint, caminho_saida)
    print("✓ Pesos salvos com sucesso")
