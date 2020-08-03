from typing import Optional
import torch

from src.cars_detector.model import SSD300


def load_model(backbone: str = 'resnet50', checkpoint_path: Optional[str] = '/tmp/cars_detection.pth', 
    device: torch.device = torch.device('cuda')) -> SSD300:

    if backbone not in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        raise ValueError(f'Unknown backbone {backbone}')

    model = SSD300(backbone=backbone)
    model.to(device)
    model.eval()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
    
    return model

