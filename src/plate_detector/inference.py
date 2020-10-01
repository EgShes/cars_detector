from typing import List, Optional

import numpy as np
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from configs import config


def load_model(
    backbone: str = "resnet50",
    checkpoint_path: Optional[str] = "/tmp/cars_detection.pth",
    device: torch.device = config.device,
) -> FasterRCNN:

    if backbone not in ["resnet50"]:
        raise ValueError(f"Unknown backbone {backbone}")

    backbone = resnet_fpn_backbone(backbone, pretrained=False)
    model = FasterRCNN(backbone, num_classes=2)
    model.to(device)
    model.eval()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model


def prepare_images(images: List[np.ndarray], device: torch.device = config.device) -> List[torch.Tensor]:
    images = [torch.FloatTensor(img.copy()).permute(2, 0, 1).to(device) / 255.0 for img in images]
    return images


@torch.no_grad()
def run_model(model: FasterRCNN, images: List[torch.Tensor], threshold: float = 0.5):

    predictions = model(images)
    result = []
    for prediction in predictions:
        boxes, scores = prediction["boxes"], prediction["scores"]
        if len(boxes) != 0:
            boxes, scores = zip(*[(box, score) for box, score in zip(boxes, scores) if score > threshold])
        boxes = [box.detach().cpu().numpy().astype(np.int32) for box in boxes]
        scores = [score.item() for score in scores]
        result.append({"boxes": boxes, "scores": scores})

    return result
