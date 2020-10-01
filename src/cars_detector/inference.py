from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from skimage import img_as_float, io, transform

from configs import config
from src.cars_detector.model import SSD300
from src.cars_detector.utils import Encoder, dboxes300_coco


class Processing:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
        img = img_as_float(io.imread(image_path))
        if len(img.shape) == 2:
            img = np.array([img, img, img]).swapaxes(0, 2)
        return img

    @staticmethod
    def rescale(img, input_height, input_width):
        """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
        aspect = img.shape[1] / float(img.shape[0])
        if aspect > 1:
            # landscape orientation - wide image
            res = int(aspect * input_height)
            imgScaled = transform.resize(img, (input_width, res))
        if aspect < 1:
            # portrait orientation - tall image
            res = int(input_width / aspect)
            imgScaled = transform.resize(img, (res, input_height))
        if aspect == 1:
            imgScaled = transform.resize(img, (input_width, input_height))
        return imgScaled

    @staticmethod
    def crop_center(img, cropx, cropy) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
        y, x, c = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx], (startx, starty)

    @staticmethod
    def normalize(img, mean=128, std=128):
        img = (img * 256 - mean) / std
        return img

    @staticmethod
    def prepare_tensor(inputs):
        NHWC = np.array(inputs)
        NCHW = np.swapaxes(np.swapaxes(NHWC, 1, 3), 2, 3)
        tensor = torch.from_numpy(NCHW)
        tensor = tensor.cuda()
        tensor = tensor.float()
        return tensor

    @classmethod
    def prepare_input(cls, img_uri) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img = cls.load_image(img_uri)
        orig_size = img.shape[:-1]
        img = cls.rescale(img, 300, 300)
        scaled_size = img.shape[:-1]
        img, offsets = cls.crop_center(img, 300, 300)
        img = cls.normalize(img)

        info = {"orig_size": orig_size, "scaled_size": scaled_size, "center_crop_offsets": offsets}
        return img, info

    @staticmethod
    def decode_results(predictions):
        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)
        ploc, plabel = [val.float() for val in predictions]
        results = encoder.decode_batch(ploc, plabel, criteria=0.5, max_output=20)
        return [[pred.detach().cpu().numpy() for pred in detections] for detections in results]

    @staticmethod
    def pick_best(detections, threshold=0.3):
        bboxes, classes, confidences = detections
        best = np.argwhere(confidences > threshold)[:, 0]
        return [pred[best] for pred in detections]

    @staticmethod
    def correct_bbox(
        bbox: Tuple[float, float, float, float], transform_info: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        orig_h, orig_w = transform_info["orig_size"]
        scaled_h, scaled_w = transform_info["scaled_size"]
        offset_x, offset_y = transform_info["center_crop_offsets"]

        left, bot, right, top = bbox
        x, y, w, h = [coord * 300 for coord in [left, bot, right - left, top - bot]]
        x, y = x + offset_x, y + offset_y
        x, y, w, h = x / scaled_w * orig_w, y / scaled_h * orig_h, w / scaled_w * orig_w, h / scaled_h * orig_h
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        return x0, y0, x1, y1


def load_model(
    backbone: str = "resnet50",
    checkpoint_path: Optional[str] = "/tmp/cars_detection.pth",
    device: torch.device = config.device,
) -> SSD300:

    if backbone not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        raise ValueError(f"Unknown backbone {backbone}")

    model = SSD300(backbone=backbone)
    model.to(device)
    model.eval()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])

    return model


def prepare_images(image_paths: List[str], device: torch.device = config.device) -> Tuple[torch.Tensor, Dict[str, Any]]:
    images, infos = zip(*[Processing.prepare_input(image) for image in image_paths])
    images = Processing.prepare_tensor(images).to(device)
    return images, infos


@torch.no_grad()
def run_network(
    model: SSD300, input: torch.Tensor, transform_info: Dict[str, Any], threshold: float = 0.6
) -> torch.Tensor:
    predictions = model(input)
    predictions = Processing.decode_results(predictions)
    corrected_predictions = []
    for prediction, info in zip(predictions, transform_info):
        prediction = Processing.pick_best(prediction, threshold)
        bboxes, classes, probs = prediction
        corrected_bboxes = [Processing.correct_bbox(bbox, info) for bbox in bboxes]
        corrected_predictions.append((corrected_bboxes, classes, probs))
    return corrected_predictions
