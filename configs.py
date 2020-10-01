import torch


class Config:

    device: torch.device = torch.device("cuda")


class CarsDetectionConfig:

    idx2cls = {i: "not_car" for i in range(80)}
    idx2cls[0] = "background"
    idx2cls[3] = "car"

    cls2idx = {cls: idx for idx, cls in idx2cls.items()}


config = Config()
cars_det_config = CarsDetectionConfig()
