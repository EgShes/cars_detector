import torch


class Config:

    device: torch.device = torch.device("cuda")


class CarsDetectionConfig:

    idx2cls = {i: "not_car" for i in range(80)}
    idx2cls[0] = "background"
    idx2cls[3] = "car"

    cls2idx = {cls: idx for idx, cls in idx2cls.items()}


class PlatesRecognitionConfig:

    alphabet = [ 
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', 'SOS', 'EOS', 'PAD'
    ]
    char2idx = {char: i for i, char in enumerate(alphabet)}
    idx2char = {i: char for char, i in char2idx.items()}


config = Config()
cars_det_config = CarsDetectionConfig()
