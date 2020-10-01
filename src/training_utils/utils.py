import random

import numpy as np
import torch


def fix_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def resnet_freeze(model, trainable_layers: int) -> None:
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    for name, param in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
