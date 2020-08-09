import random
from time import sleep

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


def write2tensorboard(train_metrics, val_metrics, writer, epoch: int = 0):
    for key in train_metrics:
        train_val, val_val = train_metrics[key], val_metrics[key]
    writer.add_scalar(f'{key}/{key}_train', train_val, epoch)
    writer.add_scalar(f'{key}/{key}_val', val_val, epoch)


def write2tensorboard_test(test_metrics, writer, epoch: int = 0):
    for key, val in test_metrics.items():
        writer.add_scalar(f'{key}/{key}_test', val, epoch)
    # to let the writer write the metrics
    sleep(10.)
