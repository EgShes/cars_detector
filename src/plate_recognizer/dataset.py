import os.path as osp
from typing import List, Optional, Tuple

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from configs import PlatesRecognitionConfig
from src.plate_recognizer.utils import letterbox_image
from torch.utils.data import Dataset


class PlatesDataset(Dataset):

    def __init__(
            self, data_path: str, set_type: str, resize_hw: Tuple[int, int], 
            transforms: Optional[albumentations.core.composition.Compose]
        ):
        
        assert set_type in ['train', 'val', 'test'], 'set type must be one of ["train", "val", "test"]'

        self.resize_hw = resize_hw
        self.data_path = data_path
        self.df = pd.read_csv(osp.join(self.data_path, f'{set_type}.csv'))

        self.transforms = transforms
        self.normalize = albumentations.Normalize()
        self.resize = letterbox_image
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[int]]:
        name, text = self.df.iloc[idx]
        image = plt.imread(osp.join(self.data_path, 'images', name))[:,:,:3]

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        image = self.normalize(image=image)['image']
        image = self.resize(image, *self.resize_hw)

        text = [PlatesRecognitionConfig.char2idx[ch] for ch in text]
        text = [PlatesRecognitionConfig.char2idx['SOS']] + text + [PlatesRecognitionConfig.char2idx['EOS']]

        image, text = torch.FloatTensor(image).permute(2, 0, 1), torch.LongTensor(text)

        return image, text

    @staticmethod
    def custom_collate(batch):
        images, texts = zip(*batch)

        images = torch.stack(images)

        max_l = max([len(text) for text in texts])
        texts = [
            torch.cat(
                [text, torch.LongTensor([PlatesRecognitionConfig.char2idx['PAD']] * (max_l - len(text)))], -1
            ) for text in texts
        ]
        texts = torch.stack(texts)

        return images, texts

