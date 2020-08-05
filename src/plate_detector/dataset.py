import os.path as osp
from typing import List, Tuple
from xml.etree import cElementTree

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class PlatesDetectionDataset(Dataset):

    def __init__(self, data_path: str, set_type: str):

        assert set_type in ['train', 'val', 'test'], 'set type must be one of ["train", "val", "test"]'

        self.data_path = data_path
        with open(osp.join(self.data_path, 'ImageSets', f'{set_type}.txt'), 'r') as f:
            self.file_names = [line.strip() for line in f]
        self.bboxes = [
            self._read_bboxes_data(osp.join(self.data_path, 'Annotations', f'{name}.xml'))  for name in self.file_names
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name, bboxes = self.file_names[idx], self.bboxes[idx]
        image = plt.imread(osp.join(self.data_path, 'Images', f'{name}.jpg'))
        labels = [1 for _ in range(len(bboxes))]

        target = {'boxes': bboxes, 'labels': labels}

        return image, target

    def _read_bboxes_data(self, xml_path: str) -> List[Tuple[int, int, int, int]]:
        xml = cElementTree.parse(xml_path)
        coords = []
        for bbox_iterator in xml.getiterator('bndbox'):
            # x0, y0, x1, y1
            coords.append(tuple(int(coord.text) for coord in bbox_iterator))
        return coords
