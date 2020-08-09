import os
import os.path as osp
import uuid

import hydra
import torch
from albumentations import (CLAHE, Compose, HorizontalFlip, JpegCompression,
                            OneOf, RandomBrightnessContrast, ShiftScaleRotate,
                            ToGray)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from src.plate_detector.dataset import PlatesDetectionDataset
from src.training_utils import RAdam, fix_seeds, write2tensorboard


def train_epoch(model, loader: DataLoader, optimizer: RAdam, device: torch.device, epoch_n: int):
    model.train()

    epoch_loss = 0.
    for images, targets in tqdm(loader, desc='{epoch_n} training', total=len(loader)):
        images = [image.to(device) for image in images]
        targets = [{'boxes': target['boxes'].to(device), 'labels': target['labels'].to(device)} for target in targets]

        model.zero_grad()

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate_epoch(model, loader: DataLoader, device: torch.device, epoch_n: int):
    # only in train mode loss is returned
    model.train()

    epoch_loss = 0.
    for images, targets in tqdm(loader, desc='{epoch_n} evaluating', total=len(loader)):
        images = [image.to(device) for image in images]
        targets = [{'boxes': target['boxes'].to(device), 'labels': target['labels'].to(device)} for target in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        epoch_loss += loss.item()

    return epoch_loss / len(loader)



@hydra.main('train_config.yaml')
def train_model(args):
    
    fix_seeds(args.seed)

    device = torch.device(args.device)

    experiment_name = f'{args.experiment_name}_{uuid.uuid4()}'
    if not args.test_run:
        writer = SummaryWriter(log_dir=osp.join('runs', experiment_name))

    os.makedirs(osp.join(args.save_path, args.experiment_name), exist_ok=True)

    transforms = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, rotate_limit=10, p=0.5),
        RandomBrightnessContrast(p=0.5),
        JpegCompression(),
        OneOf([
            ToGray(p=1),
            CLAHE(p=1)
        ])
    ])

    train_ds = PlatesDetectionDataset(args.data_path, 'train', transforms)
    val_ds = PlatesDetectionDataset(args.data_path, 'val', None)
    test_ds = PlatesDetectionDataset(args.data_path, 'test', None)

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=PlatesDetectionDataset.custom_collate)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.num_workers, collate_fn=PlatesDetectionDataset.custom_collate)
    test_dl = DataLoader(test_ds, args.batch_size, num_workers=args.num_workers, collate_fn=PlatesDetectionDataset.custom_collate)

    model = fasterrcnn_resnet50_fpn(num_classes=2, pretrained_backbone=True, trainable_backbone_layers=1)
    model.to(device)

    optimizer = RAdam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    best_loss, no_improvements = float('inf'), 0
    try:
        for epoch in range(args.num_epochs):
            if no_improvements > args.early_stopping:
                break

            train_metrics = train_epoch(model, train_dl, optimizer, device, epoch)
            eval_metrics = evaluate_epoch(model, val_dl, device, epoch)

            if eval_metrics['loss'] < best_loss:
                best_loss = eval_metrics['loss']
                no_improvements = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'experiment_name': args.experiment_name,
                    'writer_path': osp.join('runs', experiment_name)
                }, osp.join(args.save_path, args.experiment_name, f'{experiment_name}_best.pth'))
            else:
                no_improvements += 1

            lr_scheduler.step(eval_metrics['loss'])

            if not args.test_run:
                write2tensorboard(train_metrics, eval_metrics, writer, epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    except KeyboardInterrupt:
        pass

    checkpoint = torch.load(osp.join(args.save_path, args.experiment_name, f'{experiment_name}_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_epoch(model, test_dl, device, 0)
    if not args.test_run:
        write2tensorboard(test_metrics, writer, test=True)


if __name__ == '__main__':
    train_model()
