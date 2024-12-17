import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import set_seed, collate_fn
from configs import config as cfg

from dataset.XRayDataset import *
from models import *
from loss import *

import segmentation_models_pytorch as smp

from dataset.augmentation import get_train_transforms, get_val_transforms
from trainer import train_model


def main():
    # Set seed
    set_seed(cfg.RANDOM_SEED)

    # Init transforms, dataset
    trian_tf = get_train_transforms(cfg.IMAGE_SIZE)
    val_tf = get_val_transforms(cfg.IMAGE_SIZE)

    train_dataset = XRayDataset(is_train=True, transforms=trian_tf, fold=cfg.FOLD, all = cfg.ALL_DATA, crop=cfg.CROP_CHANGE)
    valid_dataset = XRayDataset(is_train=False, transforms=val_tf, fold=cfg.FOLD, all = cfg.ALL_DATA, crop=cfg.CROP_CHANGE)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        **({"collate_fn": collate_fn} if cfg.SLIDING else {})
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        **({"collate_fn": collate_fn} if cfg.SLIDING else {})
    )

    # Init model
    if cfg.MODEL == 'unetpp':
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(CLASSES),
        ).cuda()
    elif cfg.MODEL == 'conv':
        model = UperNet_ConvNext_xlarge(num_classes=len(CLASSES))
    elif cfg.MODEL == 'segformer':
        model = SegFormer_B0(num_classes=len(CLASSES))
    elif cfg.MODEL == 'deeplap':
        model = DeepLabV3p(in_channels=3, num_classes=len(CLASSES))
    else:
        print("model not found.")
        exit(0)

    if cfg.MODEL.lower() == "fcn":
        model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

    # loss, optimizer, scheduler
    criterion = BCE_Dice_loss
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-8)

    # Start training
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler)


if __name__ == "__main__":
    main()
