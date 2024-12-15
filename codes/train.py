import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import set_seed
import config as cfg

from dataset.XRayDataset import *
from dataset.XRayDatasetAll import *

import segmentation_models_pytorch as smp

from augmentation import get_train_transforms, get_val_transforms
from trainer import train_model, BCE_Dice_loss


def main():
    # Set seed
    set_seed(cfg.RANDOM_SEED)

    # Init transforms, dataset
    trian_tf = get_train_transforms(cfg.IMAGE_SIZE)
    val_tf = get_val_transforms(cfg.IMAGE_SIZE)

    train_dataset = XRayDataset(is_train=True, transforms=trian_tf, fold=cfg.FOLD, all = cfg.ALL_DATA)
    valid_dataset = XRayDataset(is_train=False, transforms=val_tf, fold=cfg.FOLD, all = cfg.ALL_DATA)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # Init model
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=29,
    ).cuda()

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
