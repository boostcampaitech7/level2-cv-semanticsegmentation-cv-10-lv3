# python native
import os
import random
import datetime

# external library
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A
import argparse

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# model, dataset
from dataset.XRayDataset import *
from loss.FocalLoss import FocalLoss
from model import *

from torch.cuda.amp import GradScaler, autocast

# model name
# fcn, unet(unetpp), deeplabv3p


############## PARSE ARGUMENT ########################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",      type=str,   default="unet")
    parser.add_argument("--tr_batch_size",   type=int,   default=1)
    parser.add_argument("--val_batch_size",  type=int,   default=1)
    parser.add_argument("--val_every",       type=int,   default=5)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--fold",            type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=21)
    parser.add_argument("--pt_name",         type=str,   default="unet3p_focal3dice7_1024.pt")

    args = parser.parse_args()
    return args


args = parse_args()


############### TRAINING SETTINGS ###############
MODEL = args.model_name
PT_NAME = args.pt_name
FOLD = args.fold
RANDOM_SEED = args.seed
TRAIN_BATCH_SIZE = args.tr_batch_size
VAL_BATCH_SIZE = args.val_batch_size
VAL_EVERY = args.val_every
NUM_EPOCHS = args.epochs
LR = args.lr
TH = args.threshold
SAVED_DIR = f"./checkpoints/result_{MODEL}"

if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)


############### Augmentation ###############
train_tf = A.Compose([
    A.Resize(1024, 1024),
    A.HorizontalFlip(p=0.5), 
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),
    A.GridDistortion(distort_limit=0.2, p=0.3),
])

valid_tf = A.Compose([
    A.Resize(1024, 1024)
])


############### Dataset ###############
train_dataset = XRayDataset(is_train=True, transforms=train_tf, fold=FOLD)
valid_dataset = XRayDataset(is_train=False, transforms=valid_tf, fold=FOLD)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    drop_last=False,
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=False
)


############### METHODS ###############
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

# focal loss 
def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    """
    Focal Loss 계산.
    - inputs: 모델의 raw logits (sigmoid를 적용하지 않은 상태로 입력)
    - targets: Ground truth
    """
    # BCEWithLogitsLoss와 유사하게 logits을 입력으로 받아 처리
    BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1 - BCE_EXP) ** gamma * BCE
    return loss

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def BCE_Dice_loss(pred, target, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

# Focal + Dice Loss 정의
def focal_dice_loss(pred, target, focal_weight=0.3, dice_weight=0.7, alpha=0.25, gamma=2, smooth=1.0):
    """
    Focal Loss와 Dice Loss를 결합한 Loss 함수입니다.
    - pred: 모델 예측값 (raw logits, sigmoid를 거치지 않은 상태)
    - target: 실제 정답 값
    """
    # Focal Loss 계산
    focal = focal_loss(pred, target, alpha=alpha, gamma=gamma)
    
    # Dice Loss 계산 (sigmoid를 명시적으로 적용)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target, smooth=smooth)
    
    # Weighted Sum
    loss = focal * focal_weight + dice * dice_weight
    return loss

def save_model(model, file_name=PT_NAME):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


############### TRAIN ###############
def train(model, data_loader, val_loader, criterion, optimizer, scheduler):
    print(f'Start training..')

    n_class = len(CLASSES)
    best_dice = 0.
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            optimizer.zero_grad()
            
            # Mixed precision: forward pass under autocast
            with autocast():
                if MODEL.lower() == "fcn":
                    outputs = model(images)['out']
                elif MODEL.lower() == "unet":
                    outputs = model(images)
                elif MODEL.lower() == "deeplabv3p":
                    outputs = model(images)

                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)

                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear", align_corners=True)

                loss = criterion(outputs, masks)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )

        # Update the learning rate with the scheduler
        scheduler.step()
        
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)


############### VALIDATION ###############
def validation(epoch, model, data_loader, criterion, thr=TH):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            if MODEL.lower() == "fcn":
                outputs = model(images)['out']
            elif MODEL.lower() == "unet":
                outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    return avg_dice


############### TRAINING SETTINGS 2 ###############

# model
# model = models.segmentation.fcn_resnet50(pretrained=True) # fcn base
# model = UNet(num_classes=len(CLASSES)) # unet base
# model = UNetPlusPlus(out_ch=len(CLASSES), supervision=False) # unet++ base
# model = DeepLabV3p(in_channels=3, num_classes=len(CLASSES))  # deeplabv3p base
model = UNet_3Plus(in_channels=3, n_classes=len(CLASSES))

# output class 개수를 dataset에 맞도록 수정합니다.
if MODEL.lower() == "fcn":
    model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

# Loss function
# criterion = nn.BCEWithLogitsLoss()  # fcn, unet, unet ++ base, deeplab3vp base
# criterion = FocalLoss(alpha=1, gamma=2, reduction='mean') # focal loss
# criterion = BCE_Dice_loss
criterion = focal_dice_loss

# Optimizer: AdamW
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=LR,  # 학습률
    weight_decay=1e-4  # 가중치 감쇠
)
# optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6) # fcn base
# unet base, unet ++ base, deeplab3vp base
# optimizer = optim.RMSprop(params=model.parameters(), lr=LR, weight_decay=1e-6)

# Scheduler: Cosine Annealing LR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=NUM_EPOCHS,  # 전체 학습 에폭 수
    eta_min=1e-6  # 최소 학습률
)

# Set_seed
set_seed()

# train
train(model, train_loader, valid_loader, criterion, optimizer, scheduler)
