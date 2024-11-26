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
import segmentation_models_pytorch as smp

# loss
# from segmentation_models_pytorch.losses import DiceLoss

# model name
# fcn, unet(unetpp), deeplabv3p

############## PARSE ARGUMENT ########################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",      type=str,   default="unet")
    parser.add_argument("--img_size",        type=int,   default=1024)
    parser.add_argument("--tr_batch_size",   type=int,   default=2)
    parser.add_argument("--val_batch_size",  type=int,   default=1)
    parser.add_argument("--val_every",       type=int,   default=10)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--fold",            type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=21)
    parser.add_argument("--pt_name",         type=str,   default="unetPP_smp_1024_eff.pt")
    parser.add_argument("--log_name",         type=str,   default="unetPP_smp_1024_eff")

    args = parser.parse_args()
    return args


args = parse_args()


############### TRAINING SETTINGS ###############
MODEL = args.model_name
IMAGE_SIZE = args.img_size
PT_NAME = args.pt_name
LOG_NAEM = args.log_name
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
trian_tf = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=0.5),
    A.ElasticTransform(alpha=15.0, sigma=2.0, p=0.4),
    A.GridDistortion(distort_limit=0.2, p=0.4),
    A.Rotate(limit=30, p=0.3),
    A.CLAHE(clip_limit=(1, 4), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(0.0, 0.3), contrast_limit=0.2, p=0.3),
    A.GridDropout(ratio=0.4, random_offset=False, holes_number_x=12, holes_number_y=12, p=0.4)
])

val_tf = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
])

############### Dataset ###############
train_dataset = XRayDataset(is_train=True, transforms=trian_tf, fold=FOLD)
valid_dataset = XRayDataset(is_train=False, transforms=val_tf, fold=FOLD)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True,
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

# print 파일 기록하기 위한 함수


def log_to_file(message, file_path=f"./log/{LOG_NAEM}.txt"):
    with open(file_path, "a") as f:
        f.write(message + "\n")


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def BCE_Dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

############### TRAIN ###############


def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')

    n_class = len(CLASSES)
    best_dice = 0.

    # 스케줄러 정의
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-8)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

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

            # Loss 계산 및 최적화
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 25 == 0:
                log_message = (
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                print(log_message)
                log_to_file(log_message)

        # 학습률 스케줄러 업데이트
        scheduler.step()

        # 현재 학습률 출력 및 기록
        current_lr = scheduler.get_last_lr()[0]
        if epoch % 10 == 0:
            log_message = f"Epoch {epoch + 1}, Current LR: {current_lr:.6f}"
            print(log_message)
            log_to_file(log_message)

        # Validation
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)

            if best_dice < dice:
                log_message = (
                    f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}\n"
                    f"Save model in {SAVED_DIR}"
                )
                print(log_message)
                log_to_file(log_message)
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
            elif MODEL.lower() == "deeplabv3p":
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
    log_to_file(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    return avg_dice


############### TRAINING SETTINGS 2 ###############
# smp
model = smp.Unet(
    encoder_name="efficientnet-b7",  # EfficientNet-L2 백본 사용
    encoder_weights="imagenet",     # 사전 학습된 가중치 설정
    in_channels=3,                       # 입력 채널 (RGB 이미지)
    classes=29                           # 출력 클래스 수
)


# output class 개수를 dataset에 맞도록 수정합니다.
if MODEL.lower() == "fcn":
    model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)


criterion = BCE_Dice_loss


optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-5)

# Set_seed
set_seed()

# train
train(model, train_loader, valid_loader, criterion, optimizer)
