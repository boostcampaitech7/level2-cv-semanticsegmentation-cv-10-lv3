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
from dataset.XRayDatasetAll import *
from loss.FocalLoss import FocalLoss
from model import *
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler

# loss
# from segmentation_models_pytorch.losses import DiceLoss

# model name
# fcn, unet(unetpp), deeplabv3p

############## PARSE ARGUMENT ########################
def parse_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--model_name",      type=str,   default="unet")
    parser.add_argument("--img_size",        type=int,   default=512)
    parser.add_argument("--tr_batch_size",   type=int,   default=3)
    parser.add_argument("--val_batch_size",  type=int,   default=1)
    parser.add_argument("--val_every",       type=int,   default=10)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--fold",            type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=21)
    parser.add_argument("--pt_name",         type=str,   default="unetPP_effcientnetl2.pt")
    parser.add_argument("--log_name",         type=str,   default="unetPP_effcientnetl2")
    
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

class CustomCenterCrop(A.ImageOnlyTransform):
    def __init__(self, height, width, shift_down_ratio, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.shift_down_ratio = shift_down_ratio  # 아래로 이동 비율

    def apply(self, img, **params):
        h, w, _ = img.shape
        # 중심 좌표 계산
        center_x = w // 2
        center_y = int(h // 2 + h * self.shift_down_ratio)  # 아래로 이동된 중심
        # 자를 영역 계산
        x1 = max(center_x - self.width // 2, 0)
        y1 = max(center_y - self.height // 2, 0)
        x2 = x1 + self.width
        y2 = y1 + self.height

        # 원본 이미지 크기 초과 방지
        x2 = min(x2, w)
        y2 = min(y2, h)
        x1 = max(x2 - self.width, 0)  # 자른 영역이 항상 width에 맞도록
        y1 = max(y2 - self.height, 0)  # 자른 영역이 항상 height에 맞도록
        
        cropped_img = img[y1:y2, x1:x2]
        
        # 크기 확인 디버깅
        # print(f"Original size: ({h}, {w}), Crop size: ({cropped_img.shape[0]}, {cropped_img.shape[1]})")
        
        return cropped_img



trian_tf = A.Compose([        
        # A.OneOf([  # 변환 중 하나를 선택
        #     CustomCenterCrop(height=512, width=512, shift_down_ratio=0.2),
        # ], p=0.5),  
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
            ),
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5)
    ])


val_tf = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        # A.ElasticTransform(p=0.2),
        # A.Sharpen()
    ])

############### Dataset ###############
train_dataset = XRayDatasetAll(is_train=True, transforms=trian_tf, fold=FOLD)
valid_dataset = XRayDatasetAll(is_train=False, transforms=val_tf, fold=FOLD)


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
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

# print 파일 기록하기 위한 함수
def log_to_file(message, file_path=f"./log/{LOG_NAEM}.txt"):
    with open(file_path, "a") as f:
        f.write(message + "\n")

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


# from torch.cuda.amp import autocast, GradScaler

def train(model, data_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_loss = float('inf')  # Loss를 기준으로 모델 저장
    model = model.cuda()

    # GradScaler 초기화
    scaler = GradScaler()

    # 스케줄러 정의
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-8)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0  # 전체 loss 계산
        
        for step, (images, masks) in enumerate(data_loader):            
            images, masks = images.cuda(), masks.cuda()
            
            optimizer.zero_grad()  # Optimizer 초기화
            
            with autocast():  # Mixed Precision Training 적용
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
                
                # 손실 계산
                loss = criterion(outputs, masks)
            
            # Scaler를 사용하여 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if (step + 1) % 25 == 0:
                log_message = (
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(), 4)}'
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

        # 현재 epoch의 평균 loss 계산
        avg_loss = total_loss / len(data_loader)
        log_message = f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}"
        print(log_message)
        log_to_file(log_message)
        
        # Loss를 기준으로 모델 저장
        if avg_loss < best_loss:
            log_message = (
                f"Best performance at epoch: {epoch + 1}, Loss {best_loss:.4f} -> {avg_loss:.4f}\n"
                f"Save model in {SAVED_DIR}"
            )
            print(log_message)
            log_to_file(log_message)
            best_loss = avg_loss
            save_model(model)






############### TRAINING SETTINGS 2 ###############

# smp
model = smp.UnetPlusPlus(    
    encoder_name="timm-efficientnet-l2",  # efficientnet-b7 timm-efficientnet-l2
    encoder_weights="noisy-student-475",     # 사전 학습된 가중치 설정
    in_channels=3,                       # 입력 채널 (RGB 이미지)
    classes=29                           # 출력 클래스 수
)


# output class 개수를 dataset에 맞도록 수정합니다.
if MODEL.lower() == "fcn":
    model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

# Loss function
criterion = BCE_Dice_loss

# Optimizer
optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-5)

# Set_seed
set_seed()

# train
train(model, train_loader, criterion, optimizer)
