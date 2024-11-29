# python native
import os
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
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
from transformers import UperNetForSemanticSegmentation

# model name
# fcn, unet(unetpp), deeplabv3p


############## PARSE ARGUMENT ########################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",      type=str,   default="deeplab")
    parser.add_argument("--tr_batch_size",   type=int,   default=2)
    parser.add_argument("--val_batch_size",  type=int,   default=1)
    parser.add_argument("--val_every",       type=int,   default=5)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--lr",              type=float, default=1e-6)
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--fold",            type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--pt_name",         type=str,   default="deeplabv3p_final_fold1_more.pt")

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
"""train_tf = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5), 
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),
    A.GridDistortion(distort_limit=0.2, p=0.3),
])

valid_tf = A.Compose([
    A.Resize(512, 512)
])
"""
class CustomAugmentation_final:
    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
    
    def get_transforms(self):
        if self.is_train:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=0.5),
                A.ElasticTransform(alpha=15.0, sigma=2.0, alpha_affine=25, p=0.4),
                A.GridDistortion(distort_limit=0.2, p=0.4),
                A.Rotate(limit=30, p=0.3),
                A.CLAHE(clip_limit=(1, 4), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(0.0, 0.3), contrast_limit=0.2, p=0.3),
                A.GridDropout(ratio=0.4, random_offset=False, holes_number_x=12, holes_number_y=12, p=0.2)
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
            ])

############### Dataset ###############
"""train_dataset = XRayDataset(is_train=True, transforms=train_tf, fold=FOLD)
valid_dataset = XRayDataset(is_train=False, transforms=valid_tf, fold=FOLD)"""

# train과 validation augmentation 인스턴스 생성
train_augmentation = CustomAugmentation_final(img_size=2048, is_train=True).get_transforms()
valid_augmentation = CustomAugmentation_final(img_size=2048, is_train=False).get_transforms()

fold_index = 1  # 실행할 Fold 번호 (0, 1, 2, 3, 4 중 하나)  

# 데이터셋에 augmentation 적용
train_dataset = XRayDataset(fold_idx=fold_index, is_train=True, transforms=train_augmentation)
valid_dataset = XRayDataset(fold_idx=fold_index, is_train=False, transforms=valid_augmentation)

train_groupnames = [os.path.basename(os.path.dirname(f)) for f in train_dataset.filenames]
print(f"  train groupnames: {train_groupnames}")

valid_groupnames = [os.path.basename(os.path.dirname(f)) for f in valid_dataset.filenames]
print(f"  Valid groupnames: {valid_groupnames}")

image, label = train_dataset[0]
print(image.shape, label.shape)


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

############### MODEL ###############
# 반복적으로 나오는 구조를 쉽게 만들기 위해서 정의한 유틸리티 함수 입니다
def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation=1):
        super().__init__()
        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2
        # TODO: depthwise conv - BN - pointwise conv로 구성된 레이어를 구현합니다
        self.depthwise = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.BN = nn.BatchNorm2d(num_features=in_ch)
        self.pointwise = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, bias=False) # bias=False는 권장 사항. 해도 되고 안 해도 됨.
        # 배치 정규화(Batch Normalization) 레이어가 뒤따르는 경우, 보통 bias=False를 설정합니다. 배치 정규화가 이미 편향과 유사한 역할을 수행하므로, 추가적인 bias 파라미터가 불필요해집니다.
        # bias=False를 설정하면 모델의 학습 파라미터 수가 줄어들고 학습 속도가 빨라지는 장점
        # bias=False가 필수는 아닙니다. 편향 값이 필요하다고 판단되면 bias=True로 설정해도 됩니다. 다만, 배치 정규화가 있을 때 bias를 추가하면 중복 효과가 발생하여 학습에 영향을 줄 수 있으므로 일반적으로는 bias=False를 사용하는 것이 권장
        
    def forward(self, x):
        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        x = self.depthwise(x)
        x = self.BN(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super().__init__()
        if in_ch != out_ch or stride !=1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = None

        if exit_flow:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, in_ch, 3, 1, dilation),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch)
            ]
        else:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch)
            ]

        if not use_1st_relu:
            block = block[1:]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = self.block(x)
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        x = output + skip
        return x


class Xception(nn.Module):
    def __init__(self, in_channels):
        super(Xception, self).__init__()
        self.entry_block_1 = nn.Sequential(
            conv_block(in_channels, 32, 3, 2, 1),
            conv_block(32, 64, 3, 1, 1, relu=False),
            XceptionBlock(64, 128, 2, 1, use_1st_relu=False)
        )
        self.relu = nn.ReLU()
        self.entry_block_2 = nn.Sequential(
            XceptionBlock(128, 256, 2, 1),
            XceptionBlock(256, 728, 2, 1)
        )

        middle_block = [XceptionBlock(728, 728, 1, 1) for _ in range(16)]
        self.middle_block = nn.Sequential(*middle_block)

        self.exit_block = nn.Sequential(
            XceptionBlock(728, 1024, 1, 1, exit_flow=True),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1024, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 2048, 3, 1, 2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.entry_block_1(x)
        features = out
        out = self.entry_block_2(out)
        out = self.middle_block(out)
        out = self.exit_block(out)
        return out, features


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # TODO: ASPP를 구성하는 모듈들을 작성합니다
        #    중간 피처맵의 채널 사이즈는 256을 사용합니다
        self.aspp1 = conv_block(in_ch, 256, k_size=1, stride=1, padding=0, dilation=1)
        self.aspp2 = conv_block(in_ch, 256, k_size=3, stride=1, padding=6, dilation=6)
        self.aspp3 = conv_block(in_ch, 256, k_size=3, stride=1, padding=12, dilation=12)
        self.aspp4 = conv_block(in_ch, 256, k_size=3, stride=1, padding=18, dilation=18)
        self.aspp5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels=in_ch, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv2d(256 * 5, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        # 각 ASPP 분기를 통해 특징을 추출하고 결합합니다.
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        aspp5 = self.aspp5(x)
        aspp5 = F.interpolate(aspp5, size=aspp4.size()[2:], mode="bilinear", align_corners=True)
        
        # 채널을 결합하여 최종 ASPP 출력을 만듭니다.
        output = torch.cat((aspp1, aspp2, aspp3, aspp4, aspp5), dim=1)
        output = self.output(output)
        return output


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = conv_block(128, 48, 1, 1, 0)
        self.block2 = nn.Sequential(
            conv_block(48+256, 256, 3, 1, 1),
            conv_block(256, 256, 3, 1, 1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, features):
        features = self.block1(features)
        feature_size = (features.shape[-1], features.shape[-2])

        out = F.interpolate(x, size=feature_size, mode="bilinear", align_corners=True)
        out = torch.cat((features, out), dim=1)
        out = self.block2(out)
        return out


class DeepLabV3p(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # TODO: DeepLabV3+ 모델을 완성하기 위해 필요한 모듈들을 작성하세요
        #   상단에서 작성한 backbone Xception 모델과 ASPP 및 decoder를 사용합니다
        #   ASPP에서 중간 피처맵 사이즈로 256을 사용했다는 점을 이용해야 합니다
        # Backbone으로 Xception 네트워크 사용
        self.backbone = Xception(in_channels)
        
        # ASPP 모듈 정의 (backbone의 마지막 출력 채널 수인 2048을 입력으로 받음)
        self.aspp = AtrousSpatialPyramidPooling(2048)
        
        # Decoder 정의
        self.decoder = Decoder(num_classes)

    def forward(self, x):

        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        # Backbone을 통해 특징 추출
        out, features = self.backbone(x)
        
        # ASPP 모듈 적용
        aspp_out = self.aspp(out)
        
        # Decoder를 통해 최종 출력 생성
        # ValueError: Target size (torch.Size([2, 29, 512, 512])) must be the same as input size (torch.Size([2, 29, 128, 128]))
        output = self.decoder(aspp_out, features)
        output = F.interpolate(output, scale_factor=4, mode="bilinear", align_corners=True)
        return output

############### TRAIN ###############
# Enable AMP by modifying the train and validation functions
def train(model, data_loader, val_loader, criterion, optimizer, scheduler):
    print(f'Start training with AMP enabled..')
    model.cuda()

    n_class = len(CLASSES)
    best_dice = 0.
    scaler = torch.cuda.amp.GradScaler()  # GradScaler for AMP # 기본값은 enabled=True

    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            
            # AMP enabled forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Scale the loss and backpropagate
            scaler.step(optimizer)
            scaler.update()

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                
        scheduler.step()
        
        # 현재 학습률 출력 및 기록
        current_lr = scheduler.get_last_lr()[0]
        log_message = f"Epoch {epoch + 1}, Current LR: {current_lr:.9f}"
        print(log_message)

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)


############### VALIDATION ###############
def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.cuda()
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()

            outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            ## Dice 계산과정을 gpu에서 진행하도록 변경
            #outputs = (outputs > thr).detach().cpu()
            #masks = masks.detach().cpu()
            outputs = (outputs > thr).float()  # Keep on GPU
            masks = masks.float()  # Ensure masks are float for dice computation
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
#model = DeepLabV3p(in_channels=3, num_classes=len(CLASSES))
model = torch.load("/data/ephemeral/home/level2-cv-semanticsegmentation-cv-10-lv3/code/checkpoints/result_deeplab/deeplabv3p_final_fold1.pt")
# Loss function
criterion = BCE_Dice_loss

# Loss function 정의
criterion = BCE_Dice_loss

optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-5) 

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-8)

# Scheduler: Cosine Annealing LR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-8)

# Set_seed
set_seed()

# train
train(model, train_loader, valid_loader, criterion, optimizer, scheduler)