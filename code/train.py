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


############## PARSE ARGUMENT ########################
def parse_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--model_name",      type=str,   default="UNET")
    parser.add_argument("--tr_batch_size",   type=int,   default=4)
    parser.add_argument("--val_batch_size",  type=int,   default=1)
    parser.add_argument("--val_every",       type=int,   default=5)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--fold",            type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=21)
    parser.add_argument("--pt_name",         type=str,   default="fnc_base.pt")
    
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
tf = A.Compose([
        A.Resize(512, 512),
        # A.ElasticTransform(p=0.2),
        # A.Sharpen()
    ])


############### Dataset ###############
train_dataset = XRayDataset(is_train=True, transforms=tf, fold=FOLD)
valid_dataset = XRayDataset(is_train=False, transforms=tf, fold=FOLD)


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


############### TRAIN ###############
def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for step, (images, masks) in enumerate(data_loader):            
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            if MODEL.lower() == "fcn":
                outputs = model(images)['out']
            elif MODEL.lower() == "unet":
                outputs = model(images)
            
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
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
model = UNetPlusPlus(out_ch=len(CLASSES), supervision=False) # unet ++ base


# output class 개수를 dataset에 맞도록 수정합니다.
if MODEL.lower() == "fcn":
    model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

# Loss function
criterion = nn.BCEWithLogitsLoss() # fcn, unet, unet ++ base
# criterion = FocalLoss(alpha=1, gamma=2, reduction='mean') # focal loss

# Optimizer
# optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6) # fcn base
optimizer = optim.RMSprop(params=model.parameters(), lr=LR, weight_decay=1e-6) # unet base, unet ++ base

# Set_seed
set_seed()

# train
train(model, train_loader, valid_loader, criterion, optimizer)

