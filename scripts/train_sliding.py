# python native
import os
import json
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

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

# 데이터 경로를 입력하세요

IMAGE_ROOT = "data/train/DCM"
LABEL_ROOT = "data/train/outputs_json"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

IND2CLASS = {v: k for k, v in CLASS2IND.items()}

BATCH_SIZE = 4
LR = 1e-4
RANDOM_SEED = 21

NUM_EPOCHS = 50
VAL_EVERY = 5

SAVED_DIR = "checkpoints"

if not os.path.exists(SAVED_DIR):                                                           
    os.makedirs(SAVED_DIR)

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

# 파트를 나누자
#pngs=pngs[:300]
#jsons=jsons[:300]

#pngs=pngs[300:550]
#jsons=jsons[300:550]

pngs=pngs[550:]
jsons=jsons[550:]

class NewCropXRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, tile_size=512, stride=256, min_class_ratio=0.10):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # Split train-validation
        groups = [os.path.dirname(fname) for fname in _filenames]
        ys = [0 for fname in _filenames]
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                if i == 0:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.tile_size = tile_size
        self.stride = stride
        self.min_class_ratio = min_class_ratio
        self.count=0
        self.tiles = []
        self.labels = []
        self.create_all_tiles()


    def create_tiles(self, image, label):
        """
        Create sliding window tiles from the image and label.
        """
        h, w, _ = image.shape
        tiles = []
        labels = []
        
        for i in range(0, h - self.tile_size + 1, self.stride):
            for j in range(0, w - self.tile_size + 1, self.stride):
                img_tile = image[i:i+self.tile_size, j:j+self.tile_size]
                lbl_tile = label[i:i+self.tile_size, j:j+self.tile_size]
                tiles.append(img_tile)
                labels.append(lbl_tile)
        return tiles, labels
    def filter_tiles_by_class_ratio(self, image_tiles, label_tiles):
        """
        Filter tiles based on the presence of class pixels above a threshold.
        """
        filtered_tiles = []
        filtered_labels = []
        for img_tile, lbl_tile in zip(image_tiles, label_tiles):
            class_pixels = lbl_tile[..., 1:].sum()  # Exclude background
            total_pixels = lbl_tile[..., 1:].max(axis=-1).sum()
            class_ratio = class_pixels / total_pixels if total_pixels > 0 else 0
            if class_ratio >= self.min_class_ratio:
                filtered_tiles.append(img_tile)
                filtered_labels.append(lbl_tile)
        return filtered_tiles, filtered_labels
    
    
    def create_all_tiles(self):
        for i in range(len(self.filenames)):
            image, label = self.__process_item__(i)
            tiles, lbls = self.create_tiles(image, label)
            filtered_tiles, filtered_labels = self.filter_tiles_by_class_ratio(tiles, lbls)
            self.tiles.extend(filtered_tiles)
            self.labels.extend(filtered_labels)

    def __process_item__(self, index):
        # 기존 __getitem__에서 이미지 및 라벨 생성 부분 분리
        print("now:",index)
        image_name = self.filenames[index]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = cv2.imread(image_path) / 255.0

        label_name = self.labelnames[index]
        label_path = os.path.join(LABEL_ROOT, label_name)
        label_shape = (image.shape[0], image.shape[1], len(CLASSES))
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        for ann in annotations["annotations"]:
            class_ind = CLASS2IND[ann["label"]]
            points = np.array(ann["points"])
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        return image, label

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        image_tile = torch.from_numpy(self.tiles[index].transpose(2, 0, 1)).float()
        label_tile = torch.from_numpy(self.labels[index].transpose(2, 0, 1)).float()
        return image_tile, label_tile
    
    # 시각화를 위한 팔레트를 설정합니다.
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# 시각화 함수입니다. 클래스가 2개 이상인 픽셀을 고려하지는 않습니다.
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image


train_dataset = NewCropXRayDataset(is_train=True, transforms=None)

valid_dataset = NewCropXRayDataset(is_train=False, transforms=None)


def collate_fn(batch):
    images, labels = zip(*batch)  # Batch 분리
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    collate_fn=collate_fn
)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name='U2P_Crops_best_model.pt'):
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

def train_sliding_window(model, data_loader, val_loader, criterion, optimizer, accumulation_steps=12):
    print(f"Start training...")
    n_class = len(CLASSES)
    best_dice = 0.0

    # 모델을 GPU로 이동
    model = model.cuda()

    for epoch in range(NUM_EPOCHS):
        model.train()
        step_loss = 0.0  # 배치 손실 추적

        optimizer.zero_grad()  # 그래디언트 초기화
        for step, (images, masks) in enumerate(data_loader):
            # GPU 할당
            images, masks = images.cuda(), masks.cuda()

            # Forward pass
            outputs = model(images)

            # Loss 계산
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps  # 손실 나누기

            # Backpropagation
            loss.backward()

            # Gradient Accumulation
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Loss 기록
            step_loss += loss.item() * accumulation_steps  # 누적된 손실 복원
            print(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
                f"Step [{step+1}/{len(data_loader)}], "
                f"Loss: {round(step_loss, 4)}"
            )
            step_loss = 0.0  # Reset step loss

        # Validation 및 모델 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output

class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, n1=64, height=512, width=512, supervision=True):
        super(UNetPlusPlus, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.ModuleList([nn.Upsample(size=(height//(2**c), width//(2**c)), mode='bilinear', align_corners=True) for c in range(4)])
        self.supervision = supervision

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.seg_outputs = nn.ModuleList([nn.Conv2d(filters[0], out_ch, kernel_size=1, padding=0) for _ in range(4)])

    def forward(self, x):
        seg_outputs = []
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up[0](x1_0)], 1))
        seg_outputs.append(self.seg_outputs[0](x0_1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up[1](x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up[0](x1_1)], 1))
        seg_outputs.append(self.seg_outputs[1](x0_2))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up[2](x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up[1](x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up[0](x1_2)], 1))
        seg_outputs.append(self.seg_outputs[2](x0_3))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up[3](x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up[2](x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up[1](x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up[0](x1_3)], 1))
        seg_outputs.append(self.seg_outputs[3](x0_4))

        if self.supervision:
            return seg_outputs
        else:
            return seg_outputs[-1]
        
model = torch.load(os.path.join(SAVED_DIR, "U2P_Crops_best_model.pt"))
#model = UNetPlusPlus(out_ch=len(CLASSES), supervision=False)

# Loss function을 정의합니다.
criterion = nn.BCEWithLogitsLoss()

# Optimizer를 정의합니다.
# optimizer = optim.RMSprop(params=model.parameters(), lr=LR, weight_decay=1e-6)
optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-5)

train_sliding_window(model, train_loader, valid_loader, criterion, optimizer, accumulation_steps=12)