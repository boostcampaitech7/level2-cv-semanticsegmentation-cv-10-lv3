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
import segmentation_models_pytorch as smp


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

SAVED_DIR = "checkpoints"

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=len(CLASSES)
)


model = torch.load(os.path.join(SAVED_DIR, "./result_unet/unetPP_smp_800_1024.pt"))

IMAGE_ROOT = "./data/test/"

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}


def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tensor
        image = image.transpose(2, 0, 1)    # make channel first
        image = torch.from_numpy(image).float()

        return image, image_name


def test_with_tta(model, data_loader, tta_transforms, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()

            # TTA 예측 생성
            outputs_list = []
            for tta_transform in tta_transforms:
                tta_images = []
                for img in images:
                    img_np = img.cpu().numpy().transpose(1, 2, 0)  # HWC로 변환
                    augmented = tta_transform(image=img_np)["image"]
                    augmented = torch.from_numpy(augmented.transpose(2, 0, 1)).float()  # CHW로 변환
                    tta_images.append(augmented)
                tta_images = torch.stack(tta_images).cuda()
                outputs = model(tta_images)
                outputs_list.append(outputs)

            # TTA 결과 평균
            outputs = torch.mean(torch.stack(outputs_list), dim=0)

            # 원본 크기로 복원
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


# TTA Transformations 정의
tta_transforms = [
    A.Compose([A.Resize(1024, 1024)]),  # 기본 Resize
]

test_dataset = XRayInferenceDataset(transforms=A.Resize(1024, 1024))

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

# TTA Inference 실행
rles, filename_and_class = test_with_tta(model, test_loader, tta_transforms)

# 결과 저장
classes, filename = zip(*[x.split("_") for x in filename_and_class])

image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv("unetpp_tta.csv", index=False)
