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
import segmentation_models_pytorch as smp

# visualization
import matplotlib.pyplot as plt
import ttach as tta

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

SAVED_DIR = "checkpoints/result_unet/"

"""model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b7",  # EfficientNet-L2 백본 사용
    encoder_weights="imagenet",     # 사전 학습된 가중치 설정
    in_channels=3,                       # 입력 채널 (RGB 이미지)
    classes=29                           # 출력 클래스 수
)"""
import torch.nn as nn
from transformers import UperNetForSemanticSegmentation

class UperNet_ConvNext_xlarge(nn.Module):
    def __init__(self, num_classes=29):
        super(UperNet_ConvNext_xlarge, self).__init__()
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-xlarge", num_labels=num_classes, ignore_mismatched_sizes=True
        )
        #self.model.gradient_checkpointing_enable()

    def forward(self, image):
        outputs = self.model(pixel_values=image)
        return outputs.logits

model = torch.load(os.path.join(SAVED_DIR, "convnext_kfold1.pt"))

IMAGE_ROOT = "/data/ephemeral/home/data/test/DCM"

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


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

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first

        image = torch.from_numpy(image).float()

        return image, image_name


def test_with_tta(model, data_loader, thr=0.6):
    model = model.cuda()
    model.eval()

    # TTA 변환 정의
    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),  # 0.0006 상승
            # tta.Add(values=[0, 3])
            # tta.Multiply(factors=[0.8, 1.0, 1.2, 1.4]),
            # tta.Scale(scales=[1.0, 1.2]),
            # tta.Add(value=[-10, 10]),
            # tta.VerticalFlip(),
            # tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )

    # TTA 모델 생성
    tta_model = tta.SegmentationTTAWrapper(model, tta_transforms)

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = tta_model(images)  # TTA 모델로 추론

            # 원본 크기 복원
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


tf = A.Resize(1024, 1024)

test_dataset = XRayInferenceDataset(transforms=tf)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

rles, filename_and_class = test_with_tta(model, test_loader)

preds = []
for rle in rles[:len(CLASSES)]:
    pred = decode_rle_to_mask(rle, height=2048, width=2048)
    preds.append(pred)

preds = np.stack(preds, 0)

# fig, ax = plt.subplots(1, 2, figsize=(24, 12))
# ax[0].imshow(image)    # remove channel dimension
# ax[1].imshow(label2rgb(preds))

# plt.show()

classes, filename = zip(*[x.split("_") for x in filename_and_class])

image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv("convnext_kfold1.csv", index=False)

df["image_name"].nunique()
