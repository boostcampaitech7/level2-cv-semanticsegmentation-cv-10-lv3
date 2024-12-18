# python native
import os

# external library
import numpy as np
import pandas as pd
import albumentations as A

# torch
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset.XRayDataset import *
from configs import config as cfg
from models import *
from tta import *


def main():
    # model 불러오기
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

    model = torch.load(os.path.join(cfg.SAVED_DIR, f"{cfg.PT_NAME}"))


    # 데이터 불러오기
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


    classes, filename = zip(*[x.split("_") for x in filename_and_class])

    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv("unetpp_base.csv", index=False)

    df["image_name"].nunique()


if __name__ == "__main__":
    main()
