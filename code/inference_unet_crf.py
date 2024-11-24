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

from model import *

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

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

SAVED_DIR = "/data/ephemeral/home/git/code/checkpoints/result_unet/"



model = torch.load(os.path.join(SAVED_DIR, "unet3p_aug2.pt"))

IMAGE_ROOT = "/data/ephemeral/home/data/test/DCM/"

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

def apply_crf(original_image, output_probs, n_classes):
    """
    Apply Conditional Random Field (CRF) to refine segmentation predictions.

    Args:
        original_image (numpy.ndarray): Original input image (H, W, 3).
        output_probs (numpy.ndarray): Predicted probabilities of shape (n_classes, H, W).
        n_classes (int): Number of classes.

    Returns:
        numpy.ndarray: Refined probabilities of shape (n_classes, H, W).
    """
    H, W = original_image.shape[:2]
    output_probs = output_probs.reshape((n_classes, -1))
    
    # Initialize CRF model
    d = dcrf.DenseCRF2D(W, H, n_classes)
    
    # Compute unary energy
    U = -np.log(output_probs + 1e-8).astype(np.float32)  # Avoid log(0) and ensure float32 precision
    d.setUnaryEnergy(U)
    
    # Add pairwise Gaussian potentials
    sxy_gaussian = 5  # Spatial standard deviation
    gaussian = create_pairwise_gaussian(sdims=(sxy_gaussian, sxy_gaussian), shape=(H, W))
    d.addPairwiseEnergy(gaussian, compat=2)
    
    # Add pairwise bilateral potentials
    sxy_bilateral = 15  # Spatial standard deviation
    srgb = 10  # Color standard deviation
    bilateral = create_pairwise_bilateral(
        sdims=(sxy_bilateral, sxy_bilateral),
        schan=(srgb, srgb, srgb),
        img=original_image,
        chdim=2,
    )
    d.addPairwiseEnergy(bilateral, compat=4)
    
    # Perform CRF inference
    Q = d.inference(1)  # Number of iterations

    # Reshape output and apply optional threshold
    Q = np.array(Q).reshape((n_classes, H, W))
    Q = (Q > 0.5).astype(np.uint8)  # Thresholding

    return Q

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
    
def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)

            # Resize to original dimensions
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = F.softmax(outputs, dim=1).detach().cpu().numpy()  # Compute class probabilities

            for output, image_name in zip(outputs, image_names):
                # Read the original image
                original_image = cv2.imread(os.path.join(IMAGE_ROOT, image_name))
                
                # Apply CRF
                crf_output = apply_crf(original_image, output, n_classes=n_class)

                # Convert CRF output to final binary masks
                crf_labels = np.argmax(crf_output, axis=0).astype(np.uint8)

                for c in range(n_class):
                    if c in IND2CLASS:  # Ensure valid class index
                        segm = (crf_labels == c).astype(np.uint8)  # Create binary mask for class `c`
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    else:
                        print(f"Warning: Invalid class index {c} encountered for {image_name}")

    return rles, filename_and_class

tf = A.Resize(512, 512)

test_dataset = XRayInferenceDataset(transforms=tf)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

rles, filename_and_class = test(model, test_loader)

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

df.to_csv("unet3p_aug2_crf3.csv", index=False)

df["image_name"].nunique()