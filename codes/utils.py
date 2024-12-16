import os
import torch
import random
import numpy as np

import config as cfg
pt_name = cfg.PT_NAME
log_name = cfg.LOG_NAME
dir_path = cfg.SAVED_DIR

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name=pt_name):
    output_path = os.path.join(dir_path, file_name)
    torch.save(model, output_path)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def log_to_file(message, file_path=f"./log/{log_name}.txt"):
    with open(file_path, "a") as f:
        f.write(message + "\n")

### for sliding
def collate_fn(batch):
    images, labels = zip(*batch)  # Batch 분리
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels