import os
import json

# external library
import cv2
import numpy as np
from sklearn.model_selection import GroupKFold

# torch
import torch
from torch.utils.data import Dataset

from configs.config import IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND

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


class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, fold=0, all=False, sliding=False, tile_size=512, stride=256, min_class_ratio=0.10):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        if all:
            ############# Train 800 ###############
            if is_train:
                # 모든 train 데이터
                filenames = list(_filenames)
                labelnames = list(_labelnames)
            else:
                # validation 데이터
                filenames = []
                labelnames = []
        else:
            groups = [os.path.dirname(fname) for fname in _filenames]
            ys = [0 for fname in _filenames]
            gkf = GroupKFold(n_splits=5)

            filenames = []
            labelnames = []
            for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
                if is_train:
                    if i == fold:
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

        self.siding = sliding

        if sliding:
            self.tile_size = tile_size
            self.stride = stride
            self.min_class_ratio = min_class_ratio
            self.count=0
            self.tiles = []
            self.labels = []
            self.create_all_tiles()

    def __len__(self):
        if self.sliding:
            return len(self.tiles)
        else:
            return len(self.filenames)

    def __getitem__(self, key):
        if self.sliding:
            image_tile = torch.from_numpy(self.tiles[key].transpose(2, 0, 1)).float()
            label_tile = torch.from_numpy(self.labels[key].transpose(2, 0, 1)).float()
            return image_tile, label_tile
        else:
            image_name = self.filenames[key]
            image_path = os.path.join(IMAGE_ROOT, image_name)

            image = cv2.imread(image_path)
            image = image / 255.

            label_name = self.labelnames[key]
            label_path = os.path.join(LABEL_ROOT, label_name)

            label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
            label = np.zeros(label_shape, dtype=np.uint8)

            with open(label_path, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]

            for ann in annotations:
                c = ann["label"]
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"])
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_ind] = class_label

            if self.transforms is not None:
                inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
                result = self.transforms(**inputs)

                image = result["image"]
                label = result["mask"] if self.is_train else label

            # to tenser will be done later
            image = image.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)

            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()

            return image, label
    
    ### for sliding
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

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        return image, image_name