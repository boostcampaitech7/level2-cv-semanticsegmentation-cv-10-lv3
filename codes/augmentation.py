import albumentations as A

def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=0.5),
        A.ElasticTransform(alpha=15.0, sigma=2.0, p=0.4),
        A.GridDistortion(distort_limit=0.2, p=0.4),
        A.Rotate(limit=30, p=0.3),
        A.CLAHE(clip_limit=(1, 4), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(0.0, 0.3), contrast_limit=0.2, p=0.3),
        A.GridDropout(ratio=0.4, random_offset=False, holes_number_x=12, holes_number_y=12, p=0.4)
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
