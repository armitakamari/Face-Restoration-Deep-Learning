import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

# Paired spatial augmentations
spatial_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
])

def motion_blur(img):
    k = 9
    kernel = np.zeros((k, k))
    angle = random.randint(0, 180)
    kernel[k // 2, :] = np.ones(k)
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

def degrade_downsample(img):
    img = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    scale = random.choice([1 / 1.2, 1 / 1.4])
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.resize(small, (w, h))

def degrade_gaussian_noise(img):
    img = img.astype(np.float32) / 255.0
    noise_std = random.uniform(0.2, 0.5)
    noise = np.random.normal(0, noise_std, img.shape)
    return np.clip(img + noise, 0, 1)

def degrade_motion_blur(img):
    img = img.astype(np.float32) / 255.0
    blurred = motion_blur((img * 255).astype(np.uint8))
    return blurred.astype(np.float32) / 255.0

def degrade(img):
    choice = random.choice(["down", "noise", "blur"])
    if choice == "down":
        return degrade_downsample(img)
    elif choice == "noise":
        return degrade_gaussian_noise(img)
    else:
        return degrade_motion_blur(img)

class FaceDataset(Dataset):
    def __init__(self, folder, is_train=True):
        self.paths = [
            os.path.join(folder, x)
            for x in os.listdir(folder)
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.paths.sort()
        self.is_train = is_train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.paths[idx])
        if img_bgr is None:
            raise RuntimeError(f"cv2.imread failed for: {self.paths[idx]}")

        img = img_bgr[:, :, ::-1]   

       
        if self.is_train:
            clean_aug = spatial_transform(image=img)['image']
        else:
            clean_aug = img

        # Degrade the augmented image
        degraded = degrade(clean_aug)

        # Scale to [0, 1]
        clean = torch.from_numpy(clean_aug.astype(np.float32) / 255.0).permute(2, 0, 1)
        bad = torch.from_numpy(degraded.astype(np.float32)).permute(2, 0, 1)

        return bad.float(), clean.float()
