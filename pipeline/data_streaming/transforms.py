"""Transformer-friendly data augmentations"""
from torchvision import transforms
import random
import numpy as np
from PIL import Image


class RandomGammaCorrection:
    """Random gamma correction for photometric augmentation"""
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range
    
    def __call__(self, img):
        gamma = random.uniform(*self.gamma_range)
        # Convert to numpy for gamma correction
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255.0).astype(np.uint8)
        return Image.fromarray(img_array)


class GaussianNoise:
    """Add slight Gaussian noise"""
    def __init__(self, std=0.02):
        self.std = std
    
    def __call__(self, img):
        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, self.std * 255, img_array.shape).astype(np.float32)
        img_array = img_array + noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


class SmallRandomRotation:
    """Small random rotation (±5-10 degrees)"""
    def __init__(self, degrees=(-10, 10)):
        self.degrees = degrees
    
    def __call__(self, img):
        angle = random.uniform(*self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)


def get_train_transforms():
    """Get training transforms with transformer-friendly augmentations"""
    return transforms.Compose([
        # Geometric augmentations
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.95, 1.0),   # very small crop variation
            ratio=(0.98, 1.02),
            interpolation=Image.BICUBIC
        ),
        
        # Photometric augmentations
        transforms.ColorJitter(
            brightness=0.2,   # ≤ 0.2
            contrast=0.2,      # ≤ 0.2
            saturation=0.2,   # ≤ 0.2
            hue=0.02          # Small hue shift
        ),
        RandomGammaCorrection(gamma_range=(0.8, 1.2)),
        GaussianNoise(std=0.01),
        transforms.ToTensor(),
    ])


def get_val_test_transforms():
    """Get validation/test transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

