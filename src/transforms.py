from typing import Dict, Any, Tuple
import random
import cv2
import numpy as np


class TrainAugment:
    def __init__(self, img_size: Tuple[int, int], roi_top_ratio: float = 0.1):
        self.w, self.h = img_size
        self.roi_top_ratio = roi_top_ratio

    def __call__(self, image: np.ndarray, mask: np.ndarray = None, meta: Dict[str, Any] = None):
        # ROI crop
        h = image.shape[0]
        top = int(h * self.roi_top_ratio)
        image = image[top:]
        if mask is not None:
            mask = mask[top:]
        # Resize
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        # Photometric
        if random.random() < 0.8:
            alpha = 1.0 + random.uniform(-0.3, 0.3)  # contrast
            beta = random.uniform(-40, 40)  # brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        # Geometric small rotation
        if random.random() < 0.5:
            ang = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((self.w/2, self.h/2), ang, 1.0)
            image = cv2.warpAffine(image, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (self.w, self.h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        # Random erase
        if random.random() < 0.5:
            for _ in range(random.randint(1, 2)):
                rh = random.randint(int(0.02*self.h), int(0.2*self.h))
                rw = random.randint(int(0.02*self.w), int(0.2*self.w))
                ry = random.randint(0, max(0, self.h - rh - 1))
                rx = random.randint(0, max(0, self.w - rw - 1))
                image[ry:ry+rh, rx:rx+rw] = random.randint(0, 255)
        return image, mask, meta or {}


class ValAugment:
    def __init__(self, img_size: Tuple[int, int], roi_top_ratio: float = 0.1):
        self.w, self.h = img_size
        self.roi_top_ratio = roi_top_ratio

    def __call__(self, image: np.ndarray, mask: np.ndarray = None, meta: Dict[str, Any] = None):
        h = image.shape[0]
        top = int(h * self.roi_top_ratio)
        image = image[top:]
        if mask is not None:
            mask = mask[top:]
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return image, mask, meta or {}
