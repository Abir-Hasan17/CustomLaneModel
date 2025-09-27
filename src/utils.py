import os
from typing import Tuple, List
import yaml

# Fallback logger if loguru is unavailable
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    class _Logger:
        def info(self, *args, **kwargs):
            print(*args)
        def warning(self, *args, **kwargs):
            print(*args)
        def error(self, *args, **kwargs):
            print(*args)
    logger = _Logger()


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def crop_roi(img_h: int, top_ratio: float) -> Tuple[int, int]:
    top = int(img_h * top_ratio)
    return top, img_h


def normalize_img(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # x: float32 HxWxC 0..1
    import numpy as np
    x = (x - np.array(mean)) / np.array(std)
    return x


def to_tensor(img):
    import torch
    import numpy as np
    if isinstance(img, torch.Tensor):
        return img
    img = img.transpose(2, 0, 1)  # HWC->CHW
    return torch.from_numpy(img).float()


def save_checkpoint(state, path):
    ensure_dir(os.path.dirname(path))
    import torch
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path}")
