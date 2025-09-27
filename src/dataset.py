import os
from typing import Tuple, List, Dict, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import normalize_img, to_tensor


class CULaneDataset(Dataset):
    def __init__(self,
                 list_file: str,
                 culane_root: str,
                 mask_root: str = None,
                 transform=None,
                 include_row_anchors: bool = True,
                 anchors_cfg: Dict[str, Any] = None,
                 row_label_root: str = None):
        super().__init__()
        self.items = self._read_list(list_file)
        self.root = culane_root + "/driver_161_90frame"
        self.mask_root = mask_root
        self.transform = transform
        self.include_row_anchors = include_row_anchors
        self.anchors_cfg = anchors_cfg
        self.row_label_root = row_label_root

    def _read_list(self, path: str) -> List[str]:
        with open(path, 'r') as f:
            items = [line.strip() for line in f.readlines() if line.strip()]
        return items

    def __len__(self):
        return len(self.items)

    def _load_img(self, rel_path: str):
        path = os.path.join(self.root, rel_path)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, rel_path: str):
        if self.mask_root is None:
            return None
        mask_path = os.path.join(self.mask_root, os.path.splitext(rel_path)[0] + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask

    def _load_row_labels(self, rel_path: str):
        if not self.include_row_anchors or self.row_label_root is None:
            return None
        path = os.path.join(self.row_label_root, os.path.splitext(rel_path)[0] + '.npz')
        if not os.path.isfile(path):
            return None
        data = np.load(path)
        return {"exist": torch.from_numpy(data['exist']), "x_idx": torch.from_numpy(data['x_idx'])}

    def __getitem__(self, idx):
        rel = self.items[idx]
        img = self._load_img(rel)
        mask = self._load_mask(rel)
        meta = {"rel_path": rel}
        if self.transform is not None:
            img, mask, meta = self.transform(img, mask, meta)
        img = img.astype(np.float32) / 255.0
        img = normalize_img(img)
        img = to_tensor(img)
        sample = {"image": img, "meta": meta}
        if mask is not None:
            mask = (mask > 127).astype(np.float32)
            sample["mask"] = torch.from_numpy(mask).unsqueeze(0)
        row = self._load_row_labels(rel)
        if row is not None:
            sample.update(row)
        return sample
