import os
import json
import argparse
from typing import List, Tuple
import cv2
import numpy as np
from utils import ensure_dir, load_yaml, logger


def draw_polyline_mask(shape: Tuple[int, int], points: List[Tuple[int, int]], thickness: int = 30):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    return mask


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--culane-root', type=str, default="data/CULane")
    ap.add_argument('--out-mask-root', type=str, default="data/processed/masks")
    ap.add_argument('--anchors-config', type=str, default="src/configs/row_anchors_590.yml")
    ap.add_argument('--thickness', type=int, default=30)
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_mask_root)
    anchors = load_yaml(args.anchors_config)
    # NOTE: We assume existence of list/train.txt etc listing relative image paths.
    list_dir = os.path.join(args.culane_root, 'list')
    for split in ['train', 'val']:
        list_file = os.path.join(list_dir, f'{split}.txt')
        if not os.path.isfile(list_file):
            logger.warning(f"List file missing: {list_file}; skipping {split}")
            continue
        with open(list_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        for rel_img in lines:
            # Derive annotation path from official structure; here we look up driver_161_90frame_labels png
            ann_path = os.path.join(args.culane_root, 'driver_161_90frame_labels', os.path.splitext(rel_img)[0] + '.png')
            img_path = os.path.join(args.culane_root, 'driver_161_90frame', rel_img)
            if not os.path.isfile(img_path):
                logger.warning(f"Image missing: {img_path}")
                continue
            seg = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            if seg is None:
                # If no per-pixel labels exist, skip (user may need to use official scripts)
                logger.warning(f"Annotation missing: {ann_path}; skipping mask render")
                continue
            # Build 30px mask: treat any label>0 as lane
            mask = (seg > 0).astype(np.uint8) * 255
            # Optionally thicken to 30px (already thick in CULane labels), apply morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            out_path = os.path.join(args.out_mask_root, os.path.splitext(rel_img)[0] + '.png')
            ensure_dir(os.path.dirname(out_path))
            cv2.imwrite(out_path, mask)
    logger.info("Preprocessing complete. Row-anchors are defined by config and consumed at training time.")


if __name__ == '__main__':
    main()
