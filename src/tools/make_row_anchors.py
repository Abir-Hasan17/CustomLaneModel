import os
import argparse
import yaml
import cv2
import numpy as np
from tqdm import tqdm

# This utility samples X positions along fixed Y rows from binary masks to create row-anchor labels.
# It writes two arrays per image: exist [L,R] and x_idx [L,R] (x bin indices) under out_dir mirroring image paths.

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list-file', type=str, default='data/CULane/list/val.txt')
    ap.add_argument('--img-root', type=str, default='data/CULane/driver_161_90frame')
    ap.add_argument('--mask-root', type=str, default='data/processed/masks')
    ap.add_argument('--anchors-config', type=str, default='src/configs/row_anchors_590.yml')
    ap.add_argument('--out-dir', type=str, default='data/processed/row_labels')
    ap.add_argument('--x-bins', type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    anchors = yaml.safe_load(open(args.anchors_config, 'r'))
    rows = anchors['rows']
    x0, x1 = anchors['x_range']
    with open(args.list_file, 'r') as f:
        rels = [l.strip() for l in f if l.strip()]

    for rel in tqdm(rels, desc='Row-anchor labels'):
        img_path = os.path.join(args.img_root, rel)
        mask_path = os.path.join(args.mask_root, os.path.splitext(rel)[0] + '.png')
        img = cv2.imread(img_path)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        H, W = msk.shape
        exist = np.zeros((anchors['num_lanes'], len(rows)), dtype=np.float32)
        x_idx = -np.ones((anchors['num_lanes'], len(rows)), dtype=np.int32)
        # Simple heuristic: split width into num_lanes vertical bands and find lane position per band
        bands = np.linspace(0, W, anchors['num_lanes'] + 1, dtype=int)
        for li in range(anchors['num_lanes']):
            for r_i, y in enumerate(rows):
                yy = np.clip(int(y), 0, H-1)
                col = msk[yy]
                seg = col[bands[li]:bands[li+1]]
                xs = np.where(seg > 127)[0]
                if xs.size > 5:
                    x = xs.mean() + bands[li]
                    exist[li, r_i] = 1.0
                    x_idx[li, r_i] = int(np.clip(round((x - x0) / max(1, (x1 - x0)) * (args.x_bins - 1)), 0, args.x_bins - 1))
        out_path = os.path.join(args.out_dir, os.path.splitext(rel)[0] + '.npz')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, exist=exist, x_idx=x_idx)

    print('Done.')

if __name__ == '__main__':
    main()
