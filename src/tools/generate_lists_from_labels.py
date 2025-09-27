import argparse
import os
from pathlib import Path
from glob import glob
import random

# Fallback: if official list.tar.gz is not available, make simple train/val lists
# by scanning images that have labels under laneseg_label_w16. Not an official split.

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--culane-root', type=str, default="data/CULane")
    ap.add_argument('--train-ratio', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.culane_root)
    # Prefer directories whose names end with .MP4 (user's structure)
    img_dirs = [p for p in root.rglob('*') if p.is_dir() and p.name.lower().endswith('.mp4')]
    if not img_dirs:
        # Fallback to classic CULane pattern
        img_dirs = [p for p in root.glob('driver_*_*frame') if p.is_dir()]
    print(f"Found {len(img_dirs)} image directories under {root}")
    label_root = "data/CULane/driver_161_90frame_labels"
    print(f"Looking for labels under {label_root}")

    samples = []
    for img_dir in img_dirs:
        for img_path in img_dir.rglob('*.jpg'):
            # Relative path under dataset root
            rel = img_path.relative_to(root)
            rell = img_path.relative_to(root / 'driver_161_90frame')
            # Exclude .jpg extension first, then map to .png under laneseg_label_w16
            rel_no_ext = rell.with_suffix('')
            label_path = (label_root / rel_no_ext).with_suffix('.png')
            
            if label_path.exists():
                samples.append(rell.as_posix())

    random.shuffle(samples)
    n_train = int(len(samples) * args.train_ratio)
    train = samples[:n_train]
    val = samples[n_train:]

    list_dir = root / 'list'
    list_dir.mkdir(parents=True, exist_ok=True)
    (list_dir / 'train.txt').write_text('\n'.join(train) + '\n')
    (list_dir / 'val.txt').write_text('\n'.join(val) + '\n')
    print(f"Wrote {len(train)} train and {len(val)} val samples to {list_dir}")


if __name__ == '__main__':
    main()
