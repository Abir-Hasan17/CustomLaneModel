import argparse
import tarfile
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=str, required=True, help='Path to .tar.gz file')
    ap.add_argument('--out-dir', type=str, required=True, help='Directory to extract into')
    return ap.parse_args()


def main():
    args = parse_args()
    arc = Path(args.archive)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arc, 'r:gz') as tf:
        tf.extractall(out)
    print(f'Extracted {arc} -> {out}')


if __name__ == '__main__':
    main()
