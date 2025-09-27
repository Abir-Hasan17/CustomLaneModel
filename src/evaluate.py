import os
import argparse
import subprocess
from utils import load_yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--split', type=str, default='val', choices=['val','test'])
    ap.add_argument('--pred-dir', type=str, default='./experiments/preds')
    ap.add_argument('--culane-eval-path', type=str, default='./external/CULane-eval')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    # NOTE: This script assumes you have cloned the official CULane eval repo under external/CULane-eval
    if not os.path.isdir(args.culane_eval_path):
        print('Missing CULane eval repo. Please clone https://github.com/XingangPan/SCNN/tree/master/tools/lane_evaluation under external/CULane-eval')
        return
    # Prepare predictions directory structure according to evaluator expectations (not implemented here)
    print('Note: Generating prediction files is task-specific. Use infer.py over val split to fill pred-dir.')
    # Run official eval (example call; adjust paths inside the eval code as necessary)
    try:
        subprocess.run(['python', os.path.join(args.culane_eval_path, 'evaluate.py')], check=True)
    except Exception as e:
        print('Evaluation failed; ensure evaluator paths are set. Error:', e)


if __name__ == '__main__':
    main()
