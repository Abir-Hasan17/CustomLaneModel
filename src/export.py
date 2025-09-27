import argparse
import torch
from utils import load_yaml, ensure_dir
from models.hybrid_head import HybridLaneModel
from models.scnn_model import SCNNSegModel
from models.uflf_model import UFLDLikeModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='./src/configs/train_hybrid_r18.yml')
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='./experiments/exports')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    anchors = load_yaml(cfg['row_anchors'])
    track = cfg.get('track', 'hybrid')
    backbone = cfg.get('backbone', 'resnet18')

    if track == 'segmentation':
        model = SCNNSegModel(backbone)
    elif track == 'row':
        model = UFLDLikeModel(backbone, num_rows=len(anchors['rows']), num_lanes=anchors['num_lanes'], x_bins=cfg.get('x_bins',200))
    else:
        model = HybridLaneModel(backbone, num_rows=len(anchors['rows']), num_lanes=anchors['num_lanes'], x_bins=cfg.get('x_bins',200))
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy = torch.randn(1,3,cfg['img_height'], cfg['img_width'])

    ensure_dir(args.outdir)
    # TorchScript
    ts_path = f"{args.outdir}/model.ts.pt"
    ts = torch.jit.trace(model, dummy)
    ts.save(ts_path)
    print('Saved TorchScript:', ts_path)

    # ONNX
    onnx_path = f"{args.outdir}/model.onnx"
    torch.onnx.export(model, dummy, onnx_path, opset_version=13, input_names=['image'], output_names=['outputs'], do_constant_folding=True)
    print('Saved ONNX:', onnx_path)

if __name__ == '__main__':
    main()
