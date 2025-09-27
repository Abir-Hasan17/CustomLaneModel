import os
import argparse
import torch
import cv2
import numpy as np
from utils import load_yaml
from postprocess import row_to_points
from models.hybrid_head import HybridLaneModel
from models.scnn_model import SCNNSegModel
from models.uflf_model import UFLDLikeModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--config', type=str, default='./src/configs/train_hybrid_r18.yml')
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
        model = UFLDLikeModel(backbone, num_rows=len(anchors['rows']), num_lanes=anchors['num_lanes'])
    else:
        model = HybridLaneModel(backbone, num_rows=len(anchors['rows']), num_lanes=anchors['num_lanes'])

    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    img = cv2.imread(args.image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (cfg['img_width'], cfg['img_height']))
    inp = (inp.astype(np.float32)/255.0 - np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
    inp = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0)

    with torch.no_grad():
        out = model(inp)
    vis = img.copy()
    if 'seg_logits' in out:
        prob = torch.sigmoid(out['seg_logits'])[0,0].cpu().numpy()
        prob = cv2.resize(prob, (img.shape[1], img.shape[0]))
        overlay = (prob*255).astype(np.uint8)
        color = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(vis, 0.6, color, 0.4, 0)
    if 'row_x_logits' in out and 'row_exist_logits' in out:
        lanes_batch = row_to_points(out, anchors)
        # row_to_points already uses anchors' absolute Y; we only need to clip to image
        for lanes in lanes_batch:
            for pts in lanes:
                pts = np.clip(pts, [0,0], [img.shape[1]-1, img.shape[0]-1])
                for i in range(1, len(pts)):
                    cv2.line(vis, tuple(pts[i-1]), tuple(pts[i]), (0,255,0), 2)

    out_path = os.path.splitext(args.image)[0] + '_lane_vis.jpg'
    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
