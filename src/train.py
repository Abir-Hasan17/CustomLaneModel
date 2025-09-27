import os
import argparse
from typing import Dict, Any
import time
import torch
from torch.utils.data import DataLoader

from utils import load_yaml, ensure_dir, save_checkpoint, logger
from transforms import TrainAugment, ValAugment
from dataset import CULaneDataset
from models.scnn_model import SCNNSegModel
from models.uflf_model import UFLDLikeModel
from models.hybrid_head import HybridLaneModel
from losses import BCEDiceLoss, RowAnchorLoss
from metrics import binary_f1_from_logits


def build_model(cfg: Dict[str, Any], anchors: Dict[str, Any]):
    track = cfg.get('track', 'hybrid')
    backbone = cfg.get('backbone', 'resnet18')
    if track == 'segmentation':
        return SCNNSegModel(backbone=backbone)
    elif track == 'row':
        return UFLDLikeModel(backbone=backbone, num_rows=len(anchors['rows']), num_lanes=anchors['num_lanes'], x_bins=cfg.get('x_bins', 200))
    else:
        return HybridLaneModel(backbone=backbone, num_rows=len(anchors['rows']), num_lanes=anchors['num_lanes'], x_bins=cfg.get('x_bins', 200))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default="src/configs/train_hybrid_r18.yml")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    anchors = load_yaml(cfg['row_anchors'])

    train_tf = TrainAugment(img_size=(cfg['img_width'], cfg['img_height']), roi_top_ratio=cfg['roi_top_ratio'])
    val_tf = ValAugment(img_size=(cfg['img_width'], cfg['img_height']), roi_top_ratio=cfg['roi_top_ratio'])

    train_ds = CULaneDataset(cfg['train_list'], cfg['culane_root'], cfg['mask_root'], transform=train_tf,
                             include_row_anchors=True, anchors_cfg=anchors, row_label_root=cfg.get('row_label_root'))
    val_ds = CULaneDataset(cfg['val_list'], cfg['culane_root'], cfg['mask_root'], transform=val_tf,
                           include_row_anchors=True, anchors_cfg=anchors, row_label_root=cfg.get('row_label_root'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = device == 'cuda'
    
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=pin_memory)

    model = build_model(cfg, anchors)
    model.to(device)

    # Losses
    seg_loss = BCEDiceLoss()
    row_loss = RowAnchorLoss()

    # Optimizer
    if cfg['optimizer'] == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', True) and device == 'cuda')

    ensure_dir(cfg['output_dir'])
    best_val = 0.0
    
    logger.info(f"Starting training with {len(train_ds)} train and {len(val_ds)} val samples")
    logger.info(f"Model: {cfg['track']} with {cfg.get('backbone', 'resnet18')} backbone")
    logger.info(f"Training resolution: {cfg['img_width']}x{cfg['img_height']}")
    logger.info(f"Device: {device}, AMP: {scaler.is_enabled()}")
    logger.info(f"Batch size: {cfg['batch_size']}, LR: {cfg['lr']}, Epochs: {cfg['epochs']}")

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        running_loss = 0.0
        running_seg_loss = 0.0
        running_row_loss = 0.0
        epoch_start = time.time()
        
        for i, batch in enumerate(train_loader, 1):
            iter_start = time.time()
            img = batch['image'].to(device)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                out = model(img)
                loss = 0.0
                seg_loss_val = 0.0
                row_loss_val = 0.0
                
                if 'seg_logits' in out and 'mask' in batch:
                    seg_loss_val = seg_loss(out['seg_logits'], batch['mask'].to(device))
                    loss += cfg['lambda_seg'] * seg_loss_val
                    
                if 'row_exist_logits' in out:
                    if 'exist' in batch and 'x_idx' in batch:
                        targets = { 'exist': batch['exist'].to(device), 'x_idx': batch['x_idx'].to(device) }
                    else:
                        B, L, R = out['row_exist_logits'].shape
                        targets = {'exist': torch.zeros((B, L, R), device=device), 'x_idx': torch.full((B, L, R), -1, device=device)}
                    row_loss_val = row_loss(out, targets)
                    loss += cfg['lambda_row'] * row_loss_val
                    
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            iter_time = time.time() - iter_start
            running_loss += loss.item()
            running_seg_loss += seg_loss_val.item() if isinstance(seg_loss_val, torch.Tensor) else seg_loss_val
            running_row_loss += row_loss_val.item() if isinstance(row_loss_val, torch.Tensor) else row_loss_val
            
            if i % cfg['log_interval'] == 0:
                avg_loss = running_loss / cfg['log_interval']
                avg_seg = running_seg_loss / cfg['log_interval']
                avg_row = running_row_loss / cfg['log_interval']
                lr = opt.param_groups[0]['lr']
                logger.info(f"Epoch {epoch:3d} [{i:4d}/{len(train_loader):4d}] "
                           f"Loss: {avg_loss:.4f} (seg: {avg_seg:.4f}, row: {avg_row:.4f}) "
                           f"LR: {lr:.6f} Time: {iter_time:.3f}s/iter")
                running_loss = running_seg_loss = running_row_loss = 0.0

        # Validation
        model.eval()
        val_start = time.time()
        val_loss = 0.0
        f1_accum = 0.0
        n_f1 = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                out = model(img)
                loss = 0.0
                if 'seg_logits' in out and 'mask' in batch:
                    logits = out['seg_logits']
                    target = batch['mask'].to(device)
                    loss += seg_loss(logits, target)
                    f1_accum += binary_f1_from_logits(logits, target)
                    n_f1 += 1
                val_loss += loss.item()
                
        val_loss /= max(1, len(val_loader))
        f1 = f1_accum / max(1, n_f1)
        val_time = time.time() - val_start
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch:3d} COMPLETE: "
                   f"ValLoss: {val_loss:.4f} SegF1: {f1:.4f} "
                   f"EpochTime: {epoch_time:.1f}s ValTime: {val_time:.1f}s")
        
        # Save best by lowest val loss
        score = -val_loss
        if score > best_val:
            best_val = score
            save_checkpoint({'model': model.state_dict(), 'cfg': cfg}, os.path.join(cfg['output_dir'], 'best.ckpt'))
            logger.info(f"New best model saved! Val loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint({'model': model.state_dict(), 'cfg': cfg, 'epoch': epoch}, 
                          os.path.join(cfg['output_dir'], f'epoch_{epoch}.ckpt'))

    save_checkpoint({'model': model.state_dict(), 'cfg': cfg}, os.path.join(cfg['output_dir'], 'last.ckpt'))
    logger.info(f"Training completed! Best val loss: {-best_val:.4f}")
    logger.info(f"Checkpoints saved in: {cfg['output_dir']}")


if __name__ == '__main__':
    main()
