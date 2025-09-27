# Lane Detection Playground (CULane)

This project implements the pipeline described in `Instructions.txt` for CULane, including:

- Data ingestion and preprocessing (30px masks + row-anchors)
- Two-track models: segmentation (SCNN-like) and row-anchor (UFLD-like)
- Hybrid head that fuses both
- Training, inference, and evaluation with official CULane eval
- Minimal baseline hyperparameters and augmentations

## Project layout

```
/project
  data/
    CULane/                 # Put raw CULane here (driver_xx_xxframe/, laneseg_label_w16/, list/)
    processed/
      masks/                # 30px-wide binary masks rendered from annotations
  src/
    dataset.py
    transforms.py
    models/
      scnn_model.py
      uflf_model.py
      hybrid_head.py
      tools/
        make_row_anchors.py   # Generate row-anchor supervision npz from masks
    train.py
    infer.py
    evaluate.py
  experiments/
  README.md
```

## Quick start

1. Download CULane and place it under `data/CULane` with the original structure.

2. Create a Python environment (Python 3.9+ recommended) and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -U pip wheel
pip install -r requirements.txt
```

3. Preprocess masks and row-anchors:

```powershell
python .\src\preprocess.py --culane-root .\data\CULane --out-mask-root .\data\processed\masks --anchors-config .\src\configs\row_anchors_590.yml
python .\src\tools\make_row_anchors.py --list-file .\data\CULane\list\train.txt --img-root .\data\CULane --mask-root .\data\processed\masks --anchors-config .\src\configs\row_anchors_590.yml --out-dir .\data\processed\row_labels
python .\src\tools\make_row_anchors.py --list-file .\data\CULane\list\val.txt --img-root .\data\CULane --mask-root .\data\processed\masks --anchors-config .\src\configs\row_anchors_590.yml --out-dir .\data\processed\row_labels
```

4. Train (hybrid baseline, ResNet-18):

```powershell
python .\src\train.py --config .\src\configs\train_hybrid_r18.yml
```

5. Validate/evaluate using official CULane eval (you must clone the CULane eval repo once):

```powershell
# After training finishes
python .\src\evaluate.py --config .\src\configs\train_hybrid_r18.yml --split val
```

6. Inference on a single image:

```powershell
python .\src\infer.py --weights .\experiments\hybrid_r18\best.ckpt --image <path_to_image>
```

## Notes

- You must manually download the CULane dataset and the official evaluation code (see evaluate.py for instructions). This repo provides integration hooks but cannot fetch them automatically.
- The official CULane evaluator expects a specific folder structure for predictions. Use `infer.py` over the validation list to generate per-image outputs, then run `src\evaluate.py` pointing to the evaluator. Some path tweaks inside the evaluator may be required on Windows.
- The training config uses mixed precision and DDP if available; on Windows single-GPU runs are fine.
- For speed, you can train at 800x288; for final eval keep 1640x590.
