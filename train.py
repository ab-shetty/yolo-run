"""
YOLOv11 Training Script for Flex AI
Handles zipped dataset extraction and training
Compatible with headless environment and NumPy 2.x
"""

import os
import zipfile
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

# -------------------------
# Dataset extraction
# -------------------------
def extract_dataset(zip_path, extract_to="/tmp/dataset"):
    print(f"Extracting dataset from {zip_path}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Dataset extracted to {extract_to}")
    return extract_to

def find_and_update_yaml(dataset_root):
    yaml_candidates = [
        os.path.join(dataset_root, "dataset", "yolov11_data_merged.yaml"),
        os.path.join(dataset_root, "yolov11_data_merged.yaml"),
    ]
    yaml_path = None
    for candidate in yaml_candidates:
        if os.path.exists(candidate):
            yaml_path = candidate
            break
    if not yaml_path:
        raise FileNotFoundError(f"Could not find YAML in {dataset_root}")

    print(f"✅ Found YAML: {yaml_path}")
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    base = os.path.join(dataset_root, "dataset")
    cfg['train'] = os.path.join(base, "train")
    cfg['val'] = os.path.join(base, "val")
    cfg['test'] = os.path.join(base, "test")

    for split, path in [('train', cfg['train']), ('val', cfg['val']), ('test', cfg['test'])]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} path does not exist: {path}")

    updated_yaml = "/tmp/data.yaml"
    with open(updated_yaml, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"✅ Updated YAML saved to: {updated_yaml}")
    return updated_yaml

# -------------------------
# YOLO training
# -------------------------
def train_yolo(args):
    print("="*60)
    print("YOLOv11 Training on Flex AI")
    print("="*60)

    # Dataset
    zip_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {args.dataset_dir}")
    dataset_zip = os.path.join(args.dataset_dir, zip_files[0])
    extract_dir = extract_dataset(dataset_zip)
    yaml_path = find_and_update_yaml(extract_dir)

    # YOLO model
    print(f"Initializing YOLOv11 model: {args.model}")
    model = YOLO(args.model)

    # Training
    print(f"\nStarting training...")
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.out_dir,
        name=args.name,
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        workers=args.workers,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        amp=False,  # avoids NumPy 1.x / 2.x ABI conflicts
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
    )

    print("="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Results saved to: {args.out_dir}/{args.name}")

    if args.validate:
        print("\nRunning validation...")
        metrics = model.val()
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

    return results

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on Flex AI')
    parser.add_argument('--dataset_dir', type=str, default='/input')
    parser.add_argument('--out_dir', type=str, default='/output-checkpoint')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        choices=['yolo11n.pt','yolo11s.pt','yolo11m.pt','yolo11l.pt','yolo11x.pt'])
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--name', type=str, default='grocery_store_yolo11')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save_period', type=int, default=10)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='auto')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=float, default=3.0)
    parser.add_argument('--warmup_momentum', type=float, default=0.8)
    parser.add_argument('--box', type=float, default=7.5)
    parser.add_argument('--cls', type=float, default=0.5)
    parser.add_argument('--dfl', type=float, default=1.5)
    parser.add_argument('--hsv_h', type=float, default=0.015)
    parser.add_argument('--hsv_s', type=float, default=0.7)
    parser.add_argument('--hsv_v', type=float, default=0.4)
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--shear', type=float, default=0.0)
    parser.add_argument('--perspective', type=float, default=0.0)
    parser.add_argument('--flipud', type=float, default=0.0)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--copy_paste', type=float, default=0.0)
    args = parser.parse_args()
    train_yolo(args)

if __name__ == '__main__':
    main()
