"""
YOLOv11 Training Script for Flex AI
Handles zipped dataset extraction and training
"""

import subprocess


import sys
# Insert the user site-packages (where pip installs) at the front of sys.path
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import cv2

print("OpenCV loaded from:", cv2.__file__)
print("OpenCV version:", cv2.__version__)





import torch, subprocess, os
torch_path = os.path.dirname(torch.__file__)
for root, _, files in os.walk(torch_path):
    for f in files:
        if f.endswith(".so"):
            so_path = os.path.join(root, f)
            break
print("Checking binary:", so_path)
print(subprocess.run(["strings", so_path, "|", "grep", "numpy"], shell=True, capture_output=True, text=True).stdout[:1000])


import zipfile
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

def extract_dataset(zip_path, extract_to="/tmp/dataset"):
    """
    Extract the zipped dataset to a temporary directory
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting dataset from {zip_path}...")
    
    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✅ Dataset extracted to {extract_to}")
    
    # Find the actual dataset directory (handle nested structures)
    extracted_contents = os.listdir(extract_to)
    print(f"Extracted contents: {extracted_contents}")
    
    return extract_to


def find_and_update_yaml(dataset_root):
    """
    Find the existing YAML file and update paths to absolute paths
    
    Args:
        dataset_root: Root directory of the extracted dataset
    
    Returns:
        Path to the updated YAML file
    """
    # Look for the existing YAML file
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
        raise FileNotFoundError(
            f"Could not find yolov11_data_merged.yaml in {dataset_root}. "
            f"Searched: {yaml_candidates}"
        )
    
    print(f"✅ Found existing YAML at: {yaml_path}")
    
    # Read the existing YAML
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    print(f"Original config:")
    print(f"   Train: {yaml_config['train']}")
    print(f"   Val: {yaml_config['val']}")
    print(f"   Test: {yaml_config['test']}")
    
    # Update paths to absolute paths in the extracted location
    dataset_base = os.path.join(dataset_root, "dataset")
    
    yaml_config['train'] = os.path.join(dataset_base, "train")
    yaml_config['val'] = os.path.join(dataset_base, "val")
    yaml_config['test'] = os.path.join(dataset_base, "test")
    
    # Verify the paths exist
    for split, path in [('train', yaml_config['train']), 
                        ('val', yaml_config['val']), 
                        ('test', yaml_config['test'])]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} path does not exist: {path}")
    
    # Save updated YAML to tmp
    updated_yaml_path = "/tmp/data.yaml"
    with open(updated_yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"\n✅ Updated YAML saved to: {updated_yaml_path}")
    print(f"   Train: {yaml_config['train']}")
    print(f"   Val: {yaml_config['val']}")
    print(f"   Test: {yaml_config['test']}")
    print(f"   Classes: {yaml_config['nc']}")
    
    return updated_yaml_path


def train_yolo(args):
    """
    Main training function
    """
    print("="*60)
    print("YOLOv11 Training on Flex AI")
    print("="*60)
    
    # Step 1: Find and extract the dataset
    dataset_zip = os.path.join(args.dataset_dir, "dataset-20251006T023122Z-1-001.zip")
    
    if not os.path.exists(dataset_zip):
        # Try to find any zip file in the dataset directory
        print(f"Looking for zip files in {args.dataset_dir}...")
        zip_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.zip')]
        if zip_files:
            dataset_zip = os.path.join(args.dataset_dir, zip_files[0])
            print(f"Found: {dataset_zip}")
        else:
            raise FileNotFoundError(f"No zip file found in {args.dataset_dir}")
    
    extract_dir = extract_dataset(dataset_zip, extract_to="/tmp/dataset")
    
    # Step 2: Find and update the existing YAML configuration
    yaml_path = find_and_update_yaml(extract_dir)
    
    # Step 3: Initialize YOLO model
    print(f"\nInitializing YOLOv11 model: {args.model}")
    model = YOLO(args.model)
    
    # Step 4: Train the model
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {args.out_dir}")
    
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
        amp=False,  # Disable AMP to avoid NumPy compatibility check
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
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Results saved to: {args.out_dir}/{args.name}")
    
    # Step 5: Validate the model
    if args.validate:
        print("\nRunning validation...")
        metrics = model.val()
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on Flex AI')
    
    # Dataset arguments
    parser.add_argument('--dataset_dir', type=str, default='/input',
                        help='Directory containing the zipped dataset')
    parser.add_argument('--out_dir', type=str, default='/output-checkpoint',
                        help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 
                                'yolo11l.pt', 'yolo11x.pt'],
                        help='YOLOv11 model size')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (e.g., 0 or 0,1,2,3 for multi-GPU)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--name', type=str, default='grocery_store_yolo11',
                        help='Experiment name')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--save_period', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--validate', type=bool, default=True,
                        help='Run validation after training')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='auto',
                        choices=['SGD', 'Adam', 'AdamW', 'auto'],
                        help='Optimizer')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimizer weight decay')
    parser.add_argument('--warmup_epochs', type=float, default=3.0,
                        help='Warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, default=0.8,
                        help='Warmup initial momentum')
    
    # Loss arguments
    parser.add_argument('--box', type=float, default=7.5,
                        help='Box loss gain')
    parser.add_argument('--cls', type=float, default=0.5,
                        help='Class loss gain')
    parser.add_argument('--dfl', type=float, default=1.5,
                        help='DFL loss gain')
    
    # Augmentation arguments
    parser.add_argument('--hsv_h', type=float, default=0.015,
                        help='HSV-Hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=0.7,
                        help='HSV-Saturation augmentation')
    parser.add_argument('--hsv_v', type=float, default=0.4,
                        help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='Image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='Image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0,
                        help='Image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0,
                        help='Image perspective (+/- fraction)')
    parser.add_argument('--flipud', type=float, default=0.0,
                        help='Image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Image flip left-right (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='Image mixup (probability)')
    parser.add_argument('--copy_paste', type=float, default=0.0,
                        help='Segment copy-paste (probability)')
    
    args = parser.parse_args()
    
    # Run training
    train_yolo(args)


if __name__ == '__main__':
    main()
