"""
YOLOv11 Hyperparameter Sweep Script for Flex AI
Runs multiple training configurations to find optimal hyperparameters
"""

import os
import zipfile
import yaml
import json
from pathlib import Path
from ultralytics import YOLO
import itertools

# -------------------------
# Dataset extraction (same as train.py)
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

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    base = os.path.join(dataset_root, "dataset")
    cfg['train'] = os.path.join(base, "train")
    cfg['val'] = os.path.join(base, "val")
    cfg['test'] = os.path.join(base, "test")

    updated_yaml = "/tmp/data.yaml"
    with open(updated_yaml, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return updated_yaml

# -------------------------
# Hyperparameter sweep configurations
# -------------------------
def get_sweep_configs():
    """
    Define hyperparameter sweep space.
    Returns a list of configuration dictionaries.
    """
    
    # Define parameter grids
    sweep_params = {
        # Learning rate sweep
        'lr': [0.001, 0.01, 0.02],
        
        # Optimizer sweep
        'optimizer': ['SGD', 'Adam', 'AdamW'],
        
        # Batch size sweep
        'batch': [8, 16, 32],
        
        # Image size sweep
        'imgsz': [640, 800],
        
        # Augmentation intensity
        'augmentation': ['light', 'medium', 'heavy'],
    }
    
    # Generate all combinations (grid search)
    # WARNING: This can create many runs! Use carefully.
    configs = []
    
    # Example 1: Grid search over learning rate and optimizer
    for lr in sweep_params['lr']:
        for opt in sweep_params['optimizer']:
            configs.append({
                'name': f'lr{lr}_opt{opt}',
                'lr0': lr,
                'optimizer': opt,
                'batch': 16,  # fixed
                'imgsz': 640,  # fixed
            })
    
    # Example 2: Test different batch sizes with best LR
    # for batch in sweep_params['batch']:
    #     configs.append({
    #         'name': f'batch{batch}',
    #         'lr0': 0.01,
    #         'optimizer': 'auto',
    #         'batch': batch,
    #         'imgsz': 640,
    #     })
    
    return configs

def get_augmentation_params(level='medium'):
    """
    Return augmentation parameters based on intensity level.
    """
    aug_configs = {
        'light': {
            'hsv_h': 0.005,
            'hsv_s': 0.3,
            'hsv_v': 0.2,
            'degrees': 0.0,
            'translate': 0.05,
            'scale': 0.3,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'mixup': 0.0,
        },
        'medium': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
        },
        'heavy': {
            'hsv_h': 0.03,
            'hsv_s': 0.9,
            'hsv_v': 0.6,
            'degrees': 10.0,
            'translate': 0.2,
            'scale': 0.8,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.15,
        }
    }
    return aug_configs.get(level, aug_configs['medium'])

# -------------------------
# Training function
# -------------------------
def run_single_experiment(config, yaml_path, out_dir, model_type='yolo11n.pt', epochs=50):
    """
    Run a single training experiment with given config.
    """
    exp_name = config['name']
    print("\n" + "="*80)
    print(f"Starting experiment: {exp_name}")
    print("="*80)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Get augmentation params if specified
    aug_params = {}
    if 'augmentation' in config:
        aug_params = get_augmentation_params(config['augmentation'])
        print(f"Augmentation level: {config['augmentation']}")
    
    # Initialize model
    model = YOLO(model_type)
    
    # Base training parameters
    train_params = {
        'data': yaml_path,
        'epochs': epochs,
        'batch': config.get('batch', 16),
        'imgsz': config.get('imgsz', 640),
        'device': '0',
        'project': out_dir,
        'name': exp_name,
        'patience': 20,  # Early stopping for faster sweeps
        'save': True,
        'save_period': -1,  # Only save best and last
        'workers': 8,
        'pretrained': True,
        'optimizer': config.get('optimizer', 'auto'),
        'amp': False,
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3.0),
        'warmup_momentum': config.get('warmup_momentum', 0.8),
        'box': config.get('box', 7.5),
        'cls': config.get('cls', 0.5),
        'dfl': config.get('dfl', 1.5),
        'verbose': True,
    }
    
    # Add augmentation params
    train_params.update(aug_params)
    
    # Override with any additional config params
    train_params.update({k: v for k, v in config.items() 
                        if k not in ['name', 'augmentation']})
    
    # Train
    results = model.train(**train_params)
    
    # Validate
    metrics = model.val()
    
    result_summary = {
        'name': exp_name,
        'config': config,
        'metrics': {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }
    }
    
    print(f"\n📊 Results for {exp_name}:")
    print(f"   mAP50: {result_summary['metrics']['mAP50']:.4f}")
    print(f"   mAP50-95: {result_summary['metrics']['mAP50-95']:.4f}")
    print(f"   Precision: {result_summary['metrics']['precision']:.4f}")
    print(f"   Recall: {result_summary['metrics']['recall']:.4f}")
    
    return result_summary

# -------------------------
# Main sweep function
# -------------------------
def run_sweep(dataset_dir='/input', out_dir='/output-checkpoints', 
              model='yolo11n.pt', epochs=50):
    """
    Run hyperparameter sweep.
    """
    print("="*80)
    print("YOLOv11 Hyperparameter Sweep")
    print("="*80)
    
    # Setup dataset
    zip_files = [f for f in os.listdir(dataset_dir) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {dataset_dir}")
    dataset_zip = os.path.join(dataset_dir, zip_files[0])
    extract_dir = extract_dataset(dataset_zip)
    yaml_path = find_and_update_yaml(extract_dir)
    
    # Get sweep configurations
    configs = get_sweep_configs()
    print(f"\nRunning {len(configs)} experiments")
    print(f"Model: {model}")
    print(f"Epochs per experiment: {epochs}")
    
    # Run all experiments
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(configs)}")
        print(f"{'='*80}")
        
        try:
            result = run_single_experiment(
                config=config,
                yaml_path=yaml_path,
                out_dir=out_dir,
                model_type=model,
                epochs=epochs
            )
            results.append(result)
        except Exception as e:
            print(f"Experiment {config['name']} failed: {e}")
            results.append({
                'name': config['name'],
                'config': config,
                'error': str(e)
            })
    
    # Save results summary
    results_file = os.path.join(out_dir, 'sweep_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Sweep Complete!")
    print("="*80)
    
    # Print summary
    print("\nResults Summary (sorted by mAP50-95):")
    print("-" * 80)
    
    valid_results = [r for r in results if 'metrics' in r]
    valid_results.sort(key=lambda x: x['metrics']['mAP50-95'], reverse=True)
    
    print(f"{'Rank':<6} {'Name':<25} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    
    for rank, result in enumerate(valid_results, 1):
        m = result['metrics']
        print(f"{rank:<6} {result['name']:<25} {m['mAP50']:<10.4f} "
              f"{m['mAP50-95']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f}")
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Print best config
    if valid_results:
        best = valid_results[0]
        print(f"\nBest Configuration: {best['name']}")
        print(f"   Config: {json.dumps(best['config'], indent=6)}")
        print(f"   mAP50-95: {best['metrics']['mAP50-95']:.4f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Hyperparameter Sweep')
    parser.add_argument('--dataset_dir', type=str, default='/input',
                        help='Directory containing dataset zip')
    parser.add_argument('--out_dir', type=str, default='/output-checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        choices=['yolo11n.pt','yolo11s.pt','yolo11m.pt','yolo11l.pt','yolo11x.pt'],
                        help='YOLO model size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs per experiment (use fewer for faster sweeps)')
    
    args = parser.parse_args()
    
    run_sweep(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        model=args.model,
        epochs=args.epochs
    )
