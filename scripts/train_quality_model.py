#!/usr/bin/env python3
"""
Professional training script for underwater image quality assessment models.

This script provides a complete pipeline for:
- Generating synthetic training data
- Training deep learning models (UnderwaterQualityNet, EfficientNet, ViT)
- Evaluation and checkpointing
- Model export for inference

Features:
- GPU acceleration with automatic device detection
- Multiple model architectures
- Data augmentation
- Early stopping and learning rate scheduling
- Comprehensive logging and visualization
- Model export in different formats

Usage:
    # Quick start with synthetic data
    python train_quality_model.py --model underwater --epochs 20

    # Train with your own data
    python train_quality_model.py --model efficientnet --data-dir /path/to/images --labels labels.json --epochs 50

    # Resume training from checkpoint
    python train_quality_model.py --model underwater --resume checkpoints/best_model.pth --epochs 30
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_models import (
    UnderwaterQualityNet,
    VisionTransformerQualityNet,
    EfficientNetQualityNet,
    MultiMetricPredictor
)
from models.trainer import (
    UnderwaterDataset,
    QualityAssessmentTrainer,
    create_synthetic_dataset
)


def load_labeled_data(data_dir: str, labels_file: str) -> Tuple[List[str], List[float], List[float]]:
    """
    Load labeled data from directory and JSON labels file.

    Args:
        data_dir: Directory containing images
        labels_file: JSON file with labels in format:
                    {"image.jpg": {"quality_score": 75.3, "blue_water_severity": 4.2}, ...}

    Returns:
        Tuple of (image_paths, quality_scores, blue_water_scores)
    """
    data_dir = Path(data_dir)

    with open(labels_file, 'r') as f:
        labels = json.load(f)

    image_paths = []
    quality_scores = []
    blue_water_scores = []

    for filename, scores in labels.items():
        img_path = data_dir / filename
        if img_path.exists():
            image_paths.append(str(img_path))
            quality_scores.append(scores.get('quality_score', 50.0))
            blue_water_scores.append(scores.get('blue_water_severity', 5.0))
        else:
            warnings.warn(f"Image not found: {img_path}")

    print(f"Loaded {len(image_paths)} labeled images from {data_dir}")
    return image_paths, quality_scores, blue_water_scores


def create_dataloaders(
    image_paths: List[str],
    quality_scores: List[float],
    blue_water_scores: List[float],
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 4,
    scale: float = 1.0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        image_paths: List of image paths
        quality_scores: Quality scores (0-100)
        blue_water_scores: Blue water severity (0-10)
        batch_size: Batch size for training
        val_split: Validation split ratio
        num_workers: Number of data loading workers
        scale: Scale factor for resizing images (default: 1.0)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = UnderwaterDataset(
        image_paths=image_paths,
        quality_scores=quality_scores,
        blue_water_scores=blue_water_scores,
        augment=False,  # Will set augment=True for train split
        scale=scale
    )

    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create separate datasets with different augmentation
    train_dataset = UnderwaterDataset(
        image_paths=[image_paths[i] for i in train_indices],
        quality_scores=[quality_scores[i] for i in train_indices],
        blue_water_scores=[blue_water_scores[i] for i in train_indices],
        augment=True,  # Augmentation for training
        scale=scale
    )

    val_dataset = UnderwaterDataset(
        image_paths=[image_paths[i] for i in val_indices],
        quality_scores=[quality_scores[i] for i in val_indices],
        blue_water_scores=[blue_water_scores[i] for i in val_indices],
        augment=False,  # No augmentation for validation
        scale=scale
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    return train_loader, val_loader


def create_model(model_name: str, device: str) -> nn.Module:
    """
    Create and initialize model.

    Args:
        model_name: Model architecture ('underwater', 'efficientnet', 'vit', 'multimetric')
        device: Device to place model on

    Returns:
        Initialized model
    """
    print(f"\nInitializing {model_name} model...")

    if model_name == 'underwater':
        model = UnderwaterQualityNet(pretrained=True, dropout=0.3)
    elif model_name == 'efficientnet':
        model = EfficientNetQualityNet(model_name='efficientnet_b3', pretrained=True)
    elif model_name == 'vit':
        model = VisionTransformerQualityNet(model_name='vit_base_patch16_224', pretrained=True)
    elif model_name == 'multimetric':
        model = MultiMetricPredictor(backbone='resnet50', pretrained=True, num_metrics=37)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def plot_training_history(history: Dict, output_path: str):
    """
    Plot training history curves.

    Args:
        history: Training history dictionary
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    axes[0, 1].plot(history['val_mae'], label='Val MAE', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 0].plot(history['learning_rates'], label='Learning Rate', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Best epoch marker
    best_epoch = np.argmin(history['val_loss'])
    axes[1, 1].text(0.5, 0.6, f"Best Epoch: {best_epoch + 1}",
                    ha='center', va='center', fontsize=16, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.4, f"Best Val Loss: {history['val_loss'][best_epoch]:.4f}",
                    ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.2, f"Best Val MAE: {history['val_mae'][best_epoch]:.4f}",
                    ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {output_path}")


def export_model_for_inference(model: nn.Module, output_path: str, example_input_size: Tuple = (1, 3, 384, 384)):
    """
    Export trained model for inference.

    Args:
        model: Trained model
        output_path: Path to save exported model
        example_input_size: Example input size for tracing
    """
    model.eval()

    # Save state dict (most common format)
    torch.save(model.state_dict(), output_path)
    print(f"Model state dict saved to: {output_path}")

    # Optionally save as TorchScript for deployment
    try:
        scripted_path = output_path.replace('.pth', '_scripted.pt')
        example_input = torch.randn(*example_input_size).to(next(model.parameters()).device)

        # Check if model needs dual input
        if hasattr(model, 'forward') and 'x_lab' in model.forward.__code__.co_varnames:
            # Dual input model
            example_lab = torch.randn(*example_input_size).to(next(model.parameters()).device)
            scripted_model = torch.jit.trace(model, (example_input, example_lab))
        else:
            # Single input model
            scripted_model = torch.jit.trace(model, example_input)

        scripted_model.save(scripted_path)
        print(f"TorchScript model saved to: {scripted_path}")
    except Exception as e:
        warnings.warn(f"Could not export TorchScript model: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train underwater image quality assessment models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with synthetic data (quick start)
  %(prog)s --model underwater --synthetic 1000 --epochs 20

  # Train with your labeled data
  %(prog)s --model efficientnet --data-dir images/ --labels labels.json --epochs 50

  # Train on HPC with multiple GPUs (if available)
  %(prog)s --model vit --synthetic 5000 --epochs 100 --batch-size 64 --workers 16

  # Resume training from checkpoint
  %(prog)s --model underwater --resume checkpoints/checkpoint_epoch_10.pth --epochs 30

Model Options:
  - underwater: Dual-stream ResNet50 (best for comprehensive assessment)
  - efficientnet: Lightweight EfficientNet-B3 (fastest inference)
  - vit: Vision Transformer (highest accuracy, needs more data)
  - multimetric: Predicts all 37 metrics (experimental)
        """
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--labels', type=str, default=None,
                       help='JSON file with quality labels')
    parser.add_argument('--synthetic', type=int, default=None,
                       help='Generate N synthetic training images')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for resizing images (e.g., 0.5 for 50%% of original size). Default: 1.0 (no scaling)')

    # Model arguments
    parser.add_argument('--model', type=str, default='underwater',
                       choices=['underwater', 'efficientnet', 'vit', 'multimetric'],
                       help='Model architecture to train')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience')

    # System arguments
    parser.add_argument('--workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--output-dir', type=str, default='training_output',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*70)
    print("UNDERWATER IMAGE QUALITY ASSESSMENT - MODEL TRAINING")
    print("="*70)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Prepare data
    print("\n" + "="*70)
    print("PREPARING DATASET")
    print("="*70)

    if args.synthetic:
        # Generate synthetic data
        print(f"Generating {args.synthetic} synthetic images...")
        synthetic_dir = output_dir / 'synthetic_data'
        image_paths, quality_scores, blue_water_scores = create_synthetic_dataset(
            num_images=args.synthetic,
            output_dir=str(synthetic_dir),
            random_seed=args.seed
        )
    elif args.data_dir and args.labels:
        # Load real labeled data
        image_paths, quality_scores, blue_water_scores = load_labeled_data(
            args.data_dir, args.labels
        )
    else:
        print("ERROR: Must specify either --synthetic or both --data-dir and --labels")
        return 1

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        image_paths,
        quality_scores,
        blue_water_scores,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.workers,
        scale=args.scale
    )

    # Create model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    model = create_model(args.model, device)

    # Create trainer
    print("\n" + "="*70)
    print("SETTING UP TRAINER")
    print("="*70)
    trainer = QualityAssessmentTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=10
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    try:
        history = trainer.train(
            num_epochs=args.epochs,
            early_stopping_patience=args.early_stopping,
            save_best=True
        )

        # Plot training history
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        plot_training_history(history, str(output_dir / 'training_history.png'))

        # Export best model
        best_model_path = checkpoint_dir / 'best_model.pth'
        if best_model_path.exists():
            trainer.load_checkpoint(str(best_model_path))
            export_path = output_dir / f'{args.model}_trained.pth'
            # Calculate scaled size for export
            scaled_size = int(384 * args.scale)
            scaled_size = max(32, scaled_size)
            export_model_for_inference(trainer.model, str(export_path),
                                      example_input_size=(1, 3, scaled_size, scaled_size))

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Best model: {checkpoint_dir / 'best_model.pth'}")
        print(f"Exported model: {output_dir / f'{args.model}_trained.pth'}")
        print(f"Training history: {output_dir / 'training_history.png'}")
        print("="*70)

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current state...")
        trainer.save_checkpoint(epoch=len(trainer.history['train_loss']), is_best=False)
        return 1
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
