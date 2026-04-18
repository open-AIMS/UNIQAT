"""
Training infrastructure for deep learning quality assessment models.

Provides complete training pipeline with GPU acceleration, data augmentation,
and experiment tracking.
"""

import inspect
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class UnderwaterDataset(Dataset):
    """
    Dataset for underwater image quality assessment.

    Supports loading images with ground truth quality scores and metrics.
    """

    def __init__(
        self,
        image_paths: list[str],
        quality_scores: list[float] | None = None,
        blue_water_scores: list[float] | None = None,
        metrics: list[dict] | None = None,
        transform: Callable | None = None,
        augment: bool = False,
        scale: float = 1.0
    ):
        """
        Initialize dataset.

        Args:
            image_paths: List of image file paths
            quality_scores: Overall quality scores (0-100)
            blue_water_scores: Blue water severity scores (0-10)
            metrics: List of dictionaries containing all metrics
            transform: Image transformation function
            augment: Apply data augmentation
            scale: Scale factor for resizing images (default: 1.0)

        Raises:
            ValueError: If image_paths is empty or if label lists have mismatched lengths
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        self.image_paths = image_paths
        self.quality_scores = quality_scores if quality_scores else [None] * len(image_paths)
        self.blue_water_scores = blue_water_scores if blue_water_scores else [None] * len(image_paths)
        self.metrics = metrics if metrics else [None] * len(image_paths)
        self.augment = augment

        # Validate list lengths
        if len(self.quality_scores) != len(image_paths):
            raise ValueError(f"Length mismatch: {len(image_paths)} images but {len(self.quality_scores)} quality scores")
        if len(self.blue_water_scores) != len(image_paths):
            raise ValueError(f"Length mismatch: {len(image_paths)} images but {len(self.blue_water_scores)} blue water scores")
        if len(self.metrics) != len(image_paths):
            raise ValueError(f"Length mismatch: {len(image_paths)} images but {len(self.metrics)} metrics")

        # Calculate target image size based on scale factor
        base_size = 384
        target_size = int(base_size * scale)
        # Ensure minimum size of 32x32 to avoid degenerate cases
        target_size = max(32, target_size)

        # Define augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(target_size, target_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(p=0.3),
                    A.RandomGamma(p=0.3),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(target_size, target_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing image tensors and labels

        Raises:
            IOError: If image cannot be loaded
            ValueError: If image is invalid
        """
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        if image is None:
            raise OSError(f"Failed to load image: {image_path}")

        if image.size == 0:
            raise ValueError(f"Image is empty: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms to RGB
        augmented = self.transform(image=image)
        image_tensor = augmented['image']

        # FIXED: Convert the AUGMENTED image to LAB (not the original)
        # This ensures RGB and LAB tensors have matching augmentations
        # Get the augmented image back from tensor for LAB conversion
        # Since albumentations normalizes, we need to denormalize first
        augmented_np = image_tensor.permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented_np = (augmented_np * std + mean) * 255
        augmented_np = np.clip(augmented_np, 0, 255).astype(np.uint8)

        # Convert to LAB
        image_lab = cv2.cvtColor(augmented_np, cv2.COLOR_RGB2LAB)

        # Normalize LAB and convert to tensor
        lab_normalized = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        lab_augmented = lab_normalized(image=image_lab)
        lab_tensor = lab_augmented['image']

        sample = {
            'image': image_tensor,
            'image_lab': lab_tensor,
            'path': image_path
        }

        # Add labels if available
        if self.quality_scores[idx] is not None:
            sample['quality_score'] = torch.tensor(self.quality_scores[idx], dtype=torch.float32)

        if self.blue_water_scores[idx] is not None:
            sample['blue_water_severity'] = torch.tensor(self.blue_water_scores[idx], dtype=torch.float32)

        if self.metrics[idx] is not None:
            # Convert metrics dict to tensor
            sample['metrics'] = self.metrics[idx]

        return sample


class QualityAssessmentTrainer:
    """
    Trainer for quality assessment models with GPU acceleration.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Log every N batches

        Raises:
            ValueError: If train_loader is empty or model is invalid
        """
        if len(train_loader) == 0:
            raise ValueError("train_loader cannot be empty")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        # Check if model accepts dual inputs (RGB + LAB)
        # FIXED: Use inspect to check forward() signature instead of attribute checking
        forward_sig = inspect.signature(model.forward)
        self.model_accepts_dual_input = len(forward_sig.parameters) >= 2

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss

        Raises:
            RuntimeError: If no valid batches found in training data
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)

            # Get labels
            quality_scores = batch.get('quality_score')
            blue_water_scores = batch.get('blue_water_severity')

            if quality_scores is None:
                continue  # Skip if no labels

            quality_scores = quality_scores.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # FIXED: Use pre-computed check instead of hasattr
            if 'image_lab' in batch and self.model_accepts_dual_input:
                images_lab = batch['image_lab'].to(self.device)
                outputs = self.model(images, images_lab)
            else:
                outputs = self.model(images)

            # Compute loss
            loss = self.compute_loss(outputs, quality_scores, blue_water_scores)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': loss.item()})

        # FIXED: Raise error if no valid batches
        if num_batches == 0:
            raise RuntimeError("No valid batches found in training data (all batches missing quality_score labels)")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> tuple[float, float]:
        """
        Validate model.

        Returns:
            Tuple of (average loss, average MAE)

        Raises:
            RuntimeError: If validation set has no valid batches
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                quality_scores = batch.get('quality_score')

                if quality_scores is None:
                    continue

                quality_scores = quality_scores.to(self.device)
                blue_water_scores = batch.get('blue_water_severity')

                # Forward pass
                if 'image_lab' in batch and self.model_accepts_dual_input:
                    images_lab = batch['image_lab'].to(self.device)
                    outputs = self.model(images, images_lab)
                else:
                    outputs = self.model(images)

                # Compute metrics
                loss = self.compute_loss(outputs, quality_scores, blue_water_scores)

                # FIXED: Handle shape compatibility for MAE calculation
                pred_quality = outputs['quality_score']
                if pred_quality.dim() == 2 and pred_quality.size(1) == 1:
                    pred_quality = pred_quality.squeeze(1)

                if quality_scores.dim() == 2 and quality_scores.size(1) == 1:
                    quality_scores = quality_scores.squeeze(1)

                mae = torch.abs(pred_quality - quality_scores).mean()

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

        # FIXED: Raise error if no valid batches
        if num_batches == 0:
            raise RuntimeError("No valid batches found in validation data (all batches missing quality_score labels)")

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        return avg_loss, avg_mae

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        quality_scores: torch.Tensor,
        blue_water_scores: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs
            quality_scores: Ground truth quality scores
            blue_water_scores: Ground truth blue water scores

        Returns:
            Total loss

        Raises:
            ValueError: If output shapes are incompatible with targets
        """
        # FIXED: Ensure shape compatibility for quality score loss
        pred_quality = outputs['quality_score']

        # Handle shape mismatches
        if pred_quality.dim() == 2 and pred_quality.size(1) == 1:
            pred_quality = pred_quality.squeeze(1)

        if quality_scores.dim() == 2 and quality_scores.size(1) == 1:
            quality_scores = quality_scores.squeeze(1)

        if pred_quality.shape != quality_scores.shape:
            raise ValueError(
                f"Shape mismatch: predicted quality {pred_quality.shape} vs "
                f"target quality {quality_scores.shape}"
            )

        # Quality score loss
        quality_loss = self.mse_loss(pred_quality, quality_scores)

        total_loss = quality_loss

        # Blue water severity loss (if available)
        if blue_water_scores is not None and 'blue_water_severity' in outputs:
            blue_water_scores = blue_water_scores.to(self.device)

            # FIXED: Handle shape compatibility for blue water scores
            pred_bw = outputs['blue_water_severity']
            if pred_bw.dim() == 2 and pred_bw.size(1) == 1:
                pred_bw = pred_bw.squeeze(1)

            if blue_water_scores.dim() == 2 and blue_water_scores.size(1) == 1:
                blue_water_scores = blue_water_scores.squeeze(1)

            if pred_bw.shape != blue_water_scores.shape:
                raise ValueError(
                    f"Shape mismatch: predicted blue water {pred_bw.shape} vs "
                    f"target blue water {blue_water_scores.shape}"
                )

            bw_loss = self.mse_loss(pred_bw, blue_water_scores)
            total_loss = total_loss + 0.5 * bw_loss

        # Feature usefulness loss (if available and labels exist)
        if 'feature_usefulness' in outputs:
            # Can add additional loss terms here
            pass

        return total_loss

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ) -> dict:
        """
        Train model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
            save_best: Save best model checkpoint

        Returns:
            Training history

        Raises:
            ValueError: If num_epochs <= 0
        """
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")

        print(f"Training on device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss, val_mae = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val MAE: {val_mae:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0

                if save_best:
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            # Save periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)

        # Save final model
        self.save_checkpoint(num_epochs, is_best=False)
        self.save_history()

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model

        Raises:
            IOError: If checkpoint cannot be saved
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'

        try:
            torch.save(checkpoint, path)
        except Exception as e:
            raise OSError(f"Failed to save checkpoint to {path}: {e}") from e

    def save_history(self):
        """
        Save training history to JSON.

        Raises:
            IOError: If history cannot be saved
        """
        history_path = self.checkpoint_dir / 'training_history.json'

        try:
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            raise OSError(f"Failed to save training history to {history_path}: {e}") from e

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is corrupt or incompatible
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}") from e

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            self.history = checkpoint['history']
        except KeyError as e:
            raise RuntimeError(f"Checkpoint is missing required key: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to restore state from checkpoint: {e}") from e

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def create_synthetic_dataset(
    num_images: int = 1000,
    output_dir: str = 'synthetic_data',
    random_seed: int | None = 42
) -> tuple[list[str], list[float], list[float]]:
    """
    Create synthetic underwater image dataset for training.

    Generates images with varying degrees of blue water problem
    and corresponding quality scores.

    Args:
        num_images: Number of images to generate
        output_dir: Output directory
        random_seed: Random seed for reproducibility (None for non-deterministic)

    Returns:
        Tuple of (image_paths, quality_scores, blue_water_scores)

    Raises:
        ValueError: If num_images <= 0
        IOError: If output directory cannot be created or images cannot be saved
    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}")

    # FIXED: Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create output directory {output_dir}: {e}") from e

    image_paths = []
    quality_scores = []
    blue_water_scores = []

    print(f"Generating {num_images} synthetic images...")

    for i in tqdm(range(num_images)):
        # Generate random base image (simulated underwater scene)
        img = np.random.rand(384, 384, 3) * 255

        # Add varying degrees of blue water problem
        blue_severity = np.random.rand() * 10  # 0-10
        blue_factor = blue_severity / 10

        # FIXED: Apply blue tint with proper clipping to avoid negative values
        # Blue channel (increases)
        img[:, :, 2] = np.clip(
            img[:, :, 2] * (1 - blue_factor) + 255 * blue_factor,
            0, 255
        )
        # Green channel (moderate increase)
        img[:, :, 1] = np.clip(
            img[:, :, 1] * (1 - blue_factor * 0.7) + 200 * blue_factor,
            0, 255
        )
        # Red channel (decreases)
        img[:, :, 0] = np.clip(
            img[:, :, 0] * (1 - blue_factor * 0.5),
            0, 255
        )

        # Reduce contrast based on severity
        img = np.clip(
            img * (1 - blue_factor * 0.5) + 128 * blue_factor * 0.5,
            0, 255
        )

        # Add noise/turbidity (clip to prevent overflow)
        noise = np.random.randn(384, 384, 3) * blue_severity
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        # Calculate quality score (inverse of blue water severity)
        quality = 100 * (1 - blue_factor)

        # Save image
        img_path = output_dir / f'synthetic_{i:05d}.jpg'

        # FIXED: Error handling for cv2.imwrite
        success = cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            warnings.warn(f"Failed to save image {img_path} (disk full or permission error?)")
            continue

        image_paths.append(str(img_path))
        quality_scores.append(quality)
        blue_water_scores.append(blue_severity)

    if len(image_paths) == 0:
        raise OSError(f"Failed to generate any images in {output_dir}")

    print(f"Generated {len(image_paths)} synthetic images in {output_dir}")

    return image_paths, quality_scores, blue_water_scores
