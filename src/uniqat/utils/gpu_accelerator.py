"""
GPU acceleration utilities for underwater image quality assessment.

Provides CUDA-accelerated batch processing and metric computation
for significant speedup on large datasets.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class GPUBatchProcessor:
    """
    GPU-accelerated batch processor for underwater image assessment.

    Uses CUDA for parallel processing of multiple images simultaneously.
    """

    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 8,
        num_workers: int = 4
    ):
        """
        Initialize GPU batch processor.

        Args:
            device: Device for computation ('cuda' or 'cpu')
            batch_size: Number of images to process simultaneously
            num_workers: Number of worker threads for data loading

        Raises:
            ValueError: If batch_size or num_workers are invalid
            RuntimeError: If CUDA device requested but not available
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}")

        # FIXED: Validate CUDA availability if requested
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"GPU Batch Processor initialized on: {device}")

        # FIXED: Check device count before accessing device properties
        if device == 'cuda' and torch.cuda.device_count() > 0:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif device == 'cuda':
            warnings.warn("CUDA is available but no GPU devices found")

    def process_batch_images(
        self,
        image_paths: list[str],
        model: torch.nn.Module | None = None
    ) -> list[dict]:
        """
        Process multiple images in batches with GPU acceleration.

        Args:
            image_paths: List of image paths
            model: Optional deep learning model for prediction

        Returns:
            List of assessment dictionaries

        Raises:
            ValueError: If image_paths is empty
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        results = []

        # Create batches
        num_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        print(f"Processing {len(image_paths)} images in {num_batches} batches...")

        for batch_idx in tqdm(range(num_batches), desc="Batch processing"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]

            # FIXED: Load images and track which ones succeeded
            batch_images, valid_paths = self._load_image_batch(batch_paths)

            # Process batch on GPU
            batch_results = self._process_gpu_batch(batch_images, valid_paths, model)

            results.extend(batch_results)

        return results

    def _load_image_batch(self, image_paths: list[str]) -> tuple[torch.Tensor, list[str]]:
        """
        Load a batch of images in parallel.

        Args:
            image_paths: List of image paths

        Returns:
            Tuple of (tensor of images (B, C, H, W), list of valid paths)
        """
        images = []
        valid_paths = []  # FIXED: Track which paths loaded successfully

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            loaded_images = list(executor.map(self._load_and_preprocess, image_paths))

        # Stack into batch and track valid paths
        for img, path in zip(loaded_images, image_paths):
            if img is not None:
                images.append(img)
                valid_paths.append(path)

        # FIXED: Return properly shaped empty tensor if no images loaded
        if not images:
            return torch.empty(0, 3, 384, 384), []

        return torch.stack(images), valid_paths

    def _load_and_preprocess(self, image_path: str) -> torch.Tensor | None:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor or None if loading failed
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                warnings.warn(f"Failed to load image: {image_path}")
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to standard size
            img = cv2.resize(img, (384, 384))

            # Normalize
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

            # Convert to tensor
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

            return img_tensor

        except Exception as e:
            warnings.warn(f"Error loading {image_path}: {e}")
            return None

    def _process_gpu_batch(
        self,
        batch: torch.Tensor,
        image_paths: list[str],
        model: torch.nn.Module | None = None
    ) -> list[dict]:
        """
        Process a batch of images on GPU.

        Args:
            batch: Tensor of images (B, C, H, W)
            image_paths: Corresponding image paths (same length as batch)
            model: Optional model for prediction

        Returns:
            List of result dictionaries
        """
        # FIXED: Check batch dimension properly
        if batch.size(0) == 0:
            return []

        # FIXED: Validate batch and paths alignment
        if batch.size(0) != len(image_paths):
            raise ValueError(
                f"Batch size {batch.size(0)} doesn't match paths length {len(image_paths)}"
            )

        # Move to GPU
        batch = batch.to(self.device)

        results = []

        with torch.no_grad():
            # Compute GPU-accelerated metrics
            if model is not None:
                # Use deep learning model
                predictions = model(batch)

                # FIXED: Validate prediction shapes
                if 'quality_score' not in predictions:
                    raise ValueError("Model output missing 'quality_score' key")

                quality_scores = predictions['quality_score']
                if quality_scores.size(0) != batch.size(0):
                    raise ValueError(
                        f"Model output batch size {quality_scores.size(0)} "
                        f"doesn't match input batch size {batch.size(0)}"
                    )

                for i in range(batch.size(0)):
                    result = {
                        'path': image_paths[i],
                        'quality_score': quality_scores[i].item() if quality_scores.dim() > 0 else quality_scores.item(),
                    }

                    # Optional outputs
                    if 'blue_water_severity' in predictions:
                        bw_scores = predictions['blue_water_severity']
                        result['blue_water_severity'] = bw_scores[i].item() if bw_scores.dim() > 0 else bw_scores.item()

                    if 'feature_usefulness' in predictions:
                        fu_scores = predictions['feature_usefulness']
                        result['feature_usefulness'] = fu_scores[i].item() if fu_scores.dim() > 0 else fu_scores.item()

                    results.append(result)

            else:
                # Compute traditional metrics on GPU
                for i in range(batch.size(0)):
                    metrics = self._compute_gpu_metrics(batch[i])
                    metrics['path'] = image_paths[i]
                    results.append(metrics)

        return results

    def _compute_gpu_metrics(self, image: torch.Tensor) -> dict:
        """
        Compute metrics on GPU for a single image.

        Args:
            image: Image tensor (C, H, W)

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Denormalize image for metric computation
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        img = image * std + mean
        img = torch.clamp(img * 255, 0, 255)

        # Color channel statistics (GPU)
        b_mean = img[2].mean().item()
        g_mean = img[1].mean().item()
        r_mean = img[0].mean().item()

        metrics['blue_channel_mean'] = b_mean
        metrics['green_channel_mean'] = g_mean
        metrics['red_channel_mean'] = r_mean

        # Blue dominance
        metrics['blue_dominance'] = (b_mean - r_mean) / 255.0

        # FIXED: Color temperature with better numerical stability
        metrics['color_temperature'] = (b_mean + g_mean) / max(r_mean, 1.0)

        # Grayscale conversion (GPU)
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]

        # Contrast (RMS) on GPU
        gray_mean = gray.mean()
        contrast = torch.sqrt(torch.mean((gray - gray_mean) ** 2)).item()
        metrics['rms_contrast'] = contrast / 255.0

        # Sharpness (Laplacian variance) on GPU
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        gray_4d = gray.unsqueeze(0).unsqueeze(0)
        laplacian = F.conv2d(gray_4d, laplacian_kernel, padding=1)
        sharpness = torch.var(laplacian).item()
        metrics['sharpness_laplacian'] = sharpness

        # Gradient magnitude (GPU)
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(gray_4d, sobel_x, padding=1)
        grad_y = F.conv2d(gray_4d, sobel_y, padding=1)
        gradient_mag = torch.sqrt(grad_x**2 + grad_y**2)
        metrics['sharpness_gradient'] = gradient_mag.mean().item()

        # Edge density (using gradient threshold)
        edge_threshold = gradient_mag.mean() + gradient_mag.std()
        edges = (gradient_mag > edge_threshold).float()
        metrics['edge_density'] = edges.mean().item()

        # Blue water severity estimation (simplified)
        blue_water_severity = metrics['blue_dominance'] * 5 + (1 - metrics['rms_contrast']) * 5
        metrics['blue_water_severity'] = float(np.clip(blue_water_severity, 0, 10))

        # Visibility estimation
        visibility = metrics['rms_contrast'] * 0.5 + (1 - abs(metrics['blue_dominance'])) * 0.5
        metrics['visibility_score'] = float(np.clip(visibility, 0, 1))

        return metrics


class CUDAMetricsAccelerator:
    """
    CUDA-accelerated metric computation for underwater images.

    Implements key quality metrics using PyTorch CUDA operations
    for 10-100x speedup over CPU implementations.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize CUDA metrics accelerator.

        Args:
            device: Device for computation ('cuda' or 'cpu')

        Raises:
            RuntimeError: If CUDA device requested but not available
        """
        # FIXED: Validate device availability
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available. "
                "Use device='cpu' for CPU computation."
            )

        self.device = torch.device(device)

        # Pre-compile kernels
        self._prepare_kernels()

    def _prepare_kernels(self):
        """Prepare convolution kernels on GPU."""
        # Sobel kernels
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        # Laplacian kernel
        self.laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        # Gaussian kernels for blur
        self.gaussian_3x3 = self._create_gaussian_kernel(3, 1.0)
        self.gaussian_5x5 = self._create_gaussian_kernel(5, 1.5)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Create Gaussian kernel on GPU.

        Args:
            kernel_size: Size of the kernel
            sigma: Standard deviation

        Returns:
            Gaussian kernel tensor
        """
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)

        # FIXED: Handle torch.meshgrid API changes across PyTorch versions
        try:
            # New API (PyTorch >= 1.10)
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        except TypeError:
            # Old API (PyTorch < 1.10)
            xx, yy = torch.meshgrid(ax, ax)

        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def compute_all_metrics_gpu(self, image: torch.Tensor) -> dict[str, float]:
        """
        Compute all metrics on GPU for maximum speed.

        Args:
            image: Image tensor (B, C, H, W) or (C, H, W)

        Returns:
            Dictionary of metrics

        Raises:
            ValueError: If image has invalid dimensions
        """
        # FIXED: Validate and handle dimensions properly
        if image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() == 2:
            raise ValueError("Image must be at least 3D (C, H, W), got 2D tensor")
        elif image.dim() > 4:
            raise ValueError(f"Image must be at most 4D (B, C, H, W), got {image.dim()}D tensor")

        if image.size(1) != 3:
            raise ValueError(f"Image must have 3 channels, got {image.size(1)}")

        image = image.to(self.device)

        metrics = {}

        # Convert to grayscale
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        gray = gray.unsqueeze(1)

        # Color statistics
        metrics.update(self._color_stats_gpu(image))

        # Contrast metrics
        metrics.update(self._contrast_gpu(gray))

        # Sharpness metrics
        metrics.update(self._sharpness_gpu(gray))

        # Edge metrics
        metrics.update(self._edge_metrics_gpu(gray))

        return metrics

    def _color_stats_gpu(self, image: torch.Tensor) -> dict[str, float]:
        """Compute color statistics on GPU."""
        b, g, r = image[:, 2], image[:, 1], image[:, 0]

        return {
            'blue_mean': b.mean().item(),
            'green_mean': g.mean().item(),
            'red_mean': r.mean().item(),
            'blue_std': b.std().item(),
            'green_std': g.std().item(),
            'red_std': r.std().item()
        }

    def _contrast_gpu(self, gray: torch.Tensor) -> dict[str, float]:
        """Compute contrast metrics on GPU."""
        # RMS contrast
        mean_val = gray.mean()
        rms_contrast = torch.sqrt(torch.mean((gray - mean_val) ** 2))

        # Michelson contrast
        max_val = gray.max()
        min_val = gray.min()
        michelson = (max_val - min_val) / (max_val + min_val + 1e-6)

        return {
            'rms_contrast': rms_contrast.item(),
            'michelson_contrast': michelson.item()
        }

    def _sharpness_gpu(self, gray: torch.Tensor) -> dict[str, float]:
        """Compute sharpness metrics on GPU."""
        # Laplacian variance
        lap = F.conv2d(gray, self.laplacian, padding=1)
        lap_var = torch.var(lap)

        # Gradient magnitude
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2)
        grad_mean = grad_mag.mean()

        return {
            'laplacian_variance': lap_var.item(),
            'gradient_magnitude': grad_mean.item()
        }

    def _edge_metrics_gpu(self, gray: torch.Tensor) -> dict[str, float]:
        """Compute edge metrics on GPU."""
        # Compute gradients
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2)

        # Edge density (threshold at mean + std)
        threshold = grad_mag.mean() + grad_mag.std()
        edges = (grad_mag > threshold).float()
        edge_density = edges.mean()

        return {
            'edge_density': edge_density.item()
        }


def benchmark_gpu_vs_cpu(image_path: str, num_iterations: int = 100):
    """
    Benchmark GPU vs CPU performance.

    Args:
        image_path: Path to test image
        num_iterations: Number of iterations to run

    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If image cannot be loaded or num_iterations <= 0
    """
    import time

    # FIXED: Validate inputs
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("=" * 70)
    print("GPU vs CPU Benchmark")
    print("=" * 70)

    # FIXED: Error handling for image loading
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (384, 384))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    except Exception as e:
        raise ValueError(f"Error loading/preprocessing image: {e}") from e

    # CPU benchmark
    print("\nCPU Benchmark...")
    img_cpu = img.to('cpu')
    accelerator_cpu = CUDAMetricsAccelerator(device='cpu')

    start = time.time()
    for _ in range(num_iterations):
        metrics = accelerator_cpu.compute_all_metrics_gpu(img_cpu)
    cpu_time = time.time() - start

    print(f"CPU Time: {cpu_time:.3f}s ({num_iterations} iterations)")
    print(f"Average: {cpu_time/num_iterations*1000:.2f}ms per image")

    # GPU benchmark
    if torch.cuda.is_available():
        print("\nGPU Benchmark...")
        img_gpu = img.to('cuda')
        accelerator_gpu = CUDAMetricsAccelerator(device='cuda')

        # Warmup
        for _ in range(10):
            metrics = accelerator_gpu.compute_all_metrics_gpu(img_gpu)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            metrics = accelerator_gpu.compute_all_metrics_gpu(img_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"GPU Time: {gpu_time:.3f}s ({num_iterations} iterations)")
        print(f"Average: {gpu_time/num_iterations*1000:.2f}ms per image")

        speedup = cpu_time / gpu_time
        print(f"\nSpeedup: {speedup:.2f}x faster on GPU")
    else:
        print("\nCUDA not available. Skipping GPU benchmark.")

    print("=" * 70)
