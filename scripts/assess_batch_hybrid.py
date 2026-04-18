#!/usr/bin/env python3
"""
Hybrid quality assessment combining traditional metrics + deep learning.

This script provides the best of both worlds:
- Traditional metrics: Interpretable, detailed breakdown
- Deep learning: Fast, accurate, end-to-end learning
- Ensemble: Combined predictions for maximum accuracy

Features:
- Computes all 37 traditional metrics
- Runs deep learning model prediction
- Combines predictions using weighted ensemble
- Provides detailed analysis with both approaches
- GPU acceleration for deep learning component

Usage:
    # Basic hybrid assessment
    python assess_batch_hybrid.py --model trained_model.pth --images images/

    # Custom ensemble weights (70% DL, 30% metrics)
    python assess_batch_hybrid.py --model model.pth --images images/ --dl-weight 0.7

    # Full detailed output with all metrics
    python assess_batch_hybrid.py --model model.pth --images images/ --detailed
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict
import json
import csv
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Allow running from a fresh clone without `pip install -e .` by adding the
# sibling src/ directory to sys.path. No-op when the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uniqat.core.assessor import UnderwaterImageAssessor
from uniqat.models.deep_models import (
    UnderwaterQualityNet,
    EfficientNetQualityNet,
    VisionTransformerQualityNet
)


def load_deep_model(model_path: str, architecture: str, device: str):
    """Load trained deep learning model."""
    print(f"Loading deep learning model ({architecture})...")

    if architecture == 'underwater':
        model = UnderwaterQualityNet(pretrained=False)
    elif architecture == 'efficientnet':
        model = EfficientNetQualityNet(pretrained=False)
    elif architecture == 'vit':
        model = VisionTransformerQualityNet(pretrained=False)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"✓ Deep learning model loaded on {device}")
    return model


def assess_single_hybrid(
    image_path: str,
    dl_model,
    device: str,
    architecture: str,
    dl_weight: float = 0.6,
    scale_factor: float = 1.0,
    compute_detailed: bool = False
) -> Dict:
    """
    Assess single image using hybrid approach.

    Args:
        image_path: Path to image
        dl_model: Deep learning model (or None to skip DL)
        device: Device for DL model
        architecture: DL model architecture
        dl_weight: Weight for DL prediction (1-dl_weight for metrics)
        scale_factor: Image scaling factor
        compute_detailed: Compute all detailed metrics

    Returns:
        Hybrid assessment result
    """
    result = {
        'filename': os.path.basename(image_path),
        'filepath': image_path
    }

    try:
        # 1. Traditional metrics assessment
        assessor = UnderwaterImageAssessor(image_path=image_path, scale_factor=scale_factor)
        traditional_assessment = assessor.assess()
        traditional_dict = traditional_assessment.to_dict()

        metrics_score = traditional_dict['overall_score']
        metrics_blue_water = traditional_dict.get('blue_water_problem_severity', 5.0)

        result['traditional_metrics'] = {
            'overall_score': metrics_score,
            'feature_usefulness': traditional_dict.get('feature_usefulness', 0),
            'marine_science_value': traditional_dict.get('marine_science_value', 0),
            'blue_water_severity': metrics_blue_water,
            'usability_category': traditional_dict.get('usability_category', 'Unknown')
        }

        if compute_detailed:
            result['traditional_metrics']['detailed'] = traditional_dict.get('detailed_metrics', {})

        # 2. Deep learning prediction (if model provided)
        if dl_model is not None:
            import cv2
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            # Load and preprocess image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

            augmented = transform(image=image_rgb)
            image_tensor = augmented['image'].unsqueeze(0).to(device)

            # Forward pass
            with torch.no_grad():
                if architecture == 'underwater':
                    # Dual input
                    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
                    lab_augmented = transform(image=image_lab)
                    lab_tensor = lab_augmented['image'].unsqueeze(0).to(device)
                    outputs = dl_model(image_tensor, lab_tensor)
                else:
                    outputs = dl_model(image_tensor)

            dl_score = float(outputs['quality_score'].cpu().item())
            dl_blue_water = float(outputs.get('blue_water_severity', torch.tensor([5.0])).cpu().item())

            result['deep_learning'] = {
                'overall_score': dl_score,
                'blue_water_severity': dl_blue_water
            }

            if 'feature_usefulness' in outputs:
                result['deep_learning']['feature_usefulness'] = float(outputs['feature_usefulness'].cpu().item())

            # 3. Hybrid ensemble
            hybrid_score = (dl_weight * dl_score) + ((1 - dl_weight) * metrics_score)
            hybrid_blue_water = (dl_weight * dl_blue_water) + ((1 - dl_weight) * metrics_blue_water)

            result['hybrid'] = {
                'overall_score': hybrid_score,
                'blue_water_severity': hybrid_blue_water,
                'dl_weight': dl_weight,
                'metrics_weight': 1 - dl_weight
            }

            # Categorize hybrid result
            if hybrid_score >= 80:
                result['hybrid']['usability_category'] = 'Excellent'
            elif hybrid_score >= 60:
                result['hybrid']['usability_category'] = 'Good'
            elif hybrid_score >= 40:
                result['hybrid']['usability_category'] = 'Fair'
            else:
                result['hybrid']['usability_category'] = 'Poor'

            # Final score (use hybrid)
            result['overall_score'] = hybrid_score
            result['usability_category'] = result['hybrid']['usability_category']

        else:
            # No DL model, use metrics only
            result['overall_score'] = metrics_score
            result['usability_category'] = result['traditional_metrics']['usability_category']

    except Exception as e:
        result['error'] = str(e)

    return result


def find_images(directory: str, recursive: bool = False) -> List[str]:
    """Find all image files in directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if Path(file).suffix.lower() in extensions:
                image_files.append(os.path.join(directory, file))

    return sorted(image_files)


def assess_batch_hybrid(
    image_paths: List[str],
    dl_model,
    device: str,
    architecture: str,
    dl_weight: float = 0.6,
    scale_factor: float = 1.0,
    workers: int = None,
    compute_detailed: bool = False,
    quiet: bool = False
) -> List[Dict]:
    """
    Assess batch of images using hybrid approach.

    Args:
        image_paths: List of image paths
        dl_model: Deep learning model
        device: Device for DL
        architecture: DL architecture
        dl_weight: Weight for DL predictions
        scale_factor: Image scaling
        workers: Number of workers for parallel processing
        compute_detailed: Compute all detailed metrics
        quiet: Suppress progress

    Returns:
        List of hybrid assessment results
    """
    import multiprocessing as mp

    if workers is None:
        cpu_count = mp.cpu_count()
        workers = max(1, cpu_count - 2) if cpu_count >= 8 else max(1, cpu_count - 1)

    if not quiet:
        print(f"Processing with {workers} workers (hybrid metrics + DL)...")

    # For hybrid, we process sequentially since DL model can't be pickled easily
    # Alternative: batch the DL predictions separately
    results = []

    for image_path in tqdm(image_paths, desc="Hybrid assessment", disable=quiet):
        result = assess_single_hybrid(
            image_path=image_path,
            dl_model=dl_model,
            device=device,
            architecture=architecture,
            dl_weight=dl_weight,
            scale_factor=scale_factor,
            compute_detailed=compute_detailed
        )
        results.append(result)

    return results


def generate_summary_statistics(results: List[Dict]) -> Dict:
    """Generate summary statistics comparing all approaches."""
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        return {}

    stats = {
        'total_images': len(results),
        'successful_assessments': len(valid_results),
        'failed_assessments': len(results) - len(valid_results)
    }

    # Compare scores from different methods
    if valid_results:
        # Traditional metrics
        traditional_scores = [r['traditional_metrics']['overall_score'] for r in valid_results]
        stats['traditional_metrics'] = {
            'mean': float(np.mean(traditional_scores)),
            'median': float(np.median(traditional_scores)),
            'std': float(np.std(traditional_scores))
        }

        # Deep learning (if available)
        if 'deep_learning' in valid_results[0]:
            dl_scores = [r['deep_learning']['overall_score'] for r in valid_results]
            stats['deep_learning'] = {
                'mean': float(np.mean(dl_scores)),
                'median': float(np.median(dl_scores)),
                'std': float(np.std(dl_scores))
            }

            # Hybrid
            hybrid_scores = [r['hybrid']['overall_score'] for r in valid_results]
            stats['hybrid'] = {
                'mean': float(np.mean(hybrid_scores)),
                'median': float(np.median(hybrid_scores)),
                'std': float(np.std(hybrid_scores))
            }

            # Correlation between methods
            stats['correlation'] = {
                'traditional_vs_dl': float(np.corrcoef(traditional_scores, dl_scores)[0, 1]),
                'traditional_vs_hybrid': float(np.corrcoef(traditional_scores, hybrid_scores)[0, 1]),
                'dl_vs_hybrid': float(np.corrcoef(dl_scores, hybrid_scores)[0, 1])
            }

        # Category distribution
        if 'usability_category' in valid_results[0]:
            categories = [r['usability_category'] for r in valid_results]
            stats['category_distribution'] = {
                cat: categories.count(cat) for cat in set(categories)
            }

    return stats


def export_to_csv(results: List[Dict], csv_path: Path, include_detailed: bool = False):
    """Export hybrid results to CSV."""
    if not results:
        return

    columns = [
        'filename',
        'overall_score',
        'usability_category',
        'traditional_score',
        'deep_learning_score',
        'hybrid_score'
    ]

    if include_detailed:
        columns.extend(['traditional_feature_usefulness', 'traditional_blue_water_severity'])

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)

        for result in results:
            if 'error' in result:
                row = [result.get('filename', ''), 'ERROR']
                writer.writerow(row)
            else:
                row = [
                    result.get('filename', ''),
                    result.get('overall_score', ''),
                    result.get('usability_category', ''),
                    result.get('traditional_metrics', {}).get('overall_score', ''),
                    result.get('deep_learning', {}).get('overall_score', ''),
                    result.get('hybrid', {}).get('overall_score', '')
                ]

                if include_detailed:
                    row.extend([
                        result.get('traditional_metrics', {}).get('feature_usefulness', ''),
                        result.get('traditional_metrics', {}).get('blue_water_severity', '')
                    ])

                writer.writerow(row)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Hybrid quality assessment: Traditional metrics + Deep learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic hybrid assessment (60% DL, 40% metrics)
  %(prog)s --model trained_model.pth --images images/

  # More weight to deep learning (80% DL, 20% metrics)
  %(prog)s --model model.pth --images images/ --dl-weight 0.8

  # More weight to traditional metrics (30% DL, 70% metrics)
  %(prog)s --model model.pth --images images/ --dl-weight 0.3

  # Full detailed metrics output
  %(prog)s --model model.pth --images images/ --detailed

  # Metrics only (no deep learning)
  %(prog)s --images images/ --no-dl

Benefits of Hybrid Approach:
  - Best accuracy: Combines strengths of both methods
  - Interpretability: Still have detailed metrics breakdown
  - Robustness: Reduces impact of individual method errors
  - Flexibility: Adjust weights based on your needs
        """
    )

    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images to assess')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained deep learning model (.pth file)')
    parser.add_argument('--arch', '--architecture', type=str, default='underwater',
                       choices=['underwater', 'efficientnet', 'vit'],
                       help='Model architecture (must match trained model)')

    parser.add_argument('--dl-weight', type=float, default=0.6,
                       help='Weight for deep learning prediction (0.0-1.0). Remaining weight goes to metrics.')
    parser.add_argument('--no-dl', action='store_true',
                       help='Skip deep learning (use traditional metrics only)')

    parser.add_argument('-o', '--output', type=str, default='hybrid_results',
                       help='Output directory for results')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Search image directory recursively')

    parser.add_argument('--scale', type=float, default=1.0,
                       help='Image scaling factor for metrics computation')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers for metrics computation')

    parser.add_argument('--detailed', action='store_true',
                       help='Include all detailed metrics in output')
    parser.add_argument('--csv', type=str, default=None,
                       help='Export results to CSV file')

    parser.add_argument('--device', type=str, default='auto',
                       help='Device for deep learning (auto, cuda, cpu)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Validate DL weight
    if args.dl_weight < 0.0 or args.dl_weight > 1.0:
        print("Error: --dl-weight must be between 0.0 and 1.0")
        return 1

    if not args.quiet:
        print("="*70)
        print("HYBRID QUALITY ASSESSMENT")
        print("="*70)
        print(f"Approach: {'Traditional Metrics Only' if args.no_dl else 'Hybrid (Metrics + Deep Learning)'}")
        if not args.no_dl:
            print(f"Ensemble weights: {args.dl_weight*100:.0f}% DL + {(1-args.dl_weight)*100:.0f}% Metrics")
        print("="*70)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load deep learning model (if not disabled)
    dl_model = None
    if not args.no_dl:
        if args.model is None:
            print("Error: --model required unless using --no-dl")
            return 1

        try:
            dl_model = load_deep_model(args.model, args.arch, device)
        except Exception as e:
            print(f"Error loading deep learning model: {e}")
            print("Continuing with traditional metrics only...")
            args.no_dl = True

    # Find images
    if os.path.isdir(args.images):
        image_paths = find_images(args.images, args.recursive)
    else:
        print(f"Error: {args.images} is not a directory")
        return 1

    if not image_paths:
        print("Error: No images found")
        return 1

    if not args.quiet:
        print(f"\nFound {len(image_paths)} images to process\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    try:
        results = assess_batch_hybrid(
            image_paths=image_paths,
            dl_model=dl_model,
            device=device,
            architecture=args.arch,
            dl_weight=args.dl_weight,
            scale_factor=args.scale,
            workers=args.workers,
            compute_detailed=args.detailed,
            quiet=args.quiet
        )

        # Save results
        json_path = output_dir / 'hybrid_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        if not args.quiet:
            print(f"\n✓ Results saved to: {json_path}")

        # Generate statistics
        stats = generate_summary_statistics(results)

        if stats:
            stats_path = output_dir / 'summary_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

            if not args.quiet:
                print(f"✓ Statistics saved to: {stats_path}")

                # Print summary
                print("\n" + "="*70)
                print("HYBRID ASSESSMENT SUMMARY")
                print("="*70)
                print(f"Total images:                 {stats['total_images']}")
                print(f"Successful:                   {stats['successful_assessments']}")
                print(f"Failed:                       {stats['failed_assessments']}")

                print(f"\nTraditional Metrics:")
                print(f"  Mean score:                 {stats['traditional_metrics']['mean']:.2f}")
                print(f"  Median score:               {stats['traditional_metrics']['median']:.2f}")
                print(f"  Std dev:                    {stats['traditional_metrics']['std']:.2f}")

                if 'deep_learning' in stats:
                    print(f"\nDeep Learning:")
                    print(f"  Mean score:                 {stats['deep_learning']['mean']:.2f}")
                    print(f"  Median score:               {stats['deep_learning']['median']:.2f}")
                    print(f"  Std dev:                    {stats['deep_learning']['std']:.2f}")

                    print(f"\nHybrid Ensemble:")
                    print(f"  Mean score:                 {stats['hybrid']['mean']:.2f}")
                    print(f"  Median score:               {stats['hybrid']['median']:.2f}")
                    print(f"  Std dev:                    {stats['hybrid']['std']:.2f}")

                    print(f"\nMethod Correlations:")
                    print(f"  Traditional vs DL:          {stats['correlation']['traditional_vs_dl']:.3f}")
                    print(f"  Traditional vs Hybrid:      {stats['correlation']['traditional_vs_hybrid']:.3f}")
                    print(f"  DL vs Hybrid:               {stats['correlation']['dl_vs_hybrid']:.3f}")

                if 'category_distribution' in stats:
                    print(f"\nCategory Distribution (Hybrid):")
                    for cat, count in stats['category_distribution'].items():
                        pct = (count / stats['successful_assessments']) * 100
                        print(f"  {cat:12} {count:4d} ({pct:5.1f}%)")

                print("="*70)

        # Export to CSV
        if args.csv or args.output:
            csv_path = Path(args.csv) if args.csv else output_dir / 'results.csv'
            export_to_csv(results, csv_path, include_detailed=args.detailed)
            if not args.quiet:
                print(f"\n✓ CSV export saved to: {csv_path}")

        return 0

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
