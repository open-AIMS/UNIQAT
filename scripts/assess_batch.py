#!/usr/bin/env python3
"""
Batch assessment script for underwater image quality with HPC multiprocessing support.

This script processes multiple underwater images in parallel and generates a comparative
analysis report with statistics and rankings.

Features:
- Automatic CPU detection and optimal worker configuration
- HPC-aware: automatically uses 75% of cores on systems with 32+ CPUs
- Configurable parallelism via --workers argument
- Progress tracking with tqdm
- Supports both PC and HPC environments
- Falls back to single-threaded mode for small batches

Usage:
    # Auto-detect optimal workers
    python assess_batch.py images/

    # HPC with many cores
    python assess_batch.py images/ --workers 64

    # Single-threaded for debugging
    python assess_batch.py images/ --workers 1
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict
import json
import csv
from tqdm import tqdm
import numpy as np
import random
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# Allow running from a fresh clone without `pip install -e .` by adding the
# sibling src/ directory to sys.path. No-op when the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uniqat.core.assessor import UnderwaterImageAssessor
from uniqat.utils.visualization import QualityVisualizer


def find_images(directory: str, recursive: bool = False) -> List[str]:
    """
    Find all image files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of image file paths
    """
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


def select_images(image_paths: List[str], first: int = None, last: int = None,
                  random_count: int = None, random_seed: int = None, limit: int = None) -> List[str]:
    """
    Select a subset of images based on specified criteria.

    Args:
        image_paths: List of image paths to select from
        first: Select first N images
        last: Select last N images
        random_count: Select N random images
        random_seed: Seed for random selection (for reproducibility)
        limit: Maximum number of images to process (takes first N)

    Returns:
        Filtered list of image paths

    Raises:
        ValueError: If multiple selection methods are specified
    """
    # Count how many selection methods are specified
    selection_methods = sum([
        first is not None,
        last is not None,
        random_count is not None,
        limit is not None
    ])

    if selection_methods > 1:
        raise ValueError("Only one of --first, --last, --random, or --limit can be specified")

    if selection_methods == 0:
        return image_paths

    if first is not None:
        if first < 1:
            raise ValueError("--first must be a positive integer")
        return image_paths[:first]

    elif last is not None:
        if last < 1:
            raise ValueError("--last must be a positive integer")
        return image_paths[-last:] if last <= len(image_paths) else image_paths

    elif random_count is not None:
        if random_count < 1:
            raise ValueError("--random must be a positive integer")
        if random_seed is not None:
            random.seed(random_seed)
        # Select random images without replacement
        sample_size = min(random_count, len(image_paths))
        selected = random.sample(image_paths, sample_size)
        return sorted(selected)  # Return in sorted order for consistency

    elif limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        return image_paths[:limit]

    return image_paths


def get_optimal_workers(workers: int = None) -> int:
    """
    Determine optimal number of worker processes.

    Args:
        workers: User-specified worker count (None for auto-detect)

    Returns:
        Optimal number of workers for this system
    """
    if workers is not None:
        return max(1, workers)

    cpu_count = mp.cpu_count()

    # HPC detection: if CPU count is very high, assume HPC environment
    if cpu_count >= 32:
        # On HPC, use 75% of cores to leave some for system
        optimal = int(cpu_count * 0.75)
        print(f"HPC environment detected ({cpu_count} CPUs). Using {optimal} workers.")
    elif cpu_count >= 8:
        # On workstation, use CPU count - 2 to keep system responsive
        optimal = max(1, cpu_count - 2)
        print(f"Workstation detected ({cpu_count} CPUs). Using {optimal} workers.")
    else:
        # On PC, use CPU count - 1
        optimal = max(1, cpu_count - 1)
        print(f"Using {optimal} workers ({cpu_count} CPUs available).")

    return optimal


def process_single_image(image_path: str, output_dir: Path, visualize: bool = False,
                        scale_factor: float = 1.0) -> Dict:
    """
    Process a single image (worker function for multiprocessing).

    Args:
        image_path: Path to image
        output_dir: Output directory for visualizations
        visualize: Whether to create visualizations
        scale_factor: Scaling factor for resizing images

    Returns:
        Assessment result dictionary
    """
    try:
        # Perform assessment
        assessor = UnderwaterImageAssessor(image_path=image_path, scale_factor=scale_factor)
        assessment = assessor.assess()

        # Add filename to results
        result = assessment.to_dict()
        result['filename'] = os.path.basename(image_path)
        result['filepath'] = image_path

        # Generate visualization if requested
        if visualize:
            image_name = Path(image_path).stem
            visualizer = QualityVisualizer(
                assessor.metrics_calculator.image,
                assessor.metrics,
                assessment.to_dict()
            )

            vis_output_dir = output_dir / "visualizations"
            vis_output_dir.mkdir(exist_ok=True, parents=True)

            comprehensive_path = vis_output_dir / f"{image_name}_report.png"
            visualizer.create_comprehensive_report(str(comprehensive_path))

        return result

    except Exception as e:
        return {
            'filename': os.path.basename(image_path),
            'filepath': image_path,
            'error': str(e)
        }


def assess_batch(image_paths: List[str], output_dir: Path,
                visualize: bool = False, quiet: bool = False, scale_factor: float = 1.0,
                workers: int = None) -> List[Dict]:
    """
    Assess a batch of images using multiprocessing.

    Args:
        image_paths: List of image paths
        output_dir: Output directory
        visualize: Whether to create visualizations
        quiet: Suppress progress output
        scale_factor: Scaling factor for resizing images (e.g., 0.5 for 50% size)
        workers: Number of worker processes (None for auto-detect)

    Returns:
        List of assessment dictionaries
    """
    # Determine number of workers
    num_workers = get_optimal_workers(workers)

    # Single-threaded mode for small batches or when workers=1
    if num_workers == 1 or len(image_paths) < 4:
        if not quiet and num_workers == 1:
            print("Running in single-threaded mode...")
        return _assess_batch_sequential(image_paths, output_dir, visualize, quiet, scale_factor)

    # Multiprocessing mode
    if not quiet:
        print(f"Processing {len(image_paths)} images with {num_workers} parallel workers...")

    results = []

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create partial function with fixed arguments
    worker_func = partial(
        process_single_image,
        output_dir=output_dir,
        visualize=visualize,
        scale_factor=scale_factor
    )

    # Process images in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(worker_func, img_path): img_path
                         for img_path in image_paths}

        # Collect results with progress bar
        with tqdm(total=len(image_paths), desc="Processing images", disable=quiet) as pbar:
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update progress bar description
                    if not quiet and 'error' not in result:
                        pbar.set_postfix_str(f"✓ {Path(img_path).name[:30]}")
                    elif not quiet:
                        pbar.set_postfix_str(f"✗ {Path(img_path).name[:30]}")

                except Exception as e:
                    if not quiet:
                        print(f"\nUnexpected error processing {img_path}: {e}", file=sys.stderr)
                    results.append({
                        'filename': os.path.basename(img_path),
                        'filepath': img_path,
                        'error': str(e)
                    })

                pbar.update(1)

    return results


def _assess_batch_sequential(image_paths: List[str], output_dir: Path,
                             visualize: bool = False, quiet: bool = False,
                             scale_factor: float = 1.0) -> List[Dict]:
    """
    Sequential (single-threaded) batch assessment.

    Used for small batches or when multiprocessing is disabled.
    """
    results = []
    iterator = tqdm(image_paths, desc="Processing images", disable=quiet)

    for image_path in iterator:
        # Update progress description
        if not quiet:
            iterator.set_description(f"Processing {Path(image_path).name}")

        result = process_single_image(image_path, output_dir, visualize, scale_factor)
        results.append(result)

        if not quiet and 'error' in result:
            print(f"\nError processing {image_path}: {result['error']}", file=sys.stderr)

    return results


def generate_summary_statistics(results: List[Dict]) -> Dict:
    """Generate summary statistics from batch results."""
    # Filter out failed assessments
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        return {}

    stats = {
        'total_images': len(results),
        'successful_assessments': len(valid_results),
        'failed_assessments': len(results) - len(valid_results),
        'average_overall_score': np.mean([r['overall_score'] for r in valid_results]),
        'average_feature_usefulness': np.mean([r['feature_usefulness'] for r in valid_results]),
        'average_marine_science_value': np.mean([r['marine_science_value'] for r in valid_results]),
        'average_blue_water_severity': np.mean([r['blue_water_problem_severity'] for r in valid_results]),
        'category_distribution': {},
        'high_quality_images': [],
        'low_quality_images': [],
        'severe_blue_water_images': []
    }

    # Category distribution
    categories = [r['usability_category'] for r in valid_results]
    for cat in set(categories):
        stats['category_distribution'][cat] = categories.count(cat)

    # Sort images by score
    sorted_results = sorted(valid_results, key=lambda x: x['overall_score'], reverse=True)

    # High quality images (top 10 or score > 70)
    stats['high_quality_images'] = [
        {'filename': r['filename'], 'score': r['overall_score']}
        for r in sorted_results[:10]
        if r['overall_score'] > 70
    ]

    # Low quality images (score < 30)
    stats['low_quality_images'] = [
        {'filename': r['filename'], 'score': r['overall_score']}
        for r in sorted_results
        if r['overall_score'] < 30
    ]

    # Severe blue water problem images
    stats['severe_blue_water_images'] = [
        {'filename': r['filename'], 'severity': r['blue_water_problem_severity']}
        for r in valid_results
        if r['blue_water_problem_severity'] > 7
    ]

    return stats


def export_to_csv(results: List[Dict], csv_path: Path):
    """Export results to CSV file."""
    if not results:
        return

    # Define columns to export
    columns = [
        'filename',
        'overall_score',
        'usability_category',
        'feature_usefulness',
        'marine_science_value',
        'blue_water_problem_severity'
    ]

    # Add key metrics
    metric_columns = [
        'blue_water_severity',
        'turbidity_score',
        'visibility_score',
        'rms_contrast',
        'sharpness_gradient',
        'edge_density',
        'uciqe_score',
        'uiqm_score'
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = columns + metric_columns
        writer.writerow(header)

        # Write data
        for result in results:
            if 'error' in result:
                row = [result.get('filename', ''), 'ERROR'] + [''] * (len(header) - 2)
            else:
                row = [result.get(col, '') for col in columns]
                row += [result['detailed_metrics'].get(col, '') for col in metric_columns]
            writer.writerow(row)


def export_labels_for_training(results: List[Dict], labels_path: Path):
    """
    Export results as training labels for deep learning models.

    Creates a JSON file compatible with train_quality_model.py --labels argument.
    Format: {"image.jpg": {"quality_score": 75.5, "blue_water_severity": 3.2}, ...}

    Args:
        results: List of assessment results
        labels_path: Path to save labels JSON file
    """
    if not results:
        print("Warning: No results to export as labels")
        return

    # Filter out failed assessments
    valid_results = [r for r in results if 'error' not in r and 'overall_score' in r]

    if not valid_results:
        print("Warning: No valid results to export as labels")
        return

    # Create labels dictionary in training format
    labels = {}

    for result in valid_results:
        filename = result['filename']

        # Extract required fields for training
        labels[filename] = {
            'quality_score': float(result['overall_score']),
            'blue_water_severity': float(result.get('blue_water_problem_severity', 5.0))
        }

        # Optionally add feature usefulness if available
        if 'feature_usefulness' in result:
            labels[filename]['feature_usefulness'] = float(result['feature_usefulness'])

    # Save to JSON file
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TRAINING LABELS EXPORTED")
    print(f"{'='*70}")
    print(f"Labels file: {labels_path}")
    print(f"Total labeled images: {len(labels)}")
    print(f"Failed assessments (excluded): {len(results) - len(valid_results)}")
    print(f"\nTo train a model with these labels:")
    print(f"  python train_quality_model.py \\")
    print(f"    --data-dir /path/to/your/images \\")
    print(f"    --labels {labels_path.name} \\")
    print(f"    --model efficientnet \\")
    print(f"    --epochs 50")
    print(f"{'='*70}\n")


def main():
    """Main function for batch assessment."""
    parser = argparse.ArgumentParser(
        description='Batch assess underwater image quality with multiprocessing support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detects optimal workers)
  %(prog)s images/

  # Process with specific number of workers
  %(prog)s images/ --workers 16

  # HPC usage: process large dataset with many cores
  %(prog)s images/ -r --workers 64 --output results/

  # PC usage: single-threaded for debugging
  %(prog)s images/ --workers 1 --visualize

  # Process with visualization and CSV export
  %(prog)s images/ --output results/ --visualize --csv results.csv

  # Export labels for deep learning training
  %(prog)s images/ --save-labels training_labels.json

  # Complete workflow: assess images and prepare for DL training
  %(prog)s images/ --recursive --workers 16 \\
      --output assessment_results \\
      --save-labels training_labels.json \\
      --csv results.csv

  # Then train a deep learning model with the labels:
  # python train_quality_model.py \\
  #     --data-dir images/ \\
  #     --labels training_labels.json \\
  #     --model efficientnet --epochs 50

Performance Tips:
  - On HPC: Use --workers to match available cores (e.g., --workers 64)
  - On PC: Auto-detection works well, or use --workers 4-8 for typical systems
  - For small batches (<10 images): Single-threaded may be faster
  - With --visualize: Reduce workers slightly to avoid memory issues
        """
    )

    parser.add_argument('--paths', type=str, nargs='+', required=True,
                       help='Image files or directories to assess')
    parser.add_argument('-o', '--output', type=str, default="./output",
                       help='Output directory for results (default: ./output)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Search directories recursively')
    parser.add_argument('-v', '--visualize', action='store_true', default=False,
                       help='Generate visual reports for each image')
    parser.add_argument('--csv', type=str, default=None,
                       help='Export results to CSV file')
    parser.add_argument('--save-labels', '--export-labels', type=str, default=None,
                       help='Export results as training labels for deep learning (JSON format compatible with train_quality_model.py)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for resizing images (e.g., 0.5 for 50%% of original size). Default: 1.0 (no scaling)')

    # Multiprocessing arguments
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='Number of parallel worker processes. Default: auto-detect (75%% of CPUs on HPC, CPU-2 on workstation). Set to 1 for single-threaded mode.')

    # Image selection arguments
    parser.add_argument('--first', type=int, default=None,
                       help='Select and process only the first N images')
    parser.add_argument('--last', type=int, default=None,
                       help='Select and process only the last N images')
    parser.add_argument('--random', type=int, default=None, dest='random_count',
                       help='Select and process N random images')
    parser.add_argument('--random-seed', type=int, default=None,
                       help='Seed for random selection (for reproducibility)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of images to process (same as --first)')

    args = parser.parse_args()

    # Collect all image paths
    image_paths = []
    for path in args.paths:
        if os.path.isfile(path):
            image_paths.append(path)
        elif os.path.isdir(path):
            image_paths.extend(find_images(path, args.recursive))
        else:
            print(f"Warning: '{path}' not found, skipping.", file=sys.stderr)

    if not image_paths:
        print("Error: No images found to process.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Found {len(image_paths)} image(s) to process.")

    # Apply image selection filtering
    try:
        image_paths = select_images(
            image_paths,
            first=args.first,
            last=args.last,
            random_count=args.random_count,
            random_seed=args.random_seed,
            limit=args.limit
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Processing {len(image_paths)} image(s).")

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Assess batch
        results = assess_batch(image_paths, output_dir, args.visualize, args.quiet,
                              args.scale, args.workers)

        # Save detailed JSON results
        json_path = output_dir / "batch_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        if not args.quiet:
            print(f"\nDetailed results saved to: {json_path}")

        # Generate summary statistics
        stats = generate_summary_statistics(results)

        if stats:
            stats_path = output_dir / "summary_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

            if not args.quiet:
                print(f"Summary statistics saved to: {stats_path}")

            # Print summary to console
            if not args.quiet:
                print("\n" + "=" * 70)
                print("BATCH PROCESSING SUMMARY")
                print("=" * 70)
                print(f"Total images processed:           {stats['total_images']}")
                print(f"Successful assessments:           {stats['successful_assessments']}")
                print(f"Failed assessments:               {stats['failed_assessments']}")
                print(f"\nAverage overall score:            {stats['average_overall_score']:.2f}/100")
                print(f"Average feature usefulness:       {stats['average_feature_usefulness']:.2f}/100")
                print(f"Average marine science value:     {stats['average_marine_science_value']:.2f}/100")
                print(f"Average blue water severity:      {stats['average_blue_water_severity']:.2f}/10")

                print(f"\nCategory Distribution:")
                for cat, count in stats['category_distribution'].items():
                    percentage = (count / stats['successful_assessments']) * 100
                    print(f"  {cat:12} {count:3d} ({percentage:5.1f}%)")

                if stats['high_quality_images']:
                    print(f"\nHigh Quality Images ({len(stats['high_quality_images'])}):")
                    for img in stats['high_quality_images'][:5]:
                        print(f"  {img['filename']:40} {img['score']:.1f}/100")

                if stats['severe_blue_water_images']:
                    print(f"\nSevere Blue Water Problem ({len(stats['severe_blue_water_images'])}):")
                    for img in stats['severe_blue_water_images'][:5]:
                        print(f"  {img['filename']:40} Severity: {img['severity']:.1f}/10")

                print("=" * 70)

        # Export to CSV if requested
        if args.csv or args.output:
            csv_path = Path(args.csv) if args.csv else output_dir / "results.csv"
            export_to_csv(results, csv_path)

            if not args.quiet:
                print(f"\nCSV export saved to: {csv_path}")

        # Export training labels if requested
        if args.save_labels:
            labels_path = Path(args.save_labels)
            export_labels_for_training(results, labels_path)

        return 0

    except Exception as e:
        print(f"Error during batch processing: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()
    sys.exit(main())
