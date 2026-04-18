#!/usr/bin/env python3
"""
Single image assessment script for underwater image quality.

This script analyzes a single underwater image and provides comprehensive
quality metrics, blue water problem detection, and detailed recommendations.
"""

import sys
import os
import argparse
from pathlib import Path

# Allow running from a fresh clone without `pip install -e .` by adding the
# sibling src/ directory to sys.path. No-op when the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uniqat.core.assessor import UnderwaterImageAssessor
from uniqat.utils.visualization import QualityVisualizer


def main():
    """Main function for single image assessment."""
    parser = argparse.ArgumentParser(
        description='Assess underwater image quality and detect blue water problem',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg
  %(prog)s image.jpg --output results/
  %(prog)s image.jpg --output results/ --visualize
  %(prog)s image.jpg --json-only
        """
    )

    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the underwater image to assess')
    parser.add_argument('-o', '--output', type=str, default="./output",
                       help='Output directory for results (default: ./output)')
    parser.add_argument('-v', '--visualize', action='store_true', default=True,
                       help='Generate visual reports')
    parser.add_argument('--json-only', action='store_true',
                       help='Output only JSON results (no text report)')
    parser.add_argument('--json-file', type=str, default=None,
                       help='Save JSON results to specified file')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress console output')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for resizing image (e.g., 0.5 for 50%% of original size). Default: 1.0 (no scaling)')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.", file=sys.stderr)
        return 1

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    # Get image name without extension
    image_name = Path(args.image_path).stem

    try:
        if not args.quiet:
            print(f"Analyzing image: {args.image_path}")
            print("Please wait, this may take a moment...")
            print()

        # Perform assessment
        assessor = UnderwaterImageAssessor(image_path=args.image_path, scale_factor=args.scale)
        assessment = assessor.assess()

        # Output results
        if args.json_only:
            print(assessment.to_json())
        else:
            # Print detailed report
            report = assessor.get_detailed_report()
            print(report)

            # Save report to file
            report_path = output_dir / f"{image_name}_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)

            if not args.quiet:
                print(f"\nText report saved to: {report_path}")

        # Save JSON if requested
        if args.json_file or args.output:
            json_path = Path(args.json_file) if args.json_file else output_dir / f"{image_name}_metrics.json"
            with open(json_path, 'w') as f:
                f.write(assessment.to_json())

            if not args.quiet:
                print(f"JSON metrics saved to: {json_path}")

        # Generate visualizations if requested
        if args.visualize:
            if not args.quiet:
                print("\nGenerating visualizations...")

            visualizer = QualityVisualizer(
                assessor.metrics_calculator.image,
                assessor.metrics,
                assessment.to_dict()
            )

            # Comprehensive report
            comprehensive_path = output_dir / f"{image_name}_comprehensive_report.png"
            visualizer.create_comprehensive_report(str(comprehensive_path))

            if not args.quiet:
                print(f"Comprehensive report saved to: {comprehensive_path}")

            # Side-by-side comparison
            comparison_path = output_dir / f"{image_name}_comparison.png"
            visualizer.create_side_by_side_comparison(str(comparison_path))

            if not args.quiet:
                print(f"Comparison image saved to: {comparison_path}")

        # Print summary
        if not args.quiet and not args.json_only:
            print("\n" + "=" * 70)
            print("QUICK SUMMARY")
            print("=" * 70)
            print(assessor.get_summary())

            # Check usability
            cv_usable, cv_reason = assessor.is_usable_for_cv_tasks()
            marine_usable, marine_reason = assessor.is_usable_for_marine_science()

            print("\nUSABILITY:")
            print(f"  Computer Vision: {'✓ YES' if cv_usable else '✗ NO'} - {cv_reason}")
            print(f"  Marine Science:  {'✓ YES' if marine_usable else '✗ NO'} - {marine_reason}")
            print("=" * 70)

        return 0

    except Exception as e:
        print(f"Error during assessment: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
