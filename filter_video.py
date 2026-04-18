#!/usr/bin/env python3
"""
Filter bad frames from underwater video using UNIQAT quality assessment.

Reads each frame, assesses quality, and writes only good frames to output video.
"""

import sys
import cv2
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm

# Allow running from a fresh clone without `pip install -e .` by adding the
# sibling src/ directory to sys.path. No-op when the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uniqat.core.assessor import UnderwaterImageAssessor


def filter_video(input_path, output_path, threshold=40.0, report_path=None):
    """
    Remove bad frames from video based on UNIQAT quality scores.

    Args:
        input_path: Path to input video
        output_path: Path for filtered output video
        threshold: Minimum overall_score to keep a frame (default: 40)
        report_path: Optional path to save frame-by-frame JSON report
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input: {Path(input_path).name}")
    print(f"  {total_frames} frames, {fps:.2f} fps, {width}x{height}")
    print(f"  Duration: {total_frames/fps:.1f}s")
    print(f"  Quality threshold: {threshold}")
    print()

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    kept = 0
    removed = 0
    scores = []
    frame_report = []

    start_time = time.time()

    for frame_idx in tqdm(range(total_frames), desc="Assessing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        try:
            assessor = UnderwaterImageAssessor(image_array=frame)
            assessment = assessor.assess()
            score = assessment.overall_score
            blue_water = assessment.blue_water_problem_severity
            category = assessment.usability_category
        except Exception as e:
            # If assessment fails, keep the frame to be safe
            score = 50.0
            blue_water = 0.0
            category = "Unknown"

        scores.append(score)
        frame_report.append({
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 2),
            "overall_score": round(score, 1),
            "blue_water_severity": round(blue_water, 1),
            "usability": category,
            "kept": score >= threshold
        })

        if score >= threshold:
            out.write(frame)
            kept += 1
        else:
            removed += 1

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    scores = np.array(scores)

    print()
    print(f"Done in {elapsed:.1f}s ({total_frames/elapsed:.1f} frames/sec)")
    print(f"  Kept:    {kept} frames ({100*kept/total_frames:.1f}%)")
    print(f"  Removed: {removed} frames ({100*removed/total_frames:.1f}%)")
    print(f"  Output duration: {kept/fps:.1f}s (was {total_frames/fps:.1f}s)")
    print(f"  Score range: {scores.min():.1f} - {scores.max():.1f}")
    print(f"  Mean score (all): {scores.mean():.1f}")
    print(f"  Mean score (kept): {scores[scores >= threshold].mean():.1f}" if kept > 0 else "")
    print(f"  Output: {output_path}")

    if report_path:
        with open(report_path, 'w') as f:
            json.dump({
                "input": str(input_path),
                "output": str(output_path),
                "threshold": threshold,
                "total_frames": total_frames,
                "kept_frames": kept,
                "removed_frames": removed,
                "fps": fps,
                "mean_score": round(float(scores.mean()), 1),
                "frames": frame_report
            }, f, indent=2)
        print(f"  Report: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter bad frames from underwater video using UNIQAT")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path (default: input_filtered.mp4)")
    parser.add_argument("-t", "--threshold", type=float, default=40.0,
                        help="Minimum quality score to keep a frame (0-100, default: 40)")
    parser.add_argument("--report", help="Save frame-by-frame JSON report")
    args = parser.parse_args()

    if not args.output:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_filtered.mp4")

    filter_video(args.input, args.output, args.threshold, args.report)
