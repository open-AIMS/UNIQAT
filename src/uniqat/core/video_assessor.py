"""
Video quality assessment for underwater footage.

Extends the image quality assessment system to handle video files
with temporal analysis and frame-level quality tracking.
"""

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .assessor import QualityAssessment, UnderwaterImageAssessor


@dataclass
class VideoQualityAssessment:
    """
    Comprehensive video quality assessment results.

    Attributes
    ----------
    video_path : str
        Path to the video file
    duration_seconds : float
        Total duration in seconds
    fps : float
        Frames per second
    total_frames : int
        Total number of frames
    analyzed_frames : int
        Number of frames analyzed
    frame_assessments : list[dict]
        List of per-frame assessments
    temporal_metrics : dict
        Temporal consistency metrics
    overall_video_score : float
        Overall video quality score (0-100)
    average_frame_score : float
        Average quality across frames
    min_frame_score : float
        Minimum frame quality
    max_frame_score : float
        Maximum frame quality
    quality_std : float
        Standard deviation of quality scores
    blue_water_severity_avg : float
        Average blue water severity
    visibility_avg : float
        Average visibility
    temporal_stability : float
        Temporal consistency score (0-1)
    recommendations : list[str]
        List of recommendations
    warnings : list[str]
        List of warnings
    """
    video_path: str
    duration_seconds: float
    fps: float
    total_frames: int
    analyzed_frames: int
    frame_assessments: list[dict]
    temporal_metrics: dict
    overall_video_score: float
    average_frame_score: float
    min_frame_score: float
    max_frame_score: float
    quality_std: float
    blue_water_severity_avg: float
    visibility_avg: float
    temporal_stability: float
    recommendations: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class VideoQualityAssessor:
    """
    Comprehensive video quality assessor for underwater footage.

    Analyzes videos frame-by-frame and computes temporal consistency metrics.
    """

    def __init__(
        self,
        video_path: str,
        frame_skip: int = 30,
        max_frames: int | None = None,
        use_gpu: bool = False
    ):
        """
        Initialize video quality assessor.

        Parameters
        ----------
        video_path : str
            Path to the video file
        frame_skip : int
            Analyze every Nth frame (default: 30 = 1 frame/second at 30fps)
        max_frames : int | None
            Maximum number of frames to analyze
        use_gpu : bool
            Use GPU acceleration for processing

        Raises
        ------
        FileNotFoundError
            If video file doesn't exist
        ValueError
            If video cannot be opened or has invalid properties
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.frame_skip = max(1, frame_skip)  # Ensure at least 1
        self.max_frames = max_frames
        self.use_gpu = use_gpu

        # Open video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # FIXED: Validate video properties
        if self.fps <= 0:
            warnings.warn(f"Invalid FPS ({self.fps}), defaulting to 30")
            self.fps = 30.0

        if self.total_frames <= 0:
            raise ValueError(f"Video has no frames or invalid frame count: {self.total_frames}")

        self.duration = self.total_frames / self.fps

        # Frame assessments
        self.frame_assessments: list[QualityAssessment] = []
        self.frame_indices: list[int] = []

        # FIXED: Store temporal_metrics as instance variable
        self._temporal_metrics: dict = {}

    def __del__(self):
        """FIXED: Ensure video capture is always released."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def assess(self) -> VideoQualityAssessment:
        """
        Perform comprehensive video quality assessment.

        Returns
        -------
        VideoQualityAssessment
            VideoQualityAssessment object with all results

        Raises
        ------
        RuntimeError
            If no frames could be analyzed
        """
        print(f"Analyzing video: {self.video_path.name}")
        print(f"Duration: {self.duration:.2f}s, FPS: {self.fps:.2f}, Total frames: {self.total_frames}")

        # Determine frames to analyze
        frames_to_analyze = list(range(0, self.total_frames, self.frame_skip))
        if self.max_frames:
            frames_to_analyze = frames_to_analyze[:self.max_frames]

        print(f"Analyzing {len(frames_to_analyze)} frames...")

        # FIXED: Track failed frames for reporting
        failed_frames = 0

        # Analyze frames
        for frame_idx in tqdm(frames_to_analyze, desc="Processing frames"):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

            if not ret:
                failed_frames += 1
                continue

            # Assess frame
            try:
                assessor = UnderwaterImageAssessor(image_array=frame)
                assessment = assessor.assess()

                self.frame_assessments.append(assessment)
                self.frame_indices.append(frame_idx)

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                failed_frames += 1
                continue

        # Release video
        self.cap.release()
        self.cap = None

        # FIXED: Warn about failed frames
        if failed_frames > 0:
            warnings.warn(f"Failed to process {failed_frames}/{len(frames_to_analyze)} frames")

        # FIXED: Check if we have any valid assessments
        if len(self.frame_assessments) == 0:
            raise RuntimeError(
                f"No frames could be analyzed from video. "
                f"Attempted {len(frames_to_analyze)} frames, all failed."
            )

        # Compute temporal metrics
        self._temporal_metrics = self._compute_temporal_metrics()

        # FIXED: Safe extraction with proper error handling
        frame_scores = [a.overall_score for a in self.frame_assessments]
        blue_water_scores = [a.blue_water_problem_severity for a in self.frame_assessments]

        # FIXED: Safe visibility score extraction with fallback
        visibility_scores = []
        for a in self.frame_assessments:
            try:
                vis_score = a.detailed_metrics.get('visibility_score', 0.5)
                visibility_scores.append(vis_score)
            except (AttributeError, KeyError):
                visibility_scores.append(0.5)  # Default fallback

        average_frame_score = float(np.mean(frame_scores))
        min_frame_score = float(np.min(frame_scores))
        max_frame_score = float(np.max(frame_scores))
        quality_std = float(np.std(frame_scores))
        blue_water_avg = float(np.mean(blue_water_scores))
        visibility_avg = float(np.mean(visibility_scores))

        # Overall video score (penalize high variability)
        variability_penalty = min(quality_std / 20, 1.0) * 10
        overall_video_score = max(0.0, average_frame_score - variability_penalty)

        # Temporal stability
        temporal_stability = self._calculate_temporal_stability()

        # Generate recommendations and warnings
        recommendations = self._generate_video_recommendations()
        warnings_list = self._generate_video_warnings()

        # Create video assessment
        video_assessment = VideoQualityAssessment(
            video_path=str(self.video_path),
            duration_seconds=self.duration,
            fps=self.fps,
            total_frames=self.total_frames,
            analyzed_frames=len(self.frame_assessments),
            frame_assessments=[a.to_dict() for a in self.frame_assessments],
            temporal_metrics=self._temporal_metrics,
            overall_video_score=overall_video_score,
            average_frame_score=average_frame_score,
            min_frame_score=min_frame_score,
            max_frame_score=max_frame_score,
            quality_std=quality_std,
            blue_water_severity_avg=blue_water_avg,
            visibility_avg=visibility_avg,
            temporal_stability=temporal_stability,
            recommendations=recommendations,
            warnings=warnings_list
        )

        return video_assessment

    def _compute_temporal_metrics(self) -> dict:
        """
        Compute temporal consistency metrics.

        Returns
        -------
        dict
            Dictionary of temporal metrics
        """
        if len(self.frame_assessments) < 2:
            return {}

        metrics = {}

        # Extract time series data
        quality_scores = np.array([a.overall_score for a in self.frame_assessments])
        blue_water_scores = np.array([a.blue_water_problem_severity for a in self.frame_assessments])

        # FIXED: Safe visibility score extraction
        visibility_scores = []
        for a in self.frame_assessments:
            try:
                vis_score = a.detailed_metrics.get('visibility_score', 0.5)
                visibility_scores.append(vis_score)
            except (AttributeError, KeyError):
                visibility_scores.append(0.5)
        visibility_scores = np.array(visibility_scores)

        # Frame-to-frame differences
        quality_diffs = np.abs(np.diff(quality_scores))
        blue_water_diffs = np.abs(np.diff(blue_water_scores))
        visibility_diffs = np.abs(np.diff(visibility_scores))

        metrics['quality_mean_diff'] = float(np.mean(quality_diffs))
        metrics['quality_max_diff'] = float(np.max(quality_diffs))
        metrics['blue_water_mean_diff'] = float(np.mean(blue_water_diffs))
        metrics['visibility_mean_diff'] = float(np.mean(visibility_diffs))

        # Temporal variance
        metrics['quality_temporal_variance'] = float(np.var(quality_scores))
        metrics['blue_water_temporal_variance'] = float(np.var(blue_water_scores))

        # FIXED: Trend analysis - only need >= 2 points for linear fit
        if len(quality_scores) >= 2:
            x = np.arange(len(quality_scores))
            try:
                quality_trend = np.polyfit(x, quality_scores, 1)[0]
                blue_water_trend = np.polyfit(x, blue_water_scores, 1)[0]

                metrics['quality_trend'] = float(quality_trend)
                metrics['blue_water_trend'] = float(blue_water_trend)
            except np.RankWarning:
                warnings.warn("Insufficient variation for trend analysis")

        # Count quality transitions
        good_frames = np.sum(quality_scores >= 60)
        poor_frames = np.sum(quality_scores < 40)

        metrics['good_frames_count'] = int(good_frames)
        metrics['poor_frames_count'] = int(poor_frames)
        metrics['good_frames_percentage'] = float(good_frames / len(quality_scores) * 100)

        return metrics

    def _calculate_temporal_stability(self) -> float:
        """
        Calculate temporal stability score (0-1).

        Higher score means more consistent quality across frames.

        Returns
        -------
        float
            Stability score between 0 and 1
        """
        if len(self.frame_assessments) < 2:
            return 1.0

        quality_scores = np.array([a.overall_score for a in self.frame_assessments])

        # Normalized standard deviation
        norm_std = np.std(quality_scores) / 100.0

        # Frame-to-frame consistency
        diffs = np.abs(np.diff(quality_scores))
        mean_diff = np.mean(diffs)
        norm_diff = mean_diff / 100.0

        # Combine metrics (lower is better, so invert)
        stability = 1.0 - min(norm_std * 0.5 + norm_diff * 0.5, 1.0)

        return float(stability)

    def _generate_video_recommendations(self) -> list[str]:
        """
        Generate video-specific recommendations.

        Returns
        -------
        list[str]
            List of recommendation strings
        """
        recommendations = []

        if len(self.frame_assessments) == 0:
            return ["Unable to analyze video frames."]

        # Quality variability
        quality_std = np.std([a.overall_score for a in self.frame_assessments])
        if quality_std > 20:
            recommendations.append(
                f"HIGH VARIABILITY: Quality varies significantly (σ={quality_std:.1f}). "
                "Consider stabilizing camera or improving lighting consistency."
            )

        # Temporal stability
        stability = self._calculate_temporal_stability()
        if stability < 0.5:
            recommendations.append(
                f"LOW TEMPORAL STABILITY ({stability:.2f}): Quality changes rapidly between frames. "
                "This may indicate unstable capture conditions or water turbulence."
            )

        # Average quality
        avg_quality = np.mean([a.overall_score for a in self.frame_assessments])
        if avg_quality < 40:
            recommendations.append(
                f"POOR AVERAGE QUALITY ({avg_quality:.1f}/100): Video has consistently low quality. "
                "Consider recapturing with better equipment or in clearer conditions."
            )
        elif avg_quality > 70:
            recommendations.append(
                f"GOOD AVERAGE QUALITY ({avg_quality:.1f}/100): Video has good overall quality. "
                "Suitable for analysis and documentation."
            )

        # Blue water problem
        avg_blue_water = np.mean([a.blue_water_problem_severity for a in self.frame_assessments])
        if avg_blue_water > 7:
            recommendations.append(
                f"SEVERE BLUE WATER PROBLEM (severity={avg_blue_water:.1f}/10): "
                "Apply color correction, white balance adjustment, or underwater image enhancement "
                "algorithms (e.g., UDCP, Sea-thru, Fusion-based methods)."
            )

        # FIXED: Check instance variable instead of undefined variable
        # Visibility trends
        if 'quality_trend' in self._temporal_metrics:
            trend = self._temporal_metrics['quality_trend']
            if trend < -0.5:
                recommendations.append(
                    "DECLINING QUALITY: Video quality decreases over time. "
                    "This may indicate increasing turbidity or changing lighting conditions."
                )
            elif trend > 0.5:
                recommendations.append(
                    "IMPROVING QUALITY: Video quality increases over time. "
                    "Early frames may be less useful than later ones."
                )

        return recommendations

    def _generate_video_warnings(self) -> list[str]:
        """
        Generate video-specific warnings.

        Returns
        -------
        list[str]
            List of warning strings
        """
        warnings_list = []

        if len(self.frame_assessments) == 0:
            warnings_list.append("⚠️ CRITICAL: No frames could be analyzed.")
            return warnings_list

        # Check for extremely poor frames
        poor_frames = sum(1 for a in self.frame_assessments if a.overall_score < 20)
        poor_percentage = poor_frames / len(self.frame_assessments) * 100

        if poor_percentage > 50:
            warnings_list.append(
                f"⚠️ CRITICAL: {poor_percentage:.1f}% of frames have very poor quality (<20/100). "
                "Video may be unusable for analysis."
            )
        elif poor_percentage > 25:
            warnings_list.append(
                f"⚠️ HIGH: {poor_percentage:.1f}% of frames have poor quality. "
                "Significant portions of video may be unusable."
            )

        # Check for rapid quality changes
        if 'quality_max_diff' in self._temporal_metrics:
            max_diff = self._temporal_metrics['quality_max_diff']
            if max_diff > 40:
                warnings_list.append(
                    f"⚠️ HIGH: Abrupt quality changes detected (max change: {max_diff:.1f}). "
                    "Video may have scene cuts or sudden lighting changes."
                )

        # Blue water problem
        severe_blue_frames = sum(
            1 for a in self.frame_assessments
            if a.blue_water_problem_severity > 7
        )
        severe_blue_percentage = severe_blue_frames / len(self.frame_assessments) * 100

        if severe_blue_percentage > 30:
            warnings_list.append(
                f"⚠️ HIGH: {severe_blue_percentage:.1f}% of frames have severe blue water problem. "
                "Color correction is strongly recommended."
            )

        return warnings_list

    def create_quality_timeline(self, output_path: str):
        """
        Create a timeline visualization of video quality metrics.

        Parameters
        ----------
        output_path : str
            Path to save the visualization

        Raises
        ------
        RuntimeError
            If no frame assessments available
        ValueError
            If FPS is invalid
        """
        if len(self.frame_assessments) == 0:
            raise RuntimeError("No frame assessments available for visualization.")

        # FIXED: Validate FPS before division
        if self.fps <= 0:
            raise ValueError(f"Invalid FPS for timeline generation: {self.fps}")

        # Extract data
        timestamps = np.array(self.frame_indices) / self.fps
        quality_scores = [a.overall_score for a in self.frame_assessments]
        blue_water_scores = [a.blue_water_problem_severity for a in self.frame_assessments]

        # FIXED: Safe visibility extraction with fallback
        visibility_scores = []
        for a in self.frame_assessments:
            try:
                vis_score = a.detailed_metrics.get('visibility_score', 0.5) * 100
                visibility_scores.append(vis_score)
            except (AttributeError, KeyError):
                visibility_scores.append(50.0)  # Default 50%

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f'Video Quality Timeline: {self.video_path.name}', fontsize=16, fontweight='bold')

        # Plot 1: Overall Quality
        axes[0].plot(timestamps, quality_scores, linewidth=2, color='#3498db', label='Quality Score')
        axes[0].fill_between(timestamps, quality_scores, alpha=0.3, color='#3498db')
        axes[0].axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Good threshold')
        axes[0].axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Fair threshold')
        axes[0].set_ylabel('Quality Score', fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot 2: Blue Water Severity
        axes[1].plot(timestamps, blue_water_scores, linewidth=2, color='#e74c3c', label='Blue Water Severity')
        axes[1].fill_between(timestamps, blue_water_scores, alpha=0.3, color='#e74c3c')
        axes[1].axhline(y=7, color='red', linestyle='--', alpha=0.5, label='Severe threshold')
        axes[1].axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
        axes[1].set_ylabel('Blue Water Severity', fontweight='bold')
        axes[1].set_ylim(0, 10)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Plot 3: Visibility
        axes[2].plot(timestamps, visibility_scores, linewidth=2, color='#2ecc71', label='Visibility Score')
        axes[2].fill_between(timestamps, visibility_scores, alpha=0.3, color='#2ecc71')
        axes[2].set_xlabel('Time (seconds)', fontweight='bold')
        axes[2].set_ylabel('Visibility Score', fontweight='bold')
        axes[2].set_ylim(0, 100)
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()

        # FIXED: Error handling for file save
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Quality timeline saved to: {output_path}")
        except Exception as e:
            raise OSError(f"Failed to save timeline to {output_path}: {e}") from e
        finally:
            plt.close()

    def extract_best_frames(
        self,
        num_frames: int = 10,
        output_dir: str | None = None
    ) -> list[tuple[int, float]]:
        """
        Extract best quality frames from video.

        Parameters
        ----------
        num_frames : int
            Number of frames to extract
        output_dir : str | None
            Directory to save frames (optional)

        Returns
        -------
        list[tuple[int, float]]
            List of (frame_index, quality_score) tuples

        Raises
        ------
        RuntimeError
            If no frame assessments available
        ValueError
            If num_frames <= 0
        """
        if len(self.frame_assessments) == 0:
            raise RuntimeError("No frame assessments available for extraction")

        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        # Sort frames by quality
        frame_quality_pairs = [
            (self.frame_indices[i], self.frame_assessments[i].overall_score)
            for i in range(len(self.frame_assessments))
        ]
        frame_quality_pairs.sort(key=lambda x: x[1], reverse=True)

        # Get top N frames
        best_frames = frame_quality_pairs[:num_frames]

        # Save frames if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # FIXED: Use try-finally to ensure cap is released
            cap = None
            try:
                cap = cv2.VideoCapture(str(self.video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to reopen video: {self.video_path}")

                saved_count = 0
                for frame_idx, quality in best_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if ret:
                        timestamp = frame_idx / self.fps
                        output_path = output_dir / f'frame_{frame_idx:06d}_quality_{quality:.1f}_t{timestamp:.2f}s.jpg'

                        # FIXED: Error handling for cv2.imwrite
                        success = cv2.imwrite(str(output_path), frame)
                        if not success:
                            warnings.warn(f"Failed to save frame {frame_idx} to {output_path}")
                        else:
                            saved_count += 1

                print(f"Extracted {saved_count}/{len(best_frames)} best frames to {output_dir}")

            finally:
                if cap is not None:
                    cap.release()

        return best_frames

    def get_summary_report(self) -> str:
        """
        Generate a text summary report.

        Returns
        -------
        str
            Formatted summary string
        """
        report = []
        report.append("=" * 70)
        report.append("VIDEO QUALITY ASSESSMENT SUMMARY")
        report.append("=" * 70)
        report.append(f"Video: {self.video_path.name}")
        report.append(f"Duration: {self.duration:.2f}s")
        report.append(f"FPS: {self.fps:.2f}")
        report.append(f"Total Frames: {self.total_frames}")
        report.append(f"Analyzed Frames: {len(self.frame_assessments)}")
        report.append("")

        if len(self.frame_assessments) == 0:
            report.append("No frames could be analyzed.")
            return "\n".join(report)

        avg_quality = np.mean([a.overall_score for a in self.frame_assessments])
        report.append(f"Average Quality Score: {avg_quality:.2f}/100")
        report.append(f"Quality Range: {np.min([a.overall_score for a in self.frame_assessments]):.2f} - {np.max([a.overall_score for a in self.frame_assessments]):.2f}")
        report.append(f"Quality Std Dev: {np.std([a.overall_score for a in self.frame_assessments]):.2f}")
        report.append(f"Temporal Stability: {self._calculate_temporal_stability():.3f}")
        report.append("")

        avg_blue_water = np.mean([a.blue_water_problem_severity for a in self.frame_assessments])
        report.append(f"Average Blue Water Severity: {avg_blue_water:.2f}/10")

        # FIXED: Safe visibility extraction
        visibility_scores = []
        for a in self.frame_assessments:
            try:
                vis_score = a.detailed_metrics.get('visibility_score', 0.5)
                visibility_scores.append(vis_score)
            except (AttributeError, KeyError):
                visibility_scores.append(0.5)

        avg_visibility = np.mean(visibility_scores)
        report.append(f"Average Visibility: {avg_visibility:.3f}")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)
