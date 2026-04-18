"""
Main underwater image quality assessor.

This module provides comprehensive assessment reports for underwater images,
including detailed analysis of the blue water problem and recommendations
for marine science applications.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np

from .metrics import UnderwaterMetrics


@dataclass
class QualityAssessment:
    """
    Comprehensive quality assessment results.

    Attributes
    ----------
    overall_score : float
        Overall quality score (0-100)
    feature_usefulness : float
        Usefulness for CV tasks (0-100)
    marine_science_value : float
        Value for marine science (0-100)
    blue_water_problem_severity : float
        Severity of blue water problem (0-10)
    usability_category : str
        Category (Excellent/Good/Fair/Poor/Unusable)
    detailed_metrics : dict
        All computed metrics
    recommendations : list[str]
        List of recommendations
    warnings : list[str]
        List of warnings about image quality
    """
    overall_score: float
    feature_usefulness: float
    marine_science_value: float
    blue_water_problem_severity: float
    usability_category: str
    detailed_metrics: dict
    recommendations: list[str]
    warnings: list[str]

    @property
    def feature_quality(self) -> float:
        """Alias for feature_usefulness (0-100)."""
        return self.feature_usefulness

    @property
    def colour_quality(self) -> float:
        """Colour quality derived from colour cast and contrast metrics (0-100)."""
        m = self.detailed_metrics
        colour_cast = m.get('blue_green_ratio', 0.5)
        contrast = m.get('rms_contrast', 0.0)
        # Penalise strong blue-green cast, reward contrast
        cast_score = max(0, 100 * (1 - abs(colour_cast - 0.5) * 2))
        contrast_score = min(100, contrast * 400)
        return round(0.6 * cast_score + 0.4 * contrast_score, 1)

    @property
    def blue_water_score(self) -> float:
        """Alias for blue_water_problem_severity (0-10)."""
        return self.blue_water_problem_severity

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['feature_quality'] = self.feature_quality
        d['colour_quality'] = self.colour_quality
        d['blue_water_score'] = self.blue_water_score
        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class UnderwaterImageAssessor:
    """
    Comprehensive underwater image quality assessor.

    This class analyzes underwater images for the blue water problem,
    feature richness, and overall quality for marine science applications.
    """

    # Thresholds for classification
    EXCELLENT_THRESHOLD = 80
    GOOD_THRESHOLD = 60
    FAIR_THRESHOLD = 40
    POOR_THRESHOLD = 20

    # Blue water problem severity levels
    BLUE_WATER_SEVERE = 0.7
    BLUE_WATER_MODERATE = 0.5
    BLUE_WATER_MILD = 0.3

    def __init__(self, image_path: str | None = None, image_array: np.ndarray | None = None, scale_factor: float = 1.0):
        """
        Initialize the assessor.

        Parameters
        ----------
        image_path : str | None
            Path to the image file
        image_array : np.ndarray | None
            Numpy array of the image (BGR format)
        scale_factor : float
            Scaling factor for resizing (e.g., 0.5 for 50% size). Default: 1.0 (no scaling)
        """
        self.metrics_calculator = UnderwaterMetrics(image_path=image_path, image_array=image_array, scale_factor=scale_factor)
        self.metrics: dict | None = None
        self.assessment: QualityAssessment | None = None

    def assess(self) -> QualityAssessment:
        """
        Perform comprehensive quality assessment.

        Returns
        -------
        QualityAssessment
            QualityAssessment object with all results
        """
        # Calculate all metrics
        self.metrics = self.metrics_calculator.calculate_all_metrics()

        # Calculate scores
        overall_score = self._calculate_overall_score()
        feature_usefulness = self.metrics['feature_usefulness_score']
        marine_science_value = self.metrics['marine_science_value']
        blue_water_severity = self._calculate_blue_water_severity()

        # Determine category
        usability_category = self._determine_category(overall_score)

        # Generate recommendations and warnings
        recommendations = self._generate_recommendations()
        warnings = self._generate_warnings()

        # Create assessment object
        self.assessment = QualityAssessment(
            overall_score=overall_score,
            feature_usefulness=feature_usefulness,
            marine_science_value=marine_science_value,
            blue_water_problem_severity=blue_water_severity,
            usability_category=usability_category,
            detailed_metrics=self.metrics,
            recommendations=recommendations,
            warnings=warnings
        )

        return self.assessment

    def _calculate_overall_score(self) -> float:
        """
        Calculate overall quality score (0-100).

        Combines multiple quality aspects with appropriate weighting.
        """
        # Weight different aspects
        feature_score = self.metrics['feature_usefulness_score'] * 0.35
        marine_score = self.metrics['marine_science_value'] * 0.35
        visibility_score = self.metrics['visibility_score'] * 100 * 0.20
        blue_water_penalty = self.metrics['blue_water_severity'] * 10

        overall = feature_score + marine_score + visibility_score - blue_water_penalty
        return float(np.clip(overall, 0, 100))

    def _calculate_blue_water_severity(self) -> float:
        """
        Calculate blue water problem severity on a 0-10 scale.

        Returns
        -------
        float
            Severity score (0=none, 10=severe)
        """
        blue_water = self.metrics['blue_water_severity']
        turbidity = self.metrics['turbidity_score']
        contrast_inv = 1 - np.clip(self.metrics['rms_contrast'] / 0.5, 0, 1)

        # Combine indicators
        severity = (blue_water * 5 + turbidity * 3 + contrast_inv * 2)
        return float(severity)

    def _determine_category(self, score: float) -> str:
        """Determine usability category based on score."""
        if score >= self.EXCELLENT_THRESHOLD:
            return "Excellent"
        elif score >= self.GOOD_THRESHOLD:
            return "Good"
        elif score >= self.FAIR_THRESHOLD:
            return "Fair"
        elif score >= self.POOR_THRESHOLD:
            return "Poor"
        else:
            return "Unusable"

    def _generate_recommendations(self) -> list[str]:
        """
        Generate actionable recommendations based on analysis.

        Returns
        -------
        list[str]
            List of recommendation strings
        """
        recommendations = []

        # Blue water problem recommendations
        if self.metrics['blue_water_severity'] > self.BLUE_WATER_SEVERE:
            recommendations.append(
                "CRITICAL: Severe blue water problem detected. Consider using "
                "white balance correction, red channel compensation, or specialized "
                "underwater image enhancement algorithms (e.g., UDCP, Sea-thru)."
            )
        elif self.metrics['blue_water_severity'] > self.BLUE_WATER_MODERATE:
            recommendations.append(
                "Moderate blue-green color cast detected. Apply color correction "
                "or histogram equalization to improve visibility."
            )

        # Contrast recommendations
        if self.metrics['rms_contrast'] < 0.2:
            recommendations.append(
                "Low contrast detected. Consider applying CLAHE (Contrast Limited "
                "Adaptive Histogram Equalization) or gamma correction."
            )

        # Blur recommendations
        if self.metrics['blur_estimate'] > 0.6:
            recommendations.append(
                "Significant blur detected. If possible, use deconvolution or "
                "sharpening filters. For future captures, ensure proper focus "
                "and reduce motion blur."
            )

        # Turbidity recommendations
        if self.metrics['turbidity_score'] > 0.7:
            recommendations.append(
                "High turbidity detected. Image may have limited scientific value. "
                "Consider capturing in clearer water conditions or using longer "
                "exposure with stabilization."
            )

        # Feature richness recommendations
        if self.metrics['edge_density'] < 0.05:
            recommendations.append(
                "Very low feature density. Image may contain insufficient detail "
                "for analysis. Verify subject distance and water clarity."
            )

        # Visibility recommendations
        if self.metrics['visibility_score'] < 0.3:
            recommendations.append(
                "Poor visibility conditions. Consider using artificial lighting, "
                "reducing camera-to-subject distance, or applying image enhancement "
                "techniques specifically designed for low-visibility scenarios."
            )

        # Sharpness recommendations
        if self.metrics['sharpness_laplacian'] < 100:
            recommendations.append(
                "Low sharpness detected. Consider using sharpening filters or "
                "unsharp masking. For future captures, ensure proper focus lock."
            )

        # Color diversity recommendations
        if self.metrics['hue_diversity'] < 3.0:
            recommendations.append(
                "Limited color diversity. This may indicate monochromatic conditions "
                "or severe color cast. Color restoration algorithms may help."
            )

        # Positive recommendations
        if self.metrics['feature_usefulness_score'] > 70:
            recommendations.append(
                "GOOD: Image has rich features suitable for computer vision tasks "
                "such as object detection, segmentation, and feature extraction."
            )

        if self.metrics['marine_science_value'] > 70:
            recommendations.append(
                "GOOD: Image has high scientific value with good visibility and "
                "clarity for marine biology research and species identification."
            )

        # If no major issues
        if not recommendations:
            recommendations.append(
                "Image quality is adequate for basic analysis. Minor enhancements "
                "may still improve results."
            )

        return recommendations

    def _generate_warnings(self) -> list[str]:
        """
        Generate warnings about critical quality issues.

        Returns
        -------
        list[str]
            List of warning strings
        """
        warnings = []

        # Critical warnings
        if self.metrics['feature_usefulness_score'] < 20:
            warnings.append(
                "⚠️ CRITICAL: Extremely low feature content. Image may be unusable "
                "for computer vision tasks."
            )

        if self.metrics['marine_science_value'] < 20:
            warnings.append(
                "⚠️ CRITICAL: Very low scientific value. Image has insufficient "
                "clarity and detail for marine research applications."
            )

        if self.metrics['blue_water_severity'] > self.BLUE_WATER_SEVERE:
            warnings.append(
                "⚠️ CRITICAL: Severe blue water problem. Significant color cast "
                "and contrast reduction detected."
            )

        if self.metrics['visibility_score'] < 0.2:
            warnings.append(
                "⚠️ CRITICAL: Extremely poor visibility. Image content may be "
                "barely discernible."
            )

        # High priority warnings
        if self.metrics['turbidity_score'] > 0.75:
            warnings.append(
                "⚠️ HIGH: Very high turbidity. Water clarity is severely compromised."
            )

        if self.metrics['michelson_contrast'] < 0.2:
            warnings.append(
                "⚠️ HIGH: Very low contrast. Image may appear washed out or flat."
            )

        if self.metrics['edge_density'] < 0.03:
            warnings.append(
                "⚠️ HIGH: Minimal edge content. Very few distinguishable features."
            )

        # Medium priority warnings
        if self.metrics['blur_estimate'] > 0.7:
            warnings.append(
                "⚠️ MEDIUM: Significant blur detected. Fine details may be lost."
            )

        if self.metrics['keypoint_density'] < 0.5:
            warnings.append(
                "⚠️ MEDIUM: Low keypoint density. Limited distinctive features "
                "for feature matching or tracking."
            )

        if self.metrics['entropy_gray'] < 4.0:
            warnings.append(
                "⚠️ MEDIUM: Low information content. Image has limited tonal variety."
            )

        # Color warnings
        if self.metrics['blue_dominance'] > 0.3:
            warnings.append(
                "⚠️ Blue channel dominance detected. Red/yellow subjects may appear "
                "severely attenuated."
            )

        if self.metrics['color_temperature'] > 2.0:
            warnings.append(
                "⚠️ High color temperature (blue-shifted). Warm colors are "
                "significantly reduced."
            )

        return warnings

    def get_detailed_report(self) -> str:
        """
        Generate a detailed human-readable report.

        Returns
        -------
        str
            Formatted string report
        """
        if self.assessment is None:
            self.assess()

        report = []
        report.append("=" * 70)
        report.append("UNDERWATER IMAGE QUALITY ASSESSMENT REPORT")
        report.append("=" * 70)
        report.append("")

        # Overall scores
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 70)
        report.append(f"Overall Quality Score:        {self.assessment.overall_score:.2f}/100")
        report.append(f"Usability Category:           {self.assessment.usability_category}")
        report.append(f"Feature Usefulness Score:     {self.assessment.feature_usefulness:.2f}/100")
        report.append(f"Marine Science Value:         {self.assessment.marine_science_value:.2f}/100")
        report.append(f"Blue Water Problem Severity:  {self.assessment.blue_water_problem_severity:.2f}/10")
        report.append("")

        # Blue water problem analysis
        report.append("BLUE WATER PROBLEM ANALYSIS")
        report.append("-" * 70)
        report.append(f"Blue-Green Color Cast:        {self.metrics['blue_water_severity']:.3f}")
        report.append(f"Blue Channel Dominance:       {self.metrics['blue_dominance']:.3f}")
        report.append(f"Green Channel Dominance:      {self.metrics['green_dominance']:.3f}")
        report.append(f"Color Temperature:            {self.metrics['color_temperature']:.3f}")
        report.append(f"Turbidity Score:              {self.metrics['turbidity_score']:.3f}")
        report.append("")

        # Contrast and sharpness
        report.append("CONTRAST & SHARPNESS")
        report.append("-" * 70)
        report.append(f"RMS Contrast:                 {self.metrics['rms_contrast']:.3f}")
        report.append(f"Michelson Contrast:           {self.metrics['michelson_contrast']:.3f}")
        report.append(f"Local Contrast:               {self.metrics['local_contrast']:.3f}")
        report.append(f"LAB Contrast:                 {self.metrics['lab_contrast']:.3f}")
        report.append(f"Sharpness (Laplacian):        {self.metrics['sharpness_laplacian']:.2f}")
        report.append(f"Sharpness (Gradient):         {self.metrics['sharpness_gradient']:.2f}")
        report.append(f"Blur Estimate:                {self.metrics['blur_estimate']:.3f}")
        report.append("")

        # Feature richness
        report.append("FEATURE RICHNESS")
        report.append("-" * 70)
        report.append(f"Edge Density:                 {self.metrics['edge_density']:.4f}")
        report.append(f"Multi-scale Edge Density:     {self.metrics['multi_scale_edge_density']:.4f}")
        report.append(f"Corner Density:               {self.metrics['corner_density']:.6f}")
        report.append(f"Keypoint Density:             {self.metrics['keypoint_density']:.4f}")
        report.append(f"Texture Complexity:           {self.metrics['texture_complexity']:.3f}")
        report.append("")

        # Underwater-specific metrics
        report.append("UNDERWATER-SPECIFIC METRICS")
        report.append("-" * 70)
        report.append(f"Visibility Score:             {self.metrics['visibility_score']:.3f}")
        report.append(f"UCIQE Score:                  {self.metrics['uciqe_score']:.4f}")
        report.append(f"UIQM Score:                   {self.metrics['uiqm_score']:.4f}")
        report.append("")

        # Information content
        report.append("INFORMATION CONTENT")
        report.append("-" * 70)
        report.append(f"Grayscale Entropy:            {self.metrics['entropy_gray']:.3f}")
        report.append(f"Color Entropy:                {self.metrics['entropy_color']:.3f}")
        report.append(f"Hue Diversity:                {self.metrics['hue_diversity']:.3f}")
        report.append("")

        # Warnings
        if self.assessment.warnings:
            report.append("WARNINGS")
            report.append("-" * 70)
            for warning in self.assessment.warnings:
                report.append(f"  {warning}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 70)
        for i, rec in enumerate(self.assessment.recommendations, 1):
            report.append(f"{i}. {rec}")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def get_summary(self) -> str:
        """
        Generate a brief summary of the assessment.

        Returns
        -------
        str
            Brief summary string
        """
        if self.assessment is None:
            self.assess()

        summary = []
        summary.append(f"Quality: {self.assessment.usability_category} "
                      f"(Score: {self.assessment.overall_score:.1f}/100)")
        summary.append(f"Blue Water Problem: {self._severity_label(self.assessment.blue_water_problem_severity)}")
        summary.append(f"Feature Usefulness: {self.assessment.feature_usefulness:.1f}/100")
        summary.append(f"Marine Science Value: {self.assessment.marine_science_value:.1f}/100")

        if self.assessment.warnings:
            summary.append(f"⚠️  {len(self.assessment.warnings)} warning(s) detected")

        return " | ".join(summary)

    def _severity_label(self, severity: float) -> str:
        """Convert severity score to label."""
        if severity >= 7:
            return "Severe"
        elif severity >= 5:
            return "Moderate"
        elif severity >= 3:
            return "Mild"
        else:
            return "Minimal"

    def is_usable_for_cv_tasks(self) -> tuple[bool, str]:
        """
        Determine if image is usable for computer vision tasks.

        Returns
        -------
        tuple[bool, str]
            Tuple of (is_usable, reason)
        """
        if self.assessment is None:
            self.assess()

        if self.assessment.feature_usefulness >= 50:
            return True, "Image has sufficient features for CV tasks"
        elif self.assessment.feature_usefulness >= 30:
            return True, "Image is marginally usable but may need enhancement"
        else:
            reasons = []
            if self.metrics['edge_density'] < 0.05:
                reasons.append("insufficient edge content")
            if self.metrics['keypoint_density'] < 0.5:
                reasons.append("too few keypoints")
            if self.metrics['visibility_score'] < 0.3:
                reasons.append("poor visibility")
            if self.metrics['blur_estimate'] > 0.7:
                reasons.append("excessive blur")

            reason = "Image unsuitable for CV tasks due to: " + ", ".join(reasons)
            return False, reason

    def is_usable_for_marine_science(self) -> tuple[bool, str]:
        """
        Determine if image is usable for marine science research.

        Returns
        -------
        tuple[bool, str]
            Tuple of (is_usable, reason)
        """
        if self.assessment is None:
            self.assess()

        if self.assessment.marine_science_value >= 50:
            return True, "Image has good scientific value"
        elif self.assessment.marine_science_value >= 30:
            return True, "Image has limited but acceptable scientific value"
        else:
            reasons = []
            if self.metrics['blue_water_severity'] > 0.7:
                reasons.append("severe blue water problem")
            if self.metrics['turbidity_score'] > 0.7:
                reasons.append("high turbidity")
            if self.metrics['visibility_score'] < 0.3:
                reasons.append("poor visibility")
            if self.metrics['rms_contrast'] < 0.2:
                reasons.append("very low contrast")

            reason = "Image has limited scientific value due to: " + ", ".join(reasons)
            return False, reason
