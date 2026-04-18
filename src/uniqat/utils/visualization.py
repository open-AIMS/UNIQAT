"""
Visualization utilities for underwater image quality assessment.

This module provides functions to create visual reports, charts, and
annotated images showing quality metrics and analysis results.
"""


import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


class QualityVisualizer:
    """
    Creates beautiful visualizations for underwater image quality assessment.
    """

    def __init__(self, image: np.ndarray, metrics: dict, assessment_summary: dict):
        """
        Initialize visualizer.

        Args:
            image: Input image (BGR format)
            metrics: Dictionary of computed metrics
            assessment_summary: Summary of assessment results
        """
        self.image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image for visualization if too large to avoid matplotlib pixel limits
        # Limit to 1200px to ensure figure stays well within 2^16 pixel limit
        max_display_dimension = 1200
        h, w = self.image_rgb.shape[:2]
        if h > max_display_dimension or w > max_display_dimension:
            scale = max_display_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            self.image_rgb = cv2.resize(self.image_rgb, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)
            self.image = cv2.resize(self.image, (new_w, new_h),
                                   interpolation=cv2.INTER_AREA)

        self.metrics = metrics
        self.summary = assessment_summary

        # Set style
        sns.set_style("whitegrid")
        self.colors = {
            'excellent': '#2ecc71',
            'good': '#27ae60',
            'fair': '#f39c12',
            'poor': '#e67e22',
            'unusable': '#e74c3c',
            'primary': '#3498db',
            'secondary': '#9b59b6'
        }

    def create_comprehensive_report(self, output_path: str):
        """
        Create a comprehensive visual report with multiple panels.

        Args:
            output_path: Path to save the report image
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Original image with info overlay
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_image_with_info(ax1)

        # 2. Overall scores
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_overall_scores(ax2)

        # 3. Color channel analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_color_channels(ax3)

        # 4. Histogram analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_histograms(ax4)

        # 5. Blue water problem indicators
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_blue_water_indicators(ax5)

        # 6. Contrast and sharpness
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_contrast_sharpness(ax6)

        # 7. Feature richness
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_feature_richness(ax7)

        # 8. Underwater metrics comparison
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_underwater_metrics(ax8)

        # 9. Edge detection visualization
        ax9 = fig.add_subplot(gs[3, 0])
        self._plot_edge_detection(ax9)

        # 10. Color space analysis
        ax10 = fig.add_subplot(gs[3, 1])
        self._plot_color_space_distribution(ax10)

        # 11. Quality indicators
        ax11 = fig.add_subplot(gs[3, 2])
        self._plot_quality_indicators(ax11)

        plt.suptitle('Underwater Image Quality Assessment Report',
                    fontsize=20, fontweight='bold', y=0.995)

        # Use lower DPI and avoid bbox_inches='tight' to prevent matplotlib pixel limit errors
        # bbox_inches='tight' can cause unexpected expansion with large images
        plt.savefig(output_path, dpi=100, facecolor='white')
        plt.close()

    def _plot_image_with_info(self, ax):
        """Plot original image with overlay information."""
        ax.imshow(self.image_rgb)
        ax.axis('off')
        ax.set_title('Original Image', fontsize=14, fontweight='bold')

        # Add text overlay
        category = self.summary['usability_category']
        score = self.summary['overall_score']

        color = self._get_category_color(category)

        # Create text box
        textstr = f'{category}\n{score:.1f}/100'
        props = dict(boxstyle='round', facecolor=color, alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=16, fontweight='bold', verticalalignment='top',
               bbox=props, color='white')

        # Add dimension info
        h, w = self.image.shape[:2]
        dim_text = f'{w}x{h}px'
        ax.text(0.98, 0.02, dim_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def _plot_overall_scores(self, ax):
        """Plot overall score gauges."""
        scores = {
            'Overall\nQuality': self.summary['overall_score'],
            'Feature\nUsefulness': self.summary['feature_usefulness'],
            'Marine\nScience': self.summary['marine_science_value']
        }

        y_pos = np.arange(len(scores))
        values = list(scores.values())
        labels = list(scores.keys())

        # Create horizontal bars
        bars = ax.barh(y_pos, values, color=[self._score_to_color(v) for v in values])

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 2, i, f'{val:.1f}', va='center', fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 105)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title('Quality Scores', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    def _plot_color_channels(self, ax):
        """Plot mean values of color channels."""
        channels = ['Blue', 'Green', 'Red']
        means = [
            self.metrics['blue_channel_mean'],
            self.metrics['green_channel_mean'],
            self.metrics['red_channel_mean']
        ]

        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax.bar(channels, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Mean Value', fontweight='bold')
        ax.set_title('Color Channel Analysis', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 270)
        ax.grid(axis='y', alpha=0.3)

        # Add blue water warning if applicable
        if self.metrics['blue_water_severity'] > 0.5:
            ax.text(0.5, 0.95, '⚠️ Blue Dominance',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    def _plot_histograms(self, ax):
        """Plot color histograms."""
        colors = ('b', 'g', 'r')
        labels = ('Blue', 'Green', 'Red')

        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            hist = hist / hist.max()  # Normalize
            ax.plot(hist, color=color, alpha=0.7, label=label, linewidth=2)

        ax.set_xlim([0, 256])
        ax.set_xlabel('Pixel Value', fontweight='bold')
        ax.set_ylabel('Normalized Frequency', fontweight='bold')
        ax.set_title('Color Histograms', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    def _plot_blue_water_indicators(self, ax):
        """Plot blue water problem indicators."""
        indicators = {
            'Blue-Green\nCast': self.metrics['blue_water_severity'] * 100,
            'Turbidity': self.metrics['turbidity_score'] * 100,
            'Color\nTemp.': np.clip(self.metrics['color_temperature'] / 3.0, 0, 1) * 100,
            'Blue\nDominance': (self.metrics['blue_dominance'] + 0.5) * 100
        }

        labels = list(indicators.keys())
        values = list(indicators.values())

        bars = ax.bar(range(len(indicators)), values,
                     color=[self._severity_to_color(v/100) for v in values],
                     alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(indicators)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Severity (%)', fontweight='bold')
        ax.set_title('Blue Water Problem Indicators', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Severe')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    def _plot_contrast_sharpness(self, ax):
        """Plot contrast and sharpness metrics."""
        metrics_data = {
            'RMS\nContrast': self.metrics['rms_contrast'] * 100,
            'Michelson\nContrast': self.metrics['michelson_contrast'] * 100,
            'Local\nContrast': np.clip(self.metrics['local_contrast'] / 50, 0, 1) * 100,
            'Sharpness\n(norm)': np.clip(self.metrics['sharpness_gradient'] / 30, 0, 1) * 100,
        }

        labels = list(metrics_data.keys())
        values = list(metrics_data.values())

        bars = ax.bar(range(len(metrics_data)), values,
                     color=self.colors['primary'], alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(metrics_data)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Score (normalized)', fontweight='bold')
        ax.set_title('Contrast & Sharpness', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)

    def _plot_feature_richness(self, ax):
        """Plot feature richness metrics."""
        metrics_data = {
            'Edge\nDensity': self.metrics['edge_density'] * 500,  # Scale for visibility
            'Corner\nDensity': self.metrics['corner_density'] * 10000,
            'Keypoint\nDensity': self.metrics['keypoint_density'] * 10,
            'Texture': self.metrics['texture_complexity'] / 2,
        }

        labels = list(metrics_data.keys())
        values = list(metrics_data.values())

        bars = ax.bar(range(len(metrics_data)), values,
                     color=self.colors['secondary'], alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(metrics_data)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Scaled Score', fontweight='bold')
        ax.set_title('Feature Richness', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.grid(axis='y', alpha=0.3)

    def _plot_underwater_metrics(self, ax):
        """Plot specialized underwater metrics."""
        metrics_data = {
            'UCIQE': self.metrics['uciqe_score'] * 100,
            'UIQM': np.clip(self.metrics['uiqm_score'] / 3, 0, 1) * 100,
            'Visibility': self.metrics['visibility_score'] * 100,
            'Clarity\n(1-Turbidity)': (1 - self.metrics['turbidity_score']) * 100
        }

        labels = list(metrics_data.keys())
        values = list(metrics_data.values())

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        values_plot = values + [values[0]]  # Close the plot
        angles_plot = np.concatenate((angles, [angles[0]]))

        ax.clear()
        ax = plt.subplot(4, 3, 9, projection='polar')

        ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color=self.colors['primary'])
        ax.fill(angles_plot, values_plot, alpha=0.25, color=self.colors['primary'])
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title('Underwater Metrics', fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)

    def _plot_edge_detection(self, ax):
        """Plot edge detection visualization."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        ax.imshow(edges, cmap='hot')
        ax.axis('off')
        ax.set_title('Edge Detection (Canny)', fontsize=12, fontweight='bold')

        # Add edge density text
        edge_density = self.metrics['edge_density']
        text = f'Edge Density: {edge_density:.4f}'
        ax.text(0.5, 0.02, text, transform=ax.transAxes,
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_color_space_distribution(self, ax):
        """Plot color distribution in LAB space."""
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)

        # Sample pixels for visualization
        pixels = lab.reshape(-1, 3)
        sample_size = min(5000, len(pixels))
        sample_idx = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_idx]

        # Plot a vs b
        scatter = ax.scatter(sample_pixels[:, 1], sample_pixels[:, 2],
                           c=sample_pixels[:, 0], cmap='viridis',
                           alpha=0.5, s=1)

        ax.set_xlabel('a* (green-red)', fontweight='bold')
        ax.set_ylabel('b* (blue-yellow)', fontweight='bold')
        ax.set_title('LAB Color Space Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('L* (lightness)', fontweight='bold')

    def _plot_quality_indicators(self, ax):
        """Plot quality indicator gauges."""
        # Create a gauge-style visualization
        indicators = [
            ('CV Task\nSuitability', self.metrics['feature_usefulness_score']),
            ('Marine Science\nValue', self.metrics['marine_science_value']),
            ('Image\nClarity', (1 - self.metrics['turbidity_score']) * 100),
        ]

        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(indicators))

        for i, (label, value) in enumerate(indicators):
            y = len(indicators) - i - 0.5

            # Draw background bar
            rect_bg = Rectangle((0.25, y - 0.3), 0.6, 0.6,
                               facecolor='#ecf0f1', edgecolor='black', linewidth=1)
            ax.add_patch(rect_bg)

            # Draw value bar
            width = 0.6 * (value / 100)
            rect_val = Rectangle((0.25, y - 0.3), width, 0.6,
                                facecolor=self._score_to_color(value),
                                edgecolor='black', linewidth=1)
            ax.add_patch(rect_val)

            # Add label
            ax.text(0.05, y, label, va='center', fontsize=10, fontweight='bold')

            # Add value
            ax.text(0.90, y, f'{value:.0f}', va='center', ha='right',
                   fontsize=12, fontweight='bold')

        ax.set_title('Quality Indicators', fontsize=12, fontweight='bold', pad=20)

    def create_side_by_side_comparison(self, output_path: str):
        """
        Create side-by-side comparison with annotations.

        Args:
            output_path: Path to save the comparison image
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original image
        axes[0].imshow(self.image_rgb)
        axes[0].axis('off')
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')

        # Annotated image with edge overlay
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Create overlay
        overlay = self.image_rgb.copy()
        overlay[edges > 0] = [255, 255, 0]  # Yellow edges

        axes[1].imshow(overlay)
        axes[1].axis('off')
        axes[1].set_title('Feature Detection Overlay', fontsize=14, fontweight='bold')

        # Add metrics text
        metrics_text = (
            f"Quality: {self.summary['usability_category']} ({self.summary['overall_score']:.1f}/100)\n"
            f"Blue Water Severity: {self.summary['blue_water_problem_severity']:.1f}/10\n"
            f"Feature Usefulness: {self.summary['feature_usefulness']:.1f}/100\n"
            f"Marine Science Value: {self.summary['marine_science_value']:.1f}/100\n"
            f"Edge Density: {self.metrics['edge_density']:.4f}\n"
            f"Visibility: {self.metrics['visibility_score']:.3f}"
        )

        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   family='monospace')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Use lower DPI and avoid bbox_inches='tight' to prevent matplotlib pixel limit errors
        plt.savefig(output_path, dpi=100, facecolor='white')
        plt.close()

    def _get_category_color(self, category: str) -> str:
        """Get color for quality category."""
        category_lower = category.lower()
        if category_lower in self.colors:
            return self.colors[category_lower]
        return self.colors['primary']

    def _score_to_color(self, score: float) -> str:
        """Convert score to color."""
        if score >= 80:
            return self.colors['excellent']
        elif score >= 60:
            return self.colors['good']
        elif score >= 40:
            return self.colors['fair']
        elif score >= 20:
            return self.colors['poor']
        else:
            return self.colors['unusable']

    def _severity_to_color(self, severity: float) -> str:
        """Convert severity (0-1) to color (inverted scale)."""
        if severity >= 0.7:
            return self.colors['unusable']
        elif severity >= 0.5:
            return self.colors['poor']
        elif severity >= 0.3:
            return self.colors['fair']
        else:
            return self.colors['good']
