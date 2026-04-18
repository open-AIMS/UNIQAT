#!/usr/bin/env python3
"""
Web interface for UNIQAT - UNderwater Image QUality Assessment Toolkit.

Provides an interactive Gradio-based web UI for assessing underwater images
and videos with real-time visualization.
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import sys
from typing import Optional, Tuple, Dict, List
import atexit
import shutil
import warnings

# Allow running from a fresh clone without `pip install -e .` by adding the
# sibling src/ directory to sys.path. No-op when the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uniqat.core.assessor import UnderwaterImageAssessor
from uniqat.core.video_assessor import VideoQualityAssessor
from uniqat.utils.visualization import QualityVisualizer


class UnderwaterIQAWebApp:
    """Web application for underwater image quality assessment."""

    def __init__(self):
        """Initialize web application with temporary directory for outputs."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='uniqat_'))

        def cleanup_temp():
            try:
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
            except Exception as e:
                warnings.warn(f"Failed to clean up temp directory {self.temp_dir}: {e}")

        atexit.register(cleanup_temp)

    def _resize_image_if_needed(self, image: np.ndarray, max_size: int = 256) -> np.ndarray:
        """Resize image to max_size maintaining aspect ratio if larger."""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def assess_image(
        self,
        image: Optional[np.ndarray],
        generate_viz: bool = True
    ) -> Tuple[str, Optional[Image.Image], Optional[Image.Image], str]:
        """
        Assess uploaded image.

        Args:
            image: Input image array (can be None if no image uploaded)
            generate_viz: Generate visualizations

        Returns:
            Tuple of (report_text, comprehensive_viz, comparison_viz, json_data)
        """
        try:
            if image is None:
                return "No image uploaded. Please upload an image to assess.", None, None, "{}"

            if not isinstance(image, np.ndarray):
                return "Invalid image format. Expected numpy array.", None, None, "{}"

            # Validate image dimensions
            if image.ndim not in [2, 3]:
                return f"Invalid image dimensions: {image.ndim}D. Expected 2D or 3D array.", None, None, "{}"

            if image.ndim == 2:
                # Grayscale - convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                # RGBA - convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] != 3:
                return f"Invalid number of channels: {image.shape[2]}. Expected 3 (RGB).", None, None, "{}"

            #Safe color conversion with validation
            try:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except cv2.error as e:
                return f"Color conversion failed: {e}", None, None, "{}"

            # Resize large images for faster web assessment
            image_bgr = self._resize_image_if_needed(image_bgr)

            # Assess image
            assessor = UnderwaterImageAssessor(image_array=image_bgr)
            assessment = assessor.assess()

            # Generate text report
            report = assessor.get_detailed_report()

            # Generate visualizations if requested
            comprehensive_img = None
            comparison_img = None

            if generate_viz:
                try:
                    visualizer = QualityVisualizer(
                        assessor.metrics_calculator.image,
                        assessor.metrics,
                        assessment.to_dict()
                    )

                    comp_path = self.temp_dir / f'comprehensive_{id(image)}.png'
                    visualizer.create_comprehensive_report(str(comp_path))
                    comprehensive_img = Image.open(comp_path)

                    comp_compare_path = self.temp_dir / f'comparison_{id(image)}.png'
                    visualizer.create_side_by_side_comparison(str(comp_compare_path))
                    comparison_img = Image.open(comp_compare_path)
                except Exception as viz_error:
                    warnings.warn(f"Visualization generation failed: {viz_error}")
                    report += f"\n\nNote: Visualization generation failed: {viz_error}"

            # JSON data
            json_data = assessment.to_json()

            return report, comprehensive_img, comparison_img, json_data

        except Exception as e:
            error_msg = f"Error during assessment: {str(e)}\n\nException type: {type(e).__name__}"
            import traceback
            error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg, None, None, "{}"

    def assess_video(
        self,
        video_path: Optional[str],
        frame_skip: int = 30,
        max_frames: int = 100
    ) -> Tuple[str, Optional[Image.Image], str]:
        """
        Assess uploaded video.

        Args:
            video_path: Path to video file (can be None if no video uploaded)
            frame_skip: Analyze every Nth frame
            max_frames: Maximum frames to analyze

        Returns:
            Tuple of (report_text, timeline_viz, json_data)
        """
        try:
            #Validate video path
            if video_path is None or video_path == "":
                return "No video uploaded. Please upload a video to assess.", None, "{}"

            if not Path(video_path).exists():
                return f"Video file not found: {video_path}", None, "{}"

            #Validate parameters
            if frame_skip <= 0:
                return f"Invalid frame_skip: {frame_skip}. Must be positive.", None, "{}"

            if max_frames <= 0:
                return f"Invalid max_frames: {max_frames}. Must be positive.", None, "{}"

            frame_skip = max(1, int(frame_skip))
            max_frames = max(1, int(max_frames))

            # Assess video
            video_assessor = VideoQualityAssessor(
                video_path,
                frame_skip=frame_skip,
                max_frames=max_frames
            )

            assessment = video_assessor.assess()

            # Generate report
            report = video_assessor.get_summary_report()

            # Add detailed statistics
            if assessment.temporal_metrics:
                report += "\n\nTEMPORAL METRICS:\n"
                report += "-" * 70 + "\n"
                for key, value in assessment.temporal_metrics.items():
                    if isinstance(value, float):
                        report += f"{key}: {value:.4f}\n"
                    else:
                        report += f"{key}: {value}\n"

            if assessment.recommendations:
                report += "\n\nRECOMMENDATIONS:\n"
                report += "-" * 70 + "\n"
                for i, rec in enumerate(assessment.recommendations, 1):
                    report += f"{i}. {rec}\n\n"

            # Generate timeline visualization
            timeline_img = None
            try:
                timeline_path = self.temp_dir / f'video_timeline_{id(video_path)}.png'
                video_assessor.create_quality_timeline(str(timeline_path))
                timeline_img = Image.open(timeline_path)
            except Exception as viz_error:
                warnings.warn(f"Timeline visualization failed: {viz_error}")
                report += f"\n\nNote: Timeline visualization failed: {viz_error}"

            # JSON data
            json_data = assessment.to_json()

            return report, timeline_img, json_data

        except Exception as e:
            error_msg = f"Error during video assessment: {str(e)}\n\nException type: {type(e).__name__}"
            import traceback
            error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg, None, "{}"

    def batch_assess(
        self,
        files: Optional[List],
        progress=gr.Progress()
    ) -> Tuple[str, Optional[str]]:
        """
        Assess multiple images in batch with progress tracking.

        Args:
            files: List of uploaded file objects
            progress: Gradio progress tracker

        Returns:
            Tuple of (summary_report, csv_file_path)
        """
        if not files or len(files) == 0:
            return "No files uploaded. Please upload at least one image.", None

        results = []
        csv_lines = ["filename,overall_score,blue_water_severity,feature_usefulness,marine_science_value,visibility,category\n"]

        progress(0, desc="Starting batch assessment...")

        for idx, file_obj in enumerate(files):
            progress((idx + 1) / len(files), desc=f"Assessing image {idx + 1}/{len(files)}")
            try:
                if not hasattr(file_obj, 'name'):
                    results.append({
                        'filename': 'unknown',
                        'score': 0,
                        'category': 'Error: Invalid file object'
                    })
                    continue

                file_path = Path(file_obj.name)

                if not file_path.exists():
                    results.append({
                        'filename': file_path.name,
                        'score': 0,
                        'category': 'Error: File not found'
                    })
                    continue

                # Load image
                img = Image.open(str(file_path))
                img_array = np.array(img)

                # Handle different image formats
                if img_array.ndim == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif img_array.ndim == 3 and img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

                img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_array_bgr = self._resize_image_if_needed(img_array_bgr)

                # Assess
                assessor = UnderwaterImageAssessor(image_array=img_array_bgr)
                assessment = assessor.assess()

                filename = file_path.name
                results.append({
                    'filename': filename,
                    'score': assessment.overall_score,
                    'category': assessment.usability_category
                })

                try:
                    visibility_score = assessment.detailed_metrics.get('visibility_score', 0.0)
                except (AttributeError, KeyError):
                    visibility_score = 0.0

                csv_lines.append(
                    f"{filename},{assessment.overall_score:.2f},"
                    f"{assessment.blue_water_problem_severity:.2f},"
                    f"{assessment.feature_usefulness:.2f},"
                    f"{assessment.marine_science_value:.2f},"
                    f"{visibility_score:.3f},"
                    f"{assessment.usability_category}\n"
                )

            except Exception as e:
                filename = file_obj.name if hasattr(file_obj, 'name') else 'unknown'
                results.append({
                    'filename': Path(filename).name,
                    'score': 0,
                    'category': f'Error: {str(e)[:50]}'
                })

        # Generate summary report
        summary = "BATCH ASSESSMENT SUMMARY\n"
        summary += "=" * 70 + "\n"
        summary += f"Total files processed: {len(results)}\n"

        successful = sum(1 for r in results if r['score'] > 0)
        failed = len(results) - successful
        summary += f"Successful assessments: {successful}\n"
        summary += f"Failed assessments: {failed}\n\n"

        results.sort(key=lambda x: x['score'], reverse=True)

        summary += "RESULTS (sorted by quality):\n"
        summary += "-" * 70 + "\n"
        for i, result in enumerate(results, 1):
            summary += f"{i}. {result['filename']}: {result['score']:.1f}/100 ({result['category']})\n"

        scores = [r['score'] for r in results if r['score'] > 0]
        if scores:
            summary += "\n" + "=" * 70 + "\n"
            summary += "STATISTICS:\n"
            summary += f"Average score: {np.mean(scores):.2f}\n"
            summary += f"Median score: {np.median(scores):.2f}\n"
            summary += f"Min score: {np.min(scores):.2f}\n"
            summary += f"Max score: {np.max(scores):.2f}\n"
            summary += f"Std deviation: {np.std(scores):.2f}\n"
        else:
            summary += "\n" + "=" * 70 + "\n"
            summary += "No valid assessments to compute statistics.\n"

        # Write CSV to downloadable file
        csv_path = self.temp_dir / "batch_results.csv"
        with open(csv_path, 'w') as f:
            f.writelines(csv_lines)

        return summary, str(csv_path)

    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="UNIQAT - Underwater Image Quality Assessment",
            theme=gr.themes.Soft(),
            css=".gradio-container {max-width: 1400px !important}"
        ) as interface:
            gr.Markdown(
                """
                # UNIQAT - Underwater Image Quality Assessment Toolkit

                **Comprehensive quality assessment for underwater imagery with blue water problem detection**

                This system analyzes underwater images and videos to detect quality issues, measure the severity
                of the blue water problem, and provide actionable recommendations for improvement.

                ---
                """
            )

            with gr.Tabs():
                # Tab 1: Single Image Assessment
                with gr.Tab("Single Image Assessment"):
                    gr.Markdown("### Upload an underwater image for comprehensive quality analysis")

                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                label="Upload Underwater Image",
                                type="numpy"
                            )
                            viz_checkbox = gr.Checkbox(
                                label="Generate detailed visualizations",
                                value=True
                            )
                            assess_btn = gr.Button(
                                "Assess Image Quality",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=1):
                            report_output = gr.Textbox(
                                label="Assessment Report",
                                lines=20,
                                max_lines=30
                            )

                    with gr.Row():
                        comprehensive_viz = gr.Image(
                            label="Comprehensive Analysis Report",
                            type="pil"
                        )

                    with gr.Row():
                        comparison_viz = gr.Image(
                            label="Feature Detection Comparison",
                            type="pil"
                        )

                    with gr.Accordion("Raw JSON Data", open=False):
                        json_output = gr.Textbox(
                            label="JSON Output",
                            lines=10
                        )

                    assess_btn.click(
                        fn=self.assess_image,
                        inputs=[image_input, viz_checkbox],
                        outputs=[report_output, comprehensive_viz, comparison_viz, json_output]
                    )

                    # Auto-discover example images if present
                    try:
                        example_dir = Path(__file__).parent / "examples"
                        if example_dir.exists():
                            example_files = sorted([
                                str(f) for f in example_dir.glob("*")
                                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                            ])
                            if example_files:
                                gr.Examples(
                                    examples=example_files,
                                    inputs=image_input,
                                    label="Example Underwater Images"
                                )
                    except Exception:
                        pass

                # Tab 2: Video Assessment
                with gr.Tab("Video Assessment"):
                    gr.Markdown("### Upload an underwater video for temporal quality analysis")

                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.Video(label="Upload Underwater Video")

                            with gr.Row():
                                frame_skip_slider = gr.Slider(
                                    minimum=1,
                                    maximum=60,
                                    value=30,
                                    step=1,
                                    label="Frame Skip (analyze every Nth frame)"
                                )
                                max_frames_slider = gr.Slider(
                                    minimum=10,
                                    maximum=500,
                                    value=100,
                                    step=10,
                                    label="Maximum Frames to Analyze"
                                )

                            video_assess_btn = gr.Button(
                                "Assess Video Quality",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=1):
                            video_report_output = gr.Textbox(
                                label="Video Assessment Report",
                                lines=20,
                                max_lines=30
                            )

                    timeline_viz = gr.Image(
                        label="Quality Timeline Visualization",
                        type="pil"
                    )

                    with gr.Accordion("Raw JSON Data", open=False):
                        video_json_output = gr.Textbox(
                            label="JSON Output",
                            lines=10
                        )

                    video_assess_btn.click(
                        fn=self.assess_video,
                        inputs=[video_input, frame_skip_slider, max_frames_slider],
                        outputs=[video_report_output, timeline_viz, video_json_output]
                    )

                # Tab 3: Batch Processing
                with gr.Tab("Batch Processing"):
                    gr.Markdown("### Upload multiple images for batch quality assessment")

                    batch_files = gr.File(
                        label="Upload Multiple Images",
                        file_count="multiple",
                        file_types=["image"]
                    )

                    batch_assess_btn = gr.Button(
                        "Assess Batch",
                        variant="primary",
                        size="lg"
                    )

                    batch_report = gr.Textbox(
                        label="Batch Assessment Summary",
                        lines=20
                    )

                    batch_csv = gr.File(
                        label="Download CSV Results",
                        interactive=False
                    )

                    batch_assess_btn.click(
                        fn=self.batch_assess,
                        inputs=[batch_files],
                        outputs=[batch_report, batch_csv]
                    )

                # Tab 4: About
                with gr.Tab("About"):
                    gr.Markdown(
                        """
                        ## About This System

                        This underwater image quality assessment system provides:

                        ### Key Features
                        - **37 Quality Metrics**: Comprehensive analysis including UCIQE, UIQM, contrast, sharpness, and more
                        - **Blue Water Problem Detection**: Specialized metrics for detecting and quantifying the blue water problem
                        - **Multi-Score System**:
                          - Overall Quality Score (0-100)
                          - Feature Usefulness for CV Tasks (0-100)
                          - Marine Science Value (0-100)
                          - Blue Water Severity (0-10)
                        - **Beautiful Visualizations**: 11-panel comprehensive reports with detailed charts
                        - **Video Support**: Temporal quality analysis for underwater footage
                        - **Batch Processing**: Analyze multiple images simultaneously
                        - **Smart Recommendations**: Actionable suggestions for image enhancement

                        ### Metrics Explained

                        **Overall Quality Score (0-100)**
                        - 80-100: Excellent - Suitable for all applications
                        - 60-79: Good - Usable with minor enhancements
                        - 40-59: Fair - Acceptable quality, enhancement recommended
                        - 20-39: Poor - Limited usefulness
                        - 0-19: Unusable - Not recommended

                        **Blue Water Severity (0-10)**
                        - 0-3: Minimal - Slight color cast
                        - 3-5: Mild - Noticeable blue-green tint
                        - 5-7: Moderate - Significant color cast
                        - 7-10: Severe - Extreme blue water problem

                        **Feature Usefulness (0-100)**
                        - Indicates suitability for computer vision tasks
                        - Higher scores mean more reliable feature extraction

                        **Marine Science Value (0-100)**
                        - Indicates value for scientific research
                        - Considers visibility, color fidelity, and clarity

                        ### The Blue Water Problem

                        The "blue water problem" refers to the blue-green color cast and low contrast
                        in underwater images caused by:
                        - Selective light absorption (red light absorbed first)
                        - Light scattering from suspended particles
                        - Increased turbidity with depth

                        This problem severely impacts:
                        - Marine biology research
                        - Species identification
                        - Underwater surveys
                        - Computer vision applications

                        ### Technology Stack
                        - **Computer Vision**: OpenCV, NumPy, SciPy
                        - **Visualization**: Matplotlib, Seaborn
                        - **Web Interface**: Gradio
                        - **Metrics**: UCIQE, UIQM, and 37 quality metrics across nine categories

                        ### Citation

                        If you use UNIQAT in your research, please cite:
                        ```
                        @article{saleh2025uniqat,
                          title={UNIQAT: An Open-Source Toolkit for Reproducible
                                 Image Quality Assessment in Marine Surveys},
                          author={Saleh, Alzayat and Chennu, Arjun},
                          journal={Methods in Ecology and Evolution},
                          year={2025}
                        }
                        ```

                        ---

                        **Version**: 1.0.0 | **License**: MIT | [GitHub](https://github.com/open-AIMS/UNIQAT)
                        """
                    )

            gr.Markdown(
                """
                ---
                UNIQAT - Australian Institute of Marine Science / James Cook University
                """
            )

        return interface


def main():
    """
    Launch web application.

    Starts Gradio server on port 7860.
    """
    print("=" * 70)
    print("UNIQAT - Underwater Image Quality Assessment Toolkit")
    print("=" * 70)
    print("\nInitializing application...")

    try:
        app = UnderwaterIQAWebApp()
        interface = app.create_interface()

        print("Starting Gradio interface...")
        print("Access the application at: http://localhost:7860")
        print("\nPress Ctrl+C to stop the server.\n")

        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"\nError starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
