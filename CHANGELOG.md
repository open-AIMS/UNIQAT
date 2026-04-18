# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-18

Initial public release of UNIQAT (UNderwater Image QUality Assessment Toolkit)
accompanying the *Methods in Ecology and Evolution* submission "UNIQAT: An
Open-Source Toolkit for Reproducible Image Quality Assessment in Marine
Surveys" (Saleh and Chennu, 2026).

This release provides a pip-installable Python package for automated,
reference-free underwater image quality assessment. It is intended for use
by marine science field teams, reef imaging programmes, and computer vision
researchers working with underwater imagery.

### Added

- **Reference-free image quality assessment engine** (`src/uniqat/core/`)
  - 37 individual image quality metrics across nine categories: colour cast,
    contrast, sharpness, feature richness, visibility, turbidity,
    underwater-specific IQA, information content, and usability.
  - Configurable weighted composite scoring producing an overall quality
    score, a usability category (Excellent, Good, Fair, Poor), a blue-water
    problem severity index, a feature usefulness score, and a marine science
    value score.
  - `UnderwaterImageAssessor` class with load-once, assess-many API and
    optional multiprocessing for batch pipelines.
  - HPC-aware worker auto-detection (SLURM-aware on JCU and AIMS clusters,
    falls back to local CPU count on workstations).
- **Video quality assessment** (`src/uniqat/core/video_assessor.py`)
  - Frame-level quality tracking with temporal stability analysis,
    frame-to-frame variance, and quality trend estimation.
  - Best-frame extraction and quality timeline visualisation.
- **Deep learning metric approximators** (`src/uniqat/models/`)
  - Three architectures for fast metric regression: a custom
    `UnderwaterQualityNet` CNN, an EfficientNet-based head, and a ViT-based
    head.
  - Training pipeline with albumentations-based augmentation, mixed-precision
    training, and checkpoint management.
- **Command line interface**
  - `uniqat-assess`: single-image and batch directory assessment with CSV
    export and visual reports.
  - `uniqat-train`: training entry point for the deep learning models.
  - `uniqat-web`: launches the Gradio web interface.
  - Back-compatible wrappers retained in `scripts/` for existing SLURM job
    scripts.
- **Interactive web interface** (`web_app.py`)
  - Drag-and-drop single image assessment, video assessment, and batch
    processing tabs.
  - Bundled example images for zero-configuration first-run experience.
  - CSV download and progress reporting for batch assessment.
- **SLURM job scripts** (`scripts/assess_batch.sh`,
  `scripts/assess_batch_paper.sh`)
  - Single-survey and array-job templates for large-scale processing on
    JCU HPC and AIMS clusters.
- **Python API surface** (re-exported from `uniqat`)
  - `UnderwaterMetrics`, `UnderwaterImageAssessor`, `QualityAssessment`,
    `VideoQualityAssessor`, `VideoQualityAssessment`, `QualityVisualizer`,
    `UnderwaterQualityNet`, `EfficientNetQuality`, `ViTQuality`,
    `MultimetricPredictor`, `UnderwaterQualityTrainer`.
  - Convenience function `assess_image(path, ...)` for one-line usage.
- **CPU and GPU install paths**
  - CPU-only install via PyTorch CPU wheel index for laptops and
    workstations without CUDA.
  - GPU install guidance pointing at the official PyTorch CUDA wheels.
- **Testing, documentation, and CI**
  - `pytest` smoke test suite covering the public API, all 37 metrics, the
    assessor, the video assessor, the CLI entry points, and deep learning
    model shape checks.
  - GitHub Actions `test` workflow running on Python 3.10 and 3.11 with
    ruff lint.
  - `pdoc`-generated API reference deployed to GitHub Pages via the `docs`
    workflow.
- **Repository infrastructure**
  - MIT licence.
  - `pyproject.toml` with PEP 621 metadata and optional-dependency extras
    (`deep`, `web`, `gpu`, `dev`, `all`).
  - `CITATION.cff` for machine-readable software citation metadata.
  - Issue templates, pull request template, and contributing guide.

### Known limitations in this release

- GPU inference for the deep learning models requires a CUDA 11 or CUDA 12
  capable device; users must install the matching PyTorch CUDA wheel
  manually following the instructions on `pytorch.org`.
- The 37 traditional metrics run on CPU at approximately two frames per
  second at 1032x688 resolution on a modern Intel workstation; higher
  throughput requires the deep learning path.
- Pre-trained deep learning weights are not bundled; users train on their
  own labelled data or generate training labels via the traditional
  pipeline with `uniqat-assess --save-labels`.

[1.0.0]: https://github.com/open-AIMS/UNIQAT/releases/tag/v1.0.0
