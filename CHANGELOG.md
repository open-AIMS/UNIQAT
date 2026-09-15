# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-09-11

Corrects the 37-metric heads added in 1.0.2, which did not match the models the
deep learning results in the accompanying manuscript were measured on.

### Fixed

- **Metric head widths on three of the four architectures.** The heads added in
  1.0.2 were written independently of the implementation used for the fidelity
  runs and used wider hidden layers, so the released package and the manuscript
  described different models. UnderwaterQualityNet goes from a 512-unit hidden
  layer to 256, VisionTransformerQualityNet from 512 to 256, and
  EfficientNetQualityNet now widens its existing `multi_metric_head` to 37
  outputs instead of carrying a second parallel head. MultiMetricPredictor was
  already correct.

  Parameter counts are now 52.47M for UnderwaterQualityNet, 86.50M for the
  Vision Transformer, 13.10M for EfficientNet-B3 and 26.15M for the
  Multi-Metric Predictor, which are the figures in Supplementary Table S2 of the
  manuscript. Verified by instantiating each model and counting.

- **Dropout in the two rewritten heads** now follows the constructor's `dropout`
  argument rather than being hardcoded at 0.3, matching the surrounding heads.

### Notes for existing users

No metric value changes: this release touches only the deep learning heads, not
the traditional pipeline. A checkpoint trained against 1.0.2 will not load into
1.0.3, because three of the four state dicts changed shape. Retrain, or pin
1.0.2 if you need to load an existing checkpoint.

This is the release the accompanying manuscript cites.

## [1.0.2] - 2026-09-10

Corrects a second metric defect and makes the deep learning branch usable for
the fidelity comparison reported in the accompanying manuscript.

### Fixed

- **UCIQE was inflated by roughly 5x.** OpenCV stores 8-bit LAB with the a and b
  channels offset by +128, so neutral grey is (128, 128) rather than (0, 0). The
  offset was not removed before computing chroma, so a neutral grey image scored
  chroma 0.71 instead of 0. Values reported in the UNIQAT manuscript were
  produced with the earlier definition and are not comparable to values from
  this release. Because UCIQE contributes 10 of the 100 points of the marine
  science value subscore, and that subscore carries a coefficient of 0.35, the
  correction also shifts the overall score by up to 3.5 points.

### Added

- **A 37-metric head on all four architectures.** Previously only the
  Multi-Metric Predictor emitted the full metric vector; the other three carried
  three to five summary heads and could not be scored on the same task. Each now
  exposes `predict_metrics()` returning shape (B, 37), and `metric_names` drawn
  from a single shared constant so the four cannot drift apart. Parameter counts
  become 52.6M, 86.6M, 13.9M and 26.1M. (These head widths were wrong; see
  1.0.3.)
- **A custom metric registry.** `register_metric()`, `unregister_metric()` and
  `registered_metrics()` let users add metrics without editing the package or
  subclassing. Registered metrics are appended after the 37 built-ins, whose
  names and order are unchanged, and registering a name that collides with a
  built-in raises rather than silently replacing it.
- Regression tests for all of the above.

### Notes for existing users

A model trained on labels from an earlier release predicts the old UCIQE
variant. Retrain, or treat that output as the pre-1.0.2 definition. Cached
assessment outputs should be recomputed if UCIQE or the overall score matters
to you.

## [1.0.1] - 2026-09-10

Patch release fixing two defects found while preparing the revision of the
accompanying manuscript. Both affect the metric outputs or the deep learning
API; neither changes the toolkit's interface.

### Fixed

- **Michelson contrast returned values far outside its [0, 1] range.**
  `analyze_contrast()` computed `(max - min) / (max + min + 1e-6)` on the uint8
  grayscale array, so `max + min` wrapped whenever it exceeded 255. Where the
  sum landed on exactly 256 the denominator collapsed to 1e-6 and the metric
  returned values of order 1e8; more commonly the sum wrapped to a small number
  and the metric returned some tens. The operands are now cast to float.
  Michelson contrast is bounded by [0, 1] and previously breached that bound
  for the large majority of survey images.

- **`MultiMetricPredictor.predict_dict()` raised `IndexError`.** `metric_names`
  carried 42 entries left over from a pre-release metric set while the
  regression head is 37 wide, so zipping names against output columns indexed
  past the end of the tensor. The list is now the 37 names that
  `UnderwaterMetrics.calculate_all_metrics()` actually emits, in emission order,
  and the five stale entries (`overall_score`, `blue_water_problem_severity`,
  `quality_index`, `clarity_score`, `information_content`) are gone.

### Added

- Regression tests for both defects: the Michelson bound, the exact
  `max + min == 256` overflow case, the metric-names/output-width invariant, a
  `predict_dict()` call, and equality between the predicted names and the
  traditional pipeline's key order.

### Notes for existing users

Quality scores are unaffected: `michelson_contrast` is not an input to the
overall score, which combines feature usefulness, marine science value,
visibility and blue-water severity. Users who consumed `michelson_contrast`
directly, or who compared metric profiles across datasets, should recompute it.

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
