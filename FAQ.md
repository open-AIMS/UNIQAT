# Frequently asked questions

## What are the minimum image requirements for UNIQAT?

UNIQAT operates on any image that OpenCV can read: JPEG, PNG, TIFF, BMP,
and WebP all work. There is no strict minimum resolution, but very small
images (below 256 pixels on the short side) produce unstable values for
the feature-richness and sharpness metrics because those metrics assume
a useful amount of spatial detail. For the bundled example images and
the paper results, 1032 by 688 and 1920 by 1080 are typical.

Colour space must be 3-channel RGB or BGR (the standard OpenCV output).
Grayscale images are treated as RGB by channel replication; this works
but invalidates the colour cast and blue-water metrics because they
require meaningful inter-channel variation.

---

## Can I run UNIQAT on a workstation without a GPU?

Yes. The 37 traditional metrics, the Gradio web UI, the CLI, and all of
the examples run on CPU. Expected throughput on a modern Intel CPU is
two to five images per second in sequential mode, rising to twelve to
twenty images per second with sixteen workers (see the performance table
in the README).

A GPU is only required for training the deep learning models and for
bulk deep-learning inference at hundreds of images per second. Install
with the CPU torch wheel:

```bash
pip install -e .[deep,web]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## How do I use the SLURM scripts on the JCU HPC?

Two reference scripts live in `scripts/`:

- `scripts/assess_batch.sh` for a single survey directory.
- `scripts/assess_batch_paper.sh` as an array-job template for multiple
  surveys.

Both scripts expect a conda or virtualenv with UNIQAT installed. Edit the
`--paths`, `--output`, `--account`, and module-load lines to match your
allocation, then submit with `sbatch`.

The batch assessor auto-detects SLURM environments and raises the worker
count to 75 percent of the allocated CPU cores. On JCU's Zodiac cluster
this means a single 64-core node achieves roughly 48 concurrent workers
without further tuning.

---

## How do I interpret the nine category scores?

UNIQAT reports nine category-level scores plus four composite scores.
The composites (`overall_score`, `feature_usefulness`,
`marine_science_value`, `blue_water_problem_severity`) are the primary
reporting numbers. The nine categories are:

1. **Colour cast**: how blue or green the image is compared to a neutral
   reference. Lower is better; 0 is neutral.
2. **Contrast**: RMS, Michelson, and Weber contrast across the image.
3. **Sharpness**: Laplacian and gradient measures of edge strength.
4. **Feature richness**: keypoint density, edge density, texture
   complexity.
5. **Visibility**: dehazing-inspired visibility estimate (0 poor, 1
   perfect).
6. **Turbidity**: particle-scattering estimate (0 clear, 1 severe).
7. **Underwater-specific IQA**: UCIQE and UIQM, two reference-free
   underwater image quality measures from the computer vision
   literature.
8. **Information content**: entropy measures for the luminance and
   colour channels.
9. **Usability**: a composite of the feature richness and visibility
   categories, mapped to a Usability Category (Excellent, Good, Fair,
   Poor, Unusable).

See Section 3 of the paper and the API reference for the exact weights
used to combine the nine categories into the composites.

---

## How do I add a custom metric?

Extend `UnderwaterMetrics` in `src/uniqat/core/metrics.py` with a new
method that returns a float, and add the computed value to the dict
returned by `calculate_all_metrics`. If your metric should contribute to
an existing category score, update the category weight map in
`UnderwaterImageAssessor` (`src/uniqat/core/assessor.py`). Add a unit
test in `tests/test_metrics.py` that exercises the new metric on the
bundled good and poor example images and asserts the output is finite.

If your metric is intended for research comparison rather than
production use, consider subclassing `UnderwaterMetrics` in your own
package instead of modifying the paper implementation.

---

## What is the difference between the three deep learning architectures?

| Architecture | Parameters | Inference throughput (RTX 3090) | When to use |
| --- | --- | --- | --- |
| `UnderwaterQualityNet` | approx. 12 M | approx. 400 img/s | Underwater-specific CNN with paired RGB and LAB streams and spatial attention. Baseline recommended for most users. |
| `EfficientNetQualityNet` | approx. 20 M | approx. 500 to 800 img/s | EfficientNet-B0 backbone with a quality head. Best single-model throughput and accuracy balance. |
| `VisionTransformerQualityNet` | approx. 86 M | approx. 200 to 400 img/s | ViT-B/16 backbone. Highest capacity; best when plenty of training data is available. |

`MultiMetricPredictor` is a shared multi-output regression head used by
all three when the goal is to predict individual metric values rather
than a single composite score.

---

## How do I train on my own labelled data?

Export training labels from the traditional pipeline:

```bash
uniqat-assess batch --paths /path/to/images/ -o out/ \
    --save-labels out/training_labels.json
```

Then train a deep learning model:

```bash
uniqat-train --data-dir /path/to/images/ --labels out/training_labels.json \
    --model efficientnet --epochs 50
```

See `DEEP_LEARNING_README.md` for the full training configuration
reference, including augmentation, learning rate scheduling, and early
stopping.

---

## Does UNIQAT work on non-marine underwater imagery (freshwater, tanks)?

The 37 metrics are domain-agnostic: they measure contrast, sharpness,
feature richness, and colour statistics that apply to any underwater
image. The blue-water problem severity metric assumes the characteristic
blue or green cast of open-ocean water and will return low severity on
clear freshwater or well-lit aquarium imagery. The Usability Category
mapping may be optimistic for freshwater images with muddy backgrounds.

For non-marine use cases, focus on the category-level scores
(`sharpness_gradient`, `uciqe_score`, `uiqm_score`, and
`feature_usefulness_score`) rather than the blue-water composite.

---

## Does it work on still images only, or also video?

Both. `uniqat.UnderwaterImageAssessor` handles still images. For video,
`uniqat.VideoQualityAssessor` extracts frames at a configurable
`frame_skip` interval, runs the per-frame assessor, and reports
temporal-stability and frame-to-frame quality-variance metrics on top
of the per-frame scores. The Gradio web UI has a dedicated Video tab
and the repository includes `filter_video.py` as a reference utility
that drops frames below a quality threshold.

---

## How do I cite UNIQAT?

Until the accompanying paper is accepted, cite both the software and
the preprint. The `CITATION.cff` file at the repository root has the
machine-readable metadata; the README's Citation section has a BibTeX
entry. Briefly:

> Saleh, A. and Chennu, A. UNIQAT: An Open-Source Toolkit for
> Reproducible Image Quality Assessment in Marine Surveys. *Methods in
> Ecology and Evolution*, 2026. In review.

If the paper is accepted before you publish, please update the citation
to point to the published version.
