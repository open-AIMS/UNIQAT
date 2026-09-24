# UNIQAT: UNderwater Image QUality Assessment Toolkit

[![test](https://github.com/open-AIMS/UNIQAT/actions/workflows/test.yml/badge.svg)](https://github.com/open-AIMS/UNIQAT/actions/workflows/test.yml)
[![docs build](https://github.com/open-AIMS/UNIQAT/actions/workflows/docs.yml/badge.svg)](https://github.com/open-AIMS/UNIQAT/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![pip install](https://img.shields.io/badge/pip%20install-editable-orange.svg)](#installation)

An open-source Python library for automated, reference-free underwater image quality assessment. UNIQAT computes 37 metrics across nine categories of image characteristics relevant to underwater environments and produces task-specific composite quality scores for marine science and computer vision applications.

UNIQAT was developed at the [Australian Institute of Marine Science (AIMS)](https://www.aims.gov.au/) and [James Cook University (JCU)](https://www.jcu.edu.au/).

## Features

- 37 reference-free image quality metrics across nine categories (colour cast, contrast, sharpness, feature richness, visibility, turbidity, underwater-specific IQA, information content, and usability)
- Configurable weighted scoring for task-specific quality assessment
- HPC-aware multiprocessing (auto-detects cluster vs workstation)
- Optional deep learning models for fast metric approximation (CNN, EfficientNet, ViT)
- Video quality assessment with temporal stability analysis
- Interactive web interface (Gradio)
- SLURM job scripts for large-scale batch processing

## Installation

UNIQAT is a standard pip-installable package. AIMS and JCU users on
workstations or laptops without a CUDA GPU should follow the CPU path; the
full 37 traditional metrics, the Gradio web UI, and the CLI all run at
useful speed on CPU. The GPU path is only needed for training or bulk
deep learning inference.

### CPU path (laptop, workstation, or CI)

```bash
git clone https://github.com/open-AIMS/UNIQAT.git
cd UNIQAT
python -m venv .venv
source .venv/bin/activate
pip install -e .[deep,web]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The CPU `torch` wheel is installed from the PyTorch CPU index to avoid
pulling the multi-gigabyte CUDA build. The `deep` extra is still useful on
CPU because it enables the training and inference scripts (just more
slowly). Drop `[deep]` if you only need the traditional pipeline and the
web UI.

### GPU path (training, large-scale inference)

```bash
git clone https://github.com/open-AIMS/UNIQAT.git
cd UNIQAT
python -m venv .venv
source .venv/bin/activate
pip install -e .[deep,web]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Pick the matching CUDA-suffixed index URL for your driver from
<https://pytorch.org/get-started/locally/> (`cu118`, `cu121`, `cu124`, and
so on). An NVIDIA GPU with at least 8 GB VRAM is recommended for training.

### Developer install

```bash
pip install -e .[deep,web,dev]
pytest tests/ -q
ruff check src/uniqat
pdoc -o site -d numpy uniqat && python docs/postprocess.py site
```

### Performance modes

Approximate throughput on 1000 underwater images at 1920x1080, adapted
from the deep learning benchmarks in `DEEP_LEARNING_README.md`:

| Mode | Hardware | Speed (images per second) | Total time |
| --- | --- | --- | --- |
| Traditional, sequential | Intel i7 CPU | 2 to 5 | 3 to 8 minutes |
| Traditional, 16 workers | Intel i7, 8 physical cores | 12 to 20 | 50 to 80 seconds |
| Deep learning (EfficientNet) | GTX 1080 GPU | 200 to 300 | 3 to 5 seconds |
| Deep learning (EfficientNet) | RTX 3090 GPU | 500 to 800 | 1 to 2 seconds |
| Hybrid (traditional + DL) | i7 CPU + RTX 3090 GPU | 8 to 12 | 80 to 125 seconds |

Use the traditional pipeline for interpretable per-metric reporting and
small to medium datasets (up to a few thousand images). Use the deep
learning path for multi-tens-of-thousands-of-images bulk inference; use
the hybrid mode to combine both.

## Quick start

After installing UNIQAT with the CPU path above, run the bundled example
to confirm everything works:

```bash
python examples/single_image_assessment.py
```

The script loads `examples/18_img_good.png`, prints the composite quality
scores, and writes a JSON report and a multi-panel visualisation PNG back
into the `examples/` directory. It completes in roughly three seconds on
a modern CPU and requires no command-line arguments.

Two more runnable examples are provided:

- `examples/batch_directory.py` walks a directory and writes a CSV of
  scores (defaults to the four bundled images).
- `examples/single_image_assessment.ipynb` is the Jupyter twin of the
  first script, with per-step explanations.

See `examples/README.md` for the full list with expected runtimes.

## Usage

### Command-line interface

Once installed, UNIQAT exposes three console-script entry points:

```bash
uniqat-assess single --image_path image.jpg -o results/
uniqat-assess batch  --paths /path/to/images/ -o results/ --csv results/scores.csv
uniqat-train  --data-dir /path/to/images/ --labels labels.json --model underwater
uniqat-web
```

The `uniqat-*` entry points are thin wrappers around the reference scripts
in `scripts/`; the reference scripts remain the canonical implementations
for SLURM and power-user workflows.

### Single image assessment

```bash
uniqat-assess single --image_path image.jpg -o results/
```

Or, using the reference script directly:

```bash
python scripts/assess_single.py --image_path image.jpg -o results/
```

Output includes a text report, JSON metrics, and visual quality report:

```
Overall Quality Score:        72.3/100
Usability Category:           Good
Blue Water Problem Severity:  3.2/10
Feature Usefulness Score:     68.5/100
Marine Science Value:         74.1/100
```

### Batch directory processing

```bash
uniqat-assess batch --paths /path/to/images/ -o results/ --csv results/scores.csv
```

Process with explicit worker count and recursive directory search:

```bash
uniqat-assess batch --paths /path/to/images/ \
    -r --workers 48 -o results/ --csv results/scores.csv
```

Export training labels for the deep learning pipeline:

```bash
uniqat-assess batch --paths /path/to/images/ \
    -o results/ --save-labels results/training_labels.json
```

### GPU-accelerated batch modes

Three batch scripts cover different throughput and interpretability
trade-offs:

- **`scripts/assess_batch.py`** (exposed as `uniqat-assess batch`):
  the 37 traditional metrics with CPU multiprocessing. Fully
  interpretable per-metric output, suitable for small to medium
  datasets up to a few thousand images.
- **`scripts/assess_batch_deep_learning.py`**: GPU batch inference with
  a trained deep learning model. Returns only the composite scores, no
  per-metric breakdown. Best for tens of thousands of images when a
  trained model is available.
- **`scripts/assess_batch_hybrid.py`**: runs both the traditional
  metrics and the deep learning model and combines them with a
  configurable weighting. Slowest of the three; useful when the deep
  learning model is new and you want to sanity-check its outputs
  against the reference metrics.

See `DEEP_LEARNING_README.md` for command-line flags and configuration
details of the two deep-learning scripts.

### Web interface

```bash
uniqat-web
```

Opens a Gradio web application at `http://localhost:7860` for
drag-and-drop image quality assessment with interactive visual reports.
The reference `python web_app.py` invocation still works.

### Python API

```python
import uniqat

# One-line convenience wrapper for scripting and notebooks.
result = uniqat.assess_image("image.jpg")

print(f"Overall quality: {result.overall_score:.1f}")
print(f"Feature quality: {result.feature_quality:.1f}")
print(f"Colour quality:  {result.colour_quality:.1f}")
print(f"Usability:       {result.usability_category}")
print(f"Blue-water:      {result.blue_water_score:.2f}")

metrics = result.detailed_metrics  # 37 individual metric values
json_str = result.to_json()
```

For batch pipelines, instantiate the assessor directly and reuse it across
images:

```python
from uniqat import UnderwaterImageAssessor, QualityVisualizer

assessor = UnderwaterImageAssessor(image_path="image.jpg", scale_factor=1.0)
result = assessor.assess()

visualiser = QualityVisualizer(
    assessor.metrics_calculator.image,
    assessor.metrics,
    result.to_dict(),
)
visualiser.create_comprehensive_report("report.png")
```

### Video analysis

```python
from uniqat import VideoQualityAssessor

assessor = VideoQualityAssessor(
    video_path="survey_transect.mp4",
    frame_skip=30,
    max_frames=None,
)
assessment = assessor.assess()

print(f"Overall:            {assessment.overall_video_score:.1f}")
print(f"Mean frame score:   {assessment.average_frame_score:.1f}")
print(f"Temporal stability: {assessment.temporal_stability:.2f}")
print(f"Blue-water severity: {assessment.blue_water_severity_avg:.2f}")
```

### HPC batch processing (SLURM)

Example SLURM job scripts are provided in `scripts/`. Edit the input and
output paths and submit:

```bash
sbatch scripts/assess_batch.sh            # single survey
sbatch scripts/assess_batch_paper.sh      # array job for multiple surveys
```

### Training a deep learning quality model

```bash
uniqat-train --data-dir /path/to/images/ \
    --labels /path/to/training_labels.json \
    --model underwater \
    --epochs 50
```

See `DEEP_LEARNING_README.md` for full documentation on model architectures
and training.

## Project Structure

```
UNIQAT/
  src/uniqat/                   Installed Python package (pip install -e .)
    __init__.py                 Public API: UnderwaterImageAssessor, ...
    cli.py                      uniqat-assess entry point
    cli_train.py                uniqat-train entry point
    cli_web.py                  uniqat-web entry point
    core/
      assessor.py               Main assessment engine and quality scoring
      metrics.py                37 individual image quality metrics
      video_assessor.py         Video quality assessment with temporal analysis
    models/
      deep_models.py            CNN, EfficientNet, and ViT architectures
      trainer.py                Training loop and data pipeline
    utils/
      visualization.py          Quality report and figure generation
      gpu_accelerator.py        CUDA-accelerated metric computation (optional)
  scripts/                      Reference scripts (kept for SLURM / power users)
    assess_single.py            Assess one image
    assess_batch.py             Batch-assess a directory with multiprocessing
    assess_batch_deep_learning.py   GPU-accelerated batch assessment
    assess_batch_hybrid.py      Combined traditional + deep learning
    train_quality_model.py      Train a deep learning quality model
    assess_batch.sh             SLURM job script (single survey)
    assess_batch_paper.sh       SLURM array job (multiple surveys)
  examples/                     Runnable examples (see examples/README.md)
  tests/                        pytest smoke suite (see .github/workflows/test.yml)
  docs/                         API reference build (pdoc -> GitHub Pages)
  web_app.py                    Gradio web interface (launched by uniqat-web)
  filter_video.py               Utility: remove bad frames from a video
  pyproject.toml                Package metadata and extras
  DEEP_LEARNING_README.md       Deep learning model and training reference
```

## Getting help

- **API reference**: build the pdoc site locally with
  `pdoc -o site -d numpy uniqat && python docs/postprocess.py site`
  then open `site/uniqat.html`. Hosted docs are not available while
  the repository is private on a non-Enterprise organisation.
- **Frequently asked questions**: [FAQ.md](FAQ.md) covers minimum image
  requirements, running without a GPU, using the SLURM scripts on the
  JCU HPC, interpreting the nine category scores, choosing a deep
  learning architecture, and non-marine use cases.
- **Bug report, feature request, or survey-specific data question**:
  open an issue at
  <https://github.com/open-AIMS/UNIQAT/issues/new/choose>. Three
  templates are provided; the Reef data question template is the right
  place for survey- or rig-specific questions.
- **Security-sensitive issues**: email the corresponding author at
  `alzayat.saleh@my.jcu.edu.au` rather than opening a public issue.

## Authors

- **Alzayat Saleh** - College of Science and Engineering, James Cook University, Townsville, QLD, Australia
- **Arjun Chennu** - Australian Institute of Marine Science, Townsville, QLD, Australia

Correspondence: alzayat.saleh@my.jcu.edu.au

## Data

The per-image quality scores for the 130,845 ReefScan images analysed in the
accompanying paper, with GPS position, sonar depth and timestamp, and the
per-survey summary tables, are archived in the AIMS Data Repository:

> Saleh, A., & Chennu, A. (2026). ReefScan underwater imagery from nine Great
> Barrier Reef surveys, with per-image quality assessments (UNIQAT) [Dataset].
> Australian Institute of Marine Science. https://doi.org/10.25845/5KRB-J094

The source images are held by the Australian Institute of Marine Science and are
not publicly archived because of their size (about 624 GB).

## Citation

If you use UNIQAT in your research, please cite:

```bibtex
@article{saleh2026uniqat,
  title={UNIQAT: An Open-Source Toolkit for Reproducible Image Quality
         Assessment in Marine Surveys},
  author={Saleh, Alzayat and Chennu, Arjun},
  journal={Methods in Ecology and Evolution},
  year={2026},
  note={Accepted}
}
```

The software release used in the paper, v1.0.3, is archived on Zenodo:

> Saleh, A., & Chennu, A. (2026). UNIQAT: UNderwater Image QUality Assessment
> Toolkit (Version 1.0.3) [Software]. Zenodo. https://doi.org/10.5281/zenodo.22928186


## Licence

This project is licensed under the MIT Licence. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work was supported by James Cook University and the Australian Institute of Marine Science. Survey data were collected under the AIMS Technology Transformation Program.
