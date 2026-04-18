# UNIQAT: UNderwater Image QUality Assessment Toolkit

[![test](https://github.com/open-AIMS/UNIQAT/actions/workflows/test.yml/badge.svg)](https://github.com/open-AIMS/UNIQAT/actions/workflows/test.yml)
[![docs](https://github.com/open-AIMS/UNIQAT/actions/workflows/docs.yml/badge.svg)](https://open-AIMS.github.io/UNIQAT/)
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

```bash
git clone https://github.com/open-AIMS/UNIQAT.git
cd UNIQAT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU-accelerated deep learning, install [PyTorch with CUDA support](https://pytorch.org/get-started/locally/) before running `pip install`.

## Usage

### Single image assessment

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
python scripts/assess_batch.py --paths /path/to/images/ \
    -o results/ --csv results/scores.csv
```

Process with explicit worker count and recursive directory search:

```bash
python scripts/assess_batch.py --paths /path/to/images/ \
    -r --workers 48 -o results/ --csv results/scores.csv
```

Export training labels for the deep learning pipeline:

```bash
python scripts/assess_batch.py --paths /path/to/images/ \
    -o results/ --save-labels results/training_labels.json
```

### Web interface

```bash
python web_app.py
```

Opens a Gradio web application at `http://localhost:7860` for drag-and-drop image quality assessment with interactive visual reports.

### Python API

```python
from core.assessor import UnderwaterImageAssessor

assessor = UnderwaterImageAssessor(
    image_path="image.jpg",
    scale_factor=1.0
)
result = assessor.assess()

print(f"Overall quality: {result.overall_score:.1f}")
print(f"Feature quality: {result.feature_quality:.1f}")
print(f"Colour quality: {result.colour_quality:.1f}")
print(f"Usability: {result.usability_category}")
print(f"Blue-water score: {result.blue_water_score:.2f}")

metrics = result.detailed_metrics
json_str = result.to_json()
```

### Video analysis

```python
from core.video_assessor import VideoQualityAssessor

assessor = VideoQualityAssessor(
    video_path="survey_transect.mp4",
    frame_skip=30,
    max_frames=None
)
assessment = assessor.assess()

print(f"Overall: {assessment.overall_video_score:.1f}")
print(f"Mean frame score: {assessment.average_frame_score:.1f}")
print(f"Temporal stability: {assessment.temporal_stability:.2f}")
print(f"Blue-water severity: {assessment.blue_water_severity_avg:.2f}")
```

### HPC batch processing (SLURM)

Example SLURM job scripts are provided in `scripts/`. Edit the input/output paths and submit:

```bash
sbatch scripts/assess_batch.sh            # single survey
sbatch scripts/assess_batch_paper.sh      # array job for multiple surveys
```

### Training a deep learning quality model

```bash
python scripts/train_quality_model.py \
    --data-dir /path/to/images/ \
    --labels /path/to/training_labels.json \
    --model underwater \
    --epochs 50
```

See `DEEP_LEARNING_README.md` for full documentation on model architectures and training.

## Project Structure

```
UNIQAT/
  core/
    assessor.py               Main assessment engine and quality scoring
    metrics.py                37 individual image quality metrics
    video_assessor.py         Video quality assessment with temporal analysis
  models/
    deep_models.py            CNN/EfficientNet/ViT architectures
    trainer.py                Training loop and data pipeline
  utils/
    visualization.py          Quality report and figure generation
    gpu_accelerator.py        CUDA-accelerated metric computation
  scripts/
    assess_single.py          CLI: assess one image
    assess_batch.py           CLI: batch-assess a directory
    assess_batch_deep_learning.py   GPU-accelerated batch assessment
    assess_batch_hybrid.py    Combined traditional + deep learning
    train_quality_model.py    Train a deep learning quality model
    assess_batch.sh           SLURM job script (single survey)
    assess_batch_paper.sh     SLURM array job (multiple surveys)
  web_app.py                  Gradio web interface
  requirements.txt
  DEEP_LEARNING_README.md
```

## Authors

- **Alzayat Saleh** - College of Science and Engineering, James Cook University, Townsville, QLD, Australia
- **Arjun Chennu** - Australian Institute of Marine Science, Townsville, QLD, Australia

Correspondence: alzayat.saleh@jcu.edu.au

## Citation

If you use UNIQAT in your research, please cite:

```bibtex
@article{saleh2025uniqat,
  title={UNIQAT: An Open-Source Toolkit for Reproducible Image Quality
         Assessment in Marine Surveys},
  author={Saleh, Alzayat and Chennu, Arjun},
  journal={Methods in Ecology and Evolution},
  year={2025},
  note={In review}
}
```

## Licence

This project is licensed under the MIT Licence. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work was supported by James Cook University and the Australian Institute of Marine Science. Survey data were collected under the AIMS Technology Transformation Program.
