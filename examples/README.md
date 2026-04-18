# UNIQAT examples

Runnable end-to-end examples that demonstrate the UNIQAT Python API. Each
example is self-contained and uses the four bundled images in this directory,
so no extra downloads are required.

Install UNIQAT with the base extras before running any of these:

```bash
pip install -e .
```

The deep learning and web UI extras (`pip install -e .[deep,web]`) are not
needed for these examples.

## Included examples

| File | What it shows | Expected CPU runtime |
| --- | --- | --- |
| `single_image_assessment.py` | Runs `UnderwaterImageAssessor` on `18_img_good.png`, prints the composite scores, writes a JSON report and a multi-panel quality visualisation PNG. | Under 10 seconds. |
| `single_image_assessment.ipynb` | Jupyter twin of the script above with per-step explanations and a demonstration of the `uniqat.assess_image` one-liner. | Under 30 seconds end-to-end. |
| `batch_directory.py` | Walks a directory of images (defaults to the bundled set), runs the assessor on each, and writes a combined CSV of scores (`uniqat_batch_scores.csv`). Pass your own directory as the first argument to use a custom input. | Under 60 seconds for the bundled 4 images. |

## Bundled example images

| File | Intended score band | Purpose |
| --- | --- | --- |
| `18_img_good.png` | Excellent | High-quality reference image for smoke tests. |
| `18_img_poor.png` | Poor | Demonstrates the lower end of the scale. |
| `REEFSCAN_DEEP_02_cam1_20240920_005753_206_4567.jpg` | Good or better | Real survey frame from the AIMS ReefScan Deep programme. |
| `REEFSCAN_DEEP_02_cam1_20240920_011107_967_7746.jpg` | Unusable | Real survey frame showing a severe blue-water problem. |

## Running the examples

From the repository root:

```bash
python examples/single_image_assessment.py
python examples/batch_directory.py
jupyter notebook examples/single_image_assessment.ipynb
```

All three examples write their outputs back into this `examples/` directory.
The `batch_directory.py` script automatically skips files whose names contain
`_uniqat`, so re-running it after a `single_image_assessment.py` run is safe
and idempotent.
