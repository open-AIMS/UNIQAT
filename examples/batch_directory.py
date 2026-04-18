"""Assess every image in a directory and write a combined CSV of scores.

Batch-equivalent of :mod:`examples.single_image_assessment` using only the
public :class:`uniqat.UnderwaterImageAssessor` API. Demonstrates how to
iterate over a directory, accumulate scores, and emit a tidy CSV that can
be loaded directly into pandas, R, or Excel.

Run with no arguments to process the four bundled example images::

    python examples/batch_directory.py

Or point it at your own survey directory::

    python examples/batch_directory.py /path/to/my/survey/images

Finishes in under a minute for the bundled set on a modern CPU. Requires
the base UNIQAT install (``pip install -e .``); no deep learning extras
needed. Output CSV is written to ``examples/uniqat_batch_scores.csv``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from uniqat import UnderwaterImageAssessor


EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLES_DIR
OUTPUT_CSV = EXAMPLES_DIR / "uniqat_batch_scores.csv"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_images(directory: Path) -> list[Path]:
    """Return the sorted list of supported image files in ``directory``.

    Files whose names contain ``_uniqat`` are assumed to be outputs of a
    previous run (visual reports, comparison images) and are skipped.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to search (non-recursive).

    Returns
    -------
    list[pathlib.Path]
        Sorted list of image paths with extensions in
        ``{.png, .jpg, .jpeg, .tif, .tiff, .bmp}``.
    """
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and "_uniqat" not in p.stem
    )


def main() -> None:
    """Run the batch example end-to-end."""
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    images = find_images(input_dir)
    if not images:
        raise SystemExit(
            f"No images with supported extensions found in {input_dir}."
        )

    print(f"Assessing {len(images)} image(s) in {input_dir} ...")
    rows: list[dict] = []
    for idx, image_path in enumerate(images, start=1):
        assessor = UnderwaterImageAssessor(image_path=str(image_path))
        a = assessor.assess()
        row = {
            "filename": image_path.name,
            "overall_score": round(a.overall_score, 2),
            "usability_category": a.usability_category,
            "feature_usefulness": round(a.feature_usefulness, 2),
            "marine_science_value": round(a.marine_science_value, 2),
            "blue_water_problem_severity": round(a.blue_water_problem_severity, 2),
        }
        rows.append(row)
        print(
            f"  [{idx}/{len(images)}] {image_path.name}: "
            f"overall={row['overall_score']}, usability={row['usability_category']}"
        )

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
