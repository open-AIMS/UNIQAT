"""Assess a single bundled example image with UNIQAT.

End-to-end demonstration of the :class:`uniqat.UnderwaterImageAssessor` and
:class:`uniqat.QualityVisualizer` APIs. Loads ``examples/18_img_good.png``,
runs the full 37-metric assessment, prints the key scores, saves the JSON
report alongside the image, and writes a PNG visualisation of the composite
quality report.

Run with::

    python examples/single_image_assessment.py

No command-line arguments are required. Finishes in under ten seconds on a
modern CPU. Requires the base UNIQAT install (``pip install -e .``); no
deep learning extras needed.

Output files (written to ``examples/``):

- ``18_img_good_uniqat.json`` - full assessment as JSON.
- ``18_img_good_uniqat_report.png`` - visual quality report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from uniqat import QualityVisualizer, UnderwaterImageAssessor


EXAMPLE_IMAGE = Path(__file__).resolve().parent / "18_img_good.png"
OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Run the single-image example end-to-end."""
    if not EXAMPLE_IMAGE.is_file():
        raise FileNotFoundError(
            f"Example image not found: {EXAMPLE_IMAGE}. Run this script from "
            "a UNIQAT repository checkout so the bundled examples/ folder is "
            "available."
        )

    print(f"Assessing {EXAMPLE_IMAGE.name} ...")
    assessor = UnderwaterImageAssessor(image_path=str(EXAMPLE_IMAGE))
    assessment = assessor.assess()

    print()
    print("=" * 60)
    print("UNIQAT composite scores")
    print("=" * 60)
    print(f"Overall quality score:       {assessment.overall_score:5.1f} / 100")
    print(f"Usability category:          {assessment.usability_category}")
    print(f"Feature usefulness score:    {assessment.feature_usefulness:5.1f} / 100")
    print(f"Marine science value score:  {assessment.marine_science_value:5.1f} / 100")
    print(f"Blue-water problem severity: {assessment.blue_water_problem_severity:5.2f} / 10")
    print()

    json_path = OUTPUT_DIR / "18_img_good_uniqat.json"
    with json_path.open("w") as handle:
        json.dump(assessment.to_dict(), handle, indent=2, default=str)
    print(f"Wrote JSON report:       {json_path}")

    report_path = OUTPUT_DIR / "18_img_good_uniqat_report.png"
    try:
        visualiser = QualityVisualizer(
            assessor.metrics_calculator.image,
            assessor.metrics,
            assessment.to_dict(),
        )
        visualiser.create_comprehensive_report(str(report_path))
        print(f"Wrote visual report:     {report_path}")
    except Exception as err:  # pragma: no cover - visualiser is best-effort
        print(f"Skipped visual report:   {err}")


if __name__ == "__main__":
    main()
