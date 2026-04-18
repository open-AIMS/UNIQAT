"""Shared fixtures and CPU-enforcing configuration for the UNIQAT test suite."""

from __future__ import annotations

import os

# Force CPU before torch is imported anywhere in the package. CI is CPU-only
# and this matches the documented UNIQAT CPU install path.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture(scope="session")
def good_image_path() -> Path:
    """Path to the bundled Excellent-band example image."""
    p = EXAMPLES_DIR / "18_img_good.png"
    if not p.is_file():
        pytest.skip(f"bundled example image missing: {p}")
    return p


@pytest.fixture(scope="session")
def poor_image_path() -> Path:
    """Path to the bundled Poor-band example image."""
    p = EXAMPLES_DIR / "18_img_poor.png"
    if not p.is_file():
        pytest.skip(f"bundled example image missing: {p}")
    return p
