"""Smoke tests that the package exposes the documented public API."""

from __future__ import annotations

import re

import uniqat


def test_version_exposed():
    """``uniqat.__version__`` is a valid semver string."""
    assert isinstance(uniqat.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", uniqat.__version__), uniqat.__version__


def test_all_names_importable():
    """Every name in ``uniqat.__all__`` is actually importable."""
    for name in uniqat.__all__:
        assert hasattr(uniqat, name), (
            f"Public name {name!r} is listed in __all__ but missing from the module"
        )


def test_required_core_api_present():
    """The brief mandates these core names at minimum."""
    required = {
        "UnderwaterMetrics",
        "UnderwaterImageAssessor",
        "QualityAssessment",
        "VideoQualityAssessor",
        "VideoQualityAssessment",
        "QualityVisualizer",
        "assess_image",
    }
    missing = required - set(uniqat.__all__)
    assert not missing, f"Missing required public names: {missing}"


def test_assess_image_callable():
    """The ``assess_image`` convenience wrapper is callable with a path."""
    assert callable(uniqat.assess_image)
