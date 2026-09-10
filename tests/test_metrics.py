"""Smoke tests for the 37 UNIQAT metrics.

Runs :class:`uniqat.UnderwaterMetrics.calculate_all_metrics` on each bundled
example image and asserts that every metric is a finite float. Does not
assert specific numerical ranges because the metrics are normalised in
different scales and the paper reports band-level characterisations rather
than point values.
"""

from __future__ import annotations

import math

import pytest

from uniqat import UnderwaterImageAssessor, UnderwaterMetrics


EXPECTED_METRIC_COUNT = 37


@pytest.fixture(scope="module")
def good_metrics(good_image_path):
    metrics = UnderwaterMetrics(image_path=str(good_image_path))
    return metrics.calculate_all_metrics()


@pytest.fixture(scope="module")
def poor_metrics(poor_image_path):
    metrics = UnderwaterMetrics(image_path=str(poor_image_path))
    return metrics.calculate_all_metrics()


def test_metric_count_good(good_metrics):
    assert len(good_metrics) == EXPECTED_METRIC_COUNT, (
        f"Expected {EXPECTED_METRIC_COUNT} metrics, got {len(good_metrics)}"
    )


def test_metric_count_poor(poor_metrics):
    assert len(poor_metrics) == EXPECTED_METRIC_COUNT


def test_all_metrics_finite_good(good_metrics):
    for name, value in good_metrics.items():
        assert isinstance(value, (int, float)), f"{name}: non-numeric ({type(value)})"
        assert math.isfinite(value), f"{name}: non-finite value {value}"


def test_all_metrics_finite_poor(poor_metrics):
    for name, value in poor_metrics.items():
        assert isinstance(value, (int, float)), f"{name}: non-numeric ({type(value)})"
        assert math.isfinite(value), f"{name}: non-finite value {value}"


def test_good_scores_exceed_poor(good_image_path, poor_image_path):
    """Composite overall_score ranks the Excellent image above the Poor image."""
    good = UnderwaterImageAssessor(image_path=str(good_image_path)).assess()
    poor = UnderwaterImageAssessor(image_path=str(poor_image_path)).assess()
    assert good.overall_score > poor.overall_score, (
        f"Expected good.overall_score > poor.overall_score, got "
        f"{good.overall_score} vs {poor.overall_score}"
    )


def test_michelson_contrast_is_bounded(good_metrics, poor_metrics):
    """Michelson contrast is (max - min) / (max + min), so it lies in [0, 1].

    image_gray is uint8: computing max + min in uint8 wraps whenever the sum
    exceeds 255, which sent the denominator to 0 and the metric to ~1e8.
    """
    for label, metrics in (("good", good_metrics), ("poor", poor_metrics)):
        value = metrics["michelson_contrast"]
        assert 0.0 <= value <= 1.0, f"{label}: michelson_contrast out of range: {value}"


def test_michelson_contrast_no_uint8_overflow():
    """A synthetic image whose max + min is exactly 256 previously overflowed."""
    import numpy as np
    from uniqat.core.metrics import UnderwaterMetrics

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = 255   # max 255
    img[:, 32:] = 1     # min 1  -> 255 + 1 == 256 wraps to 0 in uint8
    value = UnderwaterMetrics(image_array=img).analyze_contrast()["michelson_contrast"]
    assert 0.0 <= value <= 1.0, f"michelson_contrast out of range: {value}"


def test_uciqe_is_zero_for_neutral_grey():
    """OpenCV stores 8-bit LAB with a and b offset by +128. Without removing the
    offset a neutral grey scores chroma 0.71 instead of 0, inflating UCIQE."""
    import numpy as np
    from uniqat.core.metrics import UnderwaterMetrics

    grey = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert UnderwaterMetrics(image_array=grey).calculate_uciqe() < 0.05


def test_custom_metric_registry():
    """A registered metric is appended without disturbing the 37 built-ins."""
    import numpy as np
    from uniqat.core.metrics import (
        METRIC_NAMES_37, UnderwaterMetrics, register_metric, unregister_metric,
    )

    @register_metric("test_only_metric")
    def _fn(m):
        return 1.25

    try:
        out = UnderwaterMetrics(image_array=np.full((64, 64, 3), 100, np.uint8)).calculate_all_metrics()
        assert list(out)[:37] == list(METRIC_NAMES_37)   # built-ins unchanged and in order
        assert out["test_only_metric"] == 1.25
    finally:
        unregister_metric("test_only_metric")


def test_registry_refuses_to_shadow_a_builtin():
    import pytest
    from uniqat.core.metrics import register_metric

    with pytest.raises(ValueError):
        register_metric("rms_contrast")(lambda m: 0.0)
