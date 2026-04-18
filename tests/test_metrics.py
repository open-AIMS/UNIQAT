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
