"""End-to-end smoke tests for :class:`uniqat.UnderwaterImageAssessor`."""

from __future__ import annotations

import math

import pytest

from uniqat import QualityAssessment, UnderwaterImageAssessor


@pytest.fixture(scope="module")
def good_assessment(good_image_path):
    return UnderwaterImageAssessor(image_path=str(good_image_path)).assess()


@pytest.fixture(scope="module")
def poor_assessment(poor_image_path):
    return UnderwaterImageAssessor(image_path=str(poor_image_path)).assess()


def _assert_fields_finite(assessment: QualityAssessment) -> None:
    """Every numeric field of the QualityAssessment is finite (no NaN or inf)."""
    payload = assessment.to_dict()
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), f"{key}: non-finite value {value}"
        elif isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, (int, float)):
                    assert math.isfinite(inner_value), (
                        f"{key}.{inner_key}: non-finite value {inner_value}"
                    )


def test_assessment_type(good_assessment):
    assert isinstance(good_assessment, QualityAssessment)


def test_good_assessment_fields_finite(good_assessment):
    _assert_fields_finite(good_assessment)


def test_poor_assessment_fields_finite(poor_assessment):
    _assert_fields_finite(poor_assessment)


def test_good_usability_is_excellent_or_good(good_assessment):
    assert good_assessment.usability_category in {"Excellent", "Good"}, (
        good_assessment.usability_category
    )


def test_overall_score_in_range(good_assessment, poor_assessment):
    for label, a in (("good", good_assessment), ("poor", poor_assessment)):
        assert 0.0 <= a.overall_score <= 100.0, (
            f"{label}.overall_score out of [0, 100]: {a.overall_score}"
        )


def test_detailed_metrics_populated(good_assessment):
    assert isinstance(good_assessment.detailed_metrics, dict)
    assert len(good_assessment.detailed_metrics) >= 30  # 37 expected, allow minor drift


def test_compat_property_aliases(good_assessment):
    """The compatibility aliases added for the paper's S5.5 Python API."""
    a = good_assessment
    assert a.feature_quality == a.feature_usefulness
    assert a.blue_water_score == a.blue_water_problem_severity
    assert 0.0 <= a.colour_quality <= 100.0
