"""Smoke test for :class:`uniqat.VideoQualityAssessor` on a synthetic short video.

Generates a 30-frame 160x120 clip directly inside the test using OpenCV so
no large binaries need to be bundled with the repository. Asserts the
assessor returns a populated :class:`VideoQualityAssessment` with finite
scores.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from uniqat import VideoQualityAssessment, VideoQualityAssessor


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Create a short synthetic underwater-like video and return its path."""
    out_path = tmp_path_factory.mktemp("uniqat_video") / "synthetic.mp4"
    width, height, fps, n_frames = 160, 120, 10, 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV could not open an mp4v VideoWriter on this platform")

    rng = np.random.default_rng(42)
    for i in range(n_frames):
        # Blue-tinted frame with mild noise and a moving green circle.
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = 160  # blue
        frame[..., 1] = 90 + (i * 3 % 30)  # varying green
        frame[..., 2] = 40
        noise = rng.integers(-10, 10, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cx = 20 + i * 4
        cv2.circle(frame, (cx, height // 2), 12, (60, 220, 60), -1)
        writer.write(frame)

    writer.release()
    return out_path


def test_video_assessor_runs(synthetic_video):
    assessor = VideoQualityAssessor(
        video_path=str(synthetic_video),
        frame_skip=5,
        max_frames=6,
    )
    assessment = assessor.assess()

    assert isinstance(assessment, VideoQualityAssessment)
    assert assessment.analyzed_frames >= 1
    assert 0.0 <= assessment.overall_video_score <= 100.0
    assert math.isfinite(assessment.average_frame_score)
    assert 0.0 <= assessment.temporal_stability <= 1.0
    assert isinstance(assessment.frame_assessments, list)
