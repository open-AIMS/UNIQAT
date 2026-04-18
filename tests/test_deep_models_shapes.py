"""Instantiate each deep learning architecture and check forward-pass shapes.

Skips the whole module if torch is not installed (the ``deep`` extra is
optional). Runs on CPU only; no GPU required.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uniqat.models.deep_models import (  # noqa: E402
    EfficientNetQualityNet,
    MultiMetricPredictor,
    UnderwaterQualityNet,
    VisionTransformerQualityNet,
)


def _dummy_batch(batch_size: int = 1, image_size: int = 224) -> torch.Tensor:
    """Return a small CPU tensor shaped like an RGB image batch."""
    return torch.zeros((batch_size, 3, image_size, image_size))


def test_underwater_quality_net_forward_shape():
    model = UnderwaterQualityNet(pretrained=False)
    model.eval()
    # UnderwaterQualityNet takes paired RGB and LAB tensors.
    x_rgb = _dummy_batch(batch_size=1, image_size=224)
    x_lab = _dummy_batch(batch_size=1, image_size=224)
    with torch.no_grad():
        out = model(x_rgb, x_lab)
    assert isinstance(out, (torch.Tensor, tuple, list, dict)), type(out)


def test_efficientnet_quality_net_forward_shape():
    model = EfficientNetQualityNet(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(_dummy_batch(batch_size=1, image_size=224))
    assert isinstance(out, (torch.Tensor, tuple, list, dict)), type(out)


def test_vit_quality_net_forward_shape():
    model = VisionTransformerQualityNet(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(_dummy_batch(batch_size=1, image_size=224))
    assert isinstance(out, (torch.Tensor, tuple, list, dict)), type(out)


def test_multi_metric_predictor_forward_shape():
    model = MultiMetricPredictor(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(_dummy_batch(batch_size=1, image_size=224))
    assert isinstance(out, (torch.Tensor, tuple, list, dict)), type(out)
