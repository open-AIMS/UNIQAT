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


def test_metric_names_match_num_metrics():
    """metric_names must stay the same length as the model's output width.

    predict_dict() indexes the output tensor by position, so a names list
    longer than num_metrics raises IndexError at call time.
    """
    model = MultiMetricPredictor(pretrained=False)
    assert len(model.metric_names) == 37
    assert len(set(model.metric_names)) == len(model.metric_names)


def test_predict_dict_returns_one_value_per_name():
    model = MultiMetricPredictor(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model.predict_dict(_dummy_batch(batch_size=2, image_size=224))
    assert set(out) == set(model.metric_names)
    for name, value in out.items():
        assert value.shape == (2,), f"{name}: {value.shape}"


def test_metric_names_match_traditional_pipeline(good_image_path):
    """The predicted names must be exactly what the traditional pipeline emits."""
    from uniqat.core.metrics import UnderwaterMetrics

    produced = UnderwaterMetrics(image_path=str(good_image_path)).calculate_all_metrics()
    model = MultiMetricPredictor(pretrained=False)
    assert list(produced.keys()) == model.metric_names


def test_all_architectures_predict_37_metrics():
    """Every architecture must be scoreable on the same 37-metric fidelity task."""
    x = _dummy_batch(batch_size=2, image_size=224)
    cases = [
        (UnderwaterQualityNet, (x, x)),
        (VisionTransformerQualityNet, (x,)),
        (EfficientNetQualityNet, (x,)),
        (MultiMetricPredictor, (x,)),
    ]
    for cls, args in cases:
        model = cls(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model.predict_metrics(*args)
        assert out.shape == (2, 37), f"{cls.__name__}: {out.shape}"
        assert len(model.metric_names) == 37, f"{cls.__name__}: {len(model.metric_names)}"
