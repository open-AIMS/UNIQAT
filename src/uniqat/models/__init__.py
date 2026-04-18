"""Deep learning models for underwater image quality assessment.

Three architectures are provided for fast metric approximation:

- :class:`UnderwaterQualityNet`: custom CNN with spatial attention.
- :class:`EfficientNetQualityNet`: EfficientNet-B0 / B3 backbone with a
  quality head.
- :class:`VisionTransformerQualityNet`: ViT-B/16 backbone with a quality
  head.

:class:`MultiMetricPredictor` is a shared multi-output regression head
that can be attached to any of the backbones above.

:class:`QualityAssessmentTrainer` wraps the training loop with
albumentations-based augmentation, mixed-precision training, and
checkpoint management.

All classes require the optional ``deep`` extra
(``pip install uniqat[deep]``), which pulls in ``torch``, ``torchvision``,
``timm``, and ``albumentations``.
"""

from .deep_models import (
    EfficientNetQualityNet,
    MultiMetricPredictor,
    SEBlock,
    SpatialAttention,
    UnderwaterQualityNet,
    VisionTransformerQualityNet,
)
from .trainer import (
    QualityAssessmentTrainer,
    UnderwaterDataset,
    create_synthetic_dataset,
)

__all__ = [
    "UnderwaterQualityNet",
    "EfficientNetQualityNet",
    "VisionTransformerQualityNet",
    "MultiMetricPredictor",
    "SpatialAttention",
    "SEBlock",
    "QualityAssessmentTrainer",
    "UnderwaterDataset",
    "create_synthetic_dataset",
]
