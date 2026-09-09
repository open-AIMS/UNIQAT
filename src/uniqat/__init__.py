"""UNIQAT: UNderwater Image QUality Assessment Toolkit.

An open-source Python library for automated, reference-free underwater image
quality assessment. Computes 37 metrics across nine categories and produces
task-specific composite quality scores for marine science and computer vision
applications.

Public API
----------
Core assessment

- :class:`UnderwaterMetrics` - the 37 individual image quality metrics.
- :class:`UnderwaterImageAssessor` - load-once, assess-many image assessor.
- :class:`QualityAssessment` - dataclass holding a single image's full
  assessment (composite scores plus all 37 detailed metrics).
- :func:`assess_image` - one-line convenience wrapper that returns a
  :class:`QualityAssessment` for a single image path.

Video assessment

- :class:`VideoQualityAssessor` - frame-by-frame video assessment with
  temporal stability analysis.
- :class:`VideoQualityAssessment` - dataclass holding a video's full
  assessment (frame-level scores, temporal metrics, recommendations).

Visualisation

- :class:`QualityVisualizer` - composite quality report figure generation.

Deep learning metric approximators

- :class:`UnderwaterQualityNet` - custom CNN for multi-metric regression.
- :class:`EfficientNetQualityNet` - EfficientNet-based quality head.
- :class:`VisionTransformerQualityNet` - ViT-based quality head.
- :class:`MultiMetricPredictor` - multi-output regression head.
- :class:`QualityAssessmentTrainer` - training loop for the deep learning
  models (augmentation, mixed precision, checkpointing).

See the ``examples/`` directory for end-to-end workflows and the
``DEEP_LEARNING_README.md`` file for deep learning model documentation.
"""

from .core.assessor import QualityAssessment, UnderwaterImageAssessor
from .core.metrics import UnderwaterMetrics
from .core.video_assessor import VideoQualityAssessment, VideoQualityAssessor
from .utils.visualization import QualityVisualizer

__version__ = "1.0.1"
__author__ = "Alzayat Saleh"


def assess_image(image_path, scale_factor: float = 1.0) -> QualityAssessment:
    """Assess a single underwater image and return a :class:`QualityAssessment`.

    Thin one-line wrapper around :class:`UnderwaterImageAssessor` for quick
    scripting and notebook use. For batch pipelines, instantiate
    :class:`UnderwaterImageAssessor` directly and reuse it across images.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to an image file readable by OpenCV.
    scale_factor : float, default 1.0
        Optional uniform rescale applied before assessment. Values below
        1.0 downsample the image for faster assessment at a small accuracy
        cost. Use 1.0 for the paper-reported behaviour.

    Returns
    -------
    QualityAssessment
        The full assessment dataclass with composite scores and all 37
        individual metrics.
    """
    assessor = UnderwaterImageAssessor(
        image_path=str(image_path),
        scale_factor=scale_factor,
    )
    return assessor.assess()


__all__ = [
    "__version__",
    # Core
    "UnderwaterMetrics",
    "UnderwaterImageAssessor",
    "QualityAssessment",
    "assess_image",
    # Video
    "VideoQualityAssessor",
    "VideoQualityAssessment",
    # Visualisation
    "QualityVisualizer",
]


def _register_deep_learning_classes() -> None:
    """Attach deep learning model classes to the module if torch is available.

    Torch is an optional dependency, so the deep learning classes are imported
    lazily. If ``torch`` and ``timm`` are installed, the model classes and the
    trainer are exposed at the top level as ``uniqat.UnderwaterQualityNet``
    etc. If not, those names are simply absent and a friendly import hint is
    recorded at ``uniqat.__deep_learning_error__``.
    """
    try:
        from .models.deep_models import (
            EfficientNetQualityNet,
            MultiMetricPredictor,
            UnderwaterQualityNet,
            VisionTransformerQualityNet,
        )
        from .models.trainer import QualityAssessmentTrainer

        globals().update(
            {
                "UnderwaterQualityNet": UnderwaterQualityNet,
                "EfficientNetQualityNet": EfficientNetQualityNet,
                "VisionTransformerQualityNet": VisionTransformerQualityNet,
                "MultiMetricPredictor": MultiMetricPredictor,
                "QualityAssessmentTrainer": QualityAssessmentTrainer,
            }
        )
        __all__.extend(
            [
                "UnderwaterQualityNet",
                "EfficientNetQualityNet",
                "VisionTransformerQualityNet",
                "MultiMetricPredictor",
                "QualityAssessmentTrainer",
            ]
        )
    except ImportError as err:
        globals()["__deep_learning_error__"] = (
            "Deep learning models require the optional 'deep' extra. "
            "Install with `pip install uniqat[deep]`. "
            f"Underlying import error: {err}"
        )


_register_deep_learning_classes()
