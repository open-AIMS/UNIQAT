"""
UNIQAT - UNderwater Image QUality Assessment Toolkit

An open-source Python library for automated, reference-free underwater image
quality assessment. Computes 37 metrics across nine categories and produces
task-specific composite quality scores for marine science and computer vision.
"""

from .core.metrics import UnderwaterMetrics
from .core.assessor import UnderwaterImageAssessor, QualityAssessment
from .utils.visualization import QualityVisualizer

__version__ = "1.0.0"
__author__ = "Alzayat Saleh"

__all__ = [
    'UnderwaterMetrics',
    'UnderwaterImageAssessor',
    'QualityAssessment',
    'QualityVisualizer'
]
