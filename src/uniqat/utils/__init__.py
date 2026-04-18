"""Utility modules for underwater image quality assessment.

Contains the quality report visualiser (:class:`QualityVisualizer`) and, when
CUDA-capable PyTorch is available, GPU-accelerated batch and metric
acceleration helpers (:class:`GPUBatchProcessor`,
:class:`CUDAMetricsAccelerator`). The GPU helpers are imported lazily so
pure-CPU installs are not penalised.
"""

from .visualization import QualityVisualizer

__all__ = ["QualityVisualizer"]


def _register_gpu_helpers() -> None:
    """Attach GPU accelerator classes to the module if torch is available."""
    try:
        from .gpu_accelerator import (
            CUDAMetricsAccelerator,
            GPUBatchProcessor,
            benchmark_gpu_vs_cpu,
        )

        globals().update(
            {
                "GPUBatchProcessor": GPUBatchProcessor,
                "CUDAMetricsAccelerator": CUDAMetricsAccelerator,
                "benchmark_gpu_vs_cpu": benchmark_gpu_vs_cpu,
            }
        )
        __all__.extend(
            ["GPUBatchProcessor", "CUDAMetricsAccelerator", "benchmark_gpu_vs_cpu"]
        )
    except ImportError:
        pass


_register_gpu_helpers()
