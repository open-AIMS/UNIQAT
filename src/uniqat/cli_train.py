"""Command-line interface for training UNIQAT deep learning quality models.

Entry point: ``uniqat-train``. Installed by the ``[project.scripts]`` table in
``pyproject.toml``. Thin wrapper around
``scripts/train_quality_model.py``, which is the canonical training reference
implementation retained for SLURM and power-user workflows.

Requires the optional ``deep`` extra (``pip install uniqat[deep]``) for
``torch``, ``torchvision``, ``timm``, and ``albumentations``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _find_script() -> Path:
    """Locate ``scripts/train_quality_model.py``.

    Returns
    -------
    pathlib.Path
        Absolute path to the training script.

    Raises
    ------
    FileNotFoundError
        If the training script cannot be located. This typically means
        UNIQAT was installed from a wheel without the repository checkout;
        clone https://github.com/open-AIMS/UNIQAT for the full toolkit.
    """
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent.parent / "scripts" / "train_quality_model.py",
        pkg_dir.parent.parent.parent / "scripts" / "train_quality_model.py",
        Path.cwd() / "scripts" / "train_quality_model.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate scripts/train_quality_model.py. The uniqat-train "
        "command requires a checkout of the UNIQAT repository. Clone "
        "https://github.com/open-AIMS/UNIQAT and install with "
        "`pip install -e .[deep]`."
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``uniqat-train`` console script.

    Parameters
    ----------
    argv : list[str] or None
        Arguments to forward to the training script. If ``None``, reads
        from ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code. ``0`` on success, non-zero on error.
    """
    script_path = _find_script()
    forwarded = sys.argv[1:] if argv is None else argv
    saved_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *forwarded]
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    sys.exit(main())
