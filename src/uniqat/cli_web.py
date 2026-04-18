"""Command-line interface for the UNIQAT web application.

Entry point: ``uniqat-web``. Installed by the ``[project.scripts]`` table in
``pyproject.toml``. Launches the Gradio web UI defined in ``web_app.py`` at
the repository root.

Requires the optional ``web`` extra (``pip install uniqat[web]``) for
``gradio``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _find_web_app() -> Path:
    """Locate the repository's ``web_app.py`` launcher.

    Returns
    -------
    pathlib.Path
        Absolute path to ``web_app.py``.

    Raises
    ------
    FileNotFoundError
        If ``web_app.py`` cannot be located. This typically means UNIQAT
        was installed from a wheel without the repository checkout; clone
        https://github.com/open-AIMS/UNIQAT for the full toolkit.
    """
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent.parent / "web_app.py",
        pkg_dir.parent.parent.parent / "web_app.py",
        Path.cwd() / "web_app.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate web_app.py. The uniqat-web command requires a "
        "checkout of the UNIQAT repository. Clone "
        "https://github.com/open-AIMS/UNIQAT and install with "
        "`pip install -e .[web]`."
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``uniqat-web`` console script.

    Parameters
    ----------
    argv : list[str] or None
        Arguments to forward to the web app launcher. If ``None``, reads
        from ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code. ``0`` on success, non-zero on error.
    """
    script_path = _find_web_app()
    forwarded = sys.argv[1:] if argv is None else argv

    if forwarded and forwarded[0] in {"-h", "--help"}:
        print(
            "usage: uniqat-web [-h]\n\n"
            "Launch the UNIQAT Gradio web interface at http://localhost:7860.\n\n"
            "optional arguments:\n"
            "  -h, --help  show this help message and exit\n"
        )
        return 0

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
