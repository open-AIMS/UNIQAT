"""Command-line interface for UNIQAT assessment.

Entry point: ``uniqat-assess``. Installed by the ``[project.scripts]`` table
in ``pyproject.toml``. Dispatches to one of two subcommands:

- ``uniqat-assess single`` assesses a single image and writes a text report,
  JSON metrics, and optional visualisation to an output directory.
- ``uniqat-assess batch`` assesses every image in a directory (optionally
  recursively), writing a CSV of scores and per-image JSON reports.

Both subcommands are thin wrappers around the reference scripts in
``scripts/``; the scripts are the canonical implementations kept for SLURM
and power-user workflows. See ``uniqat-assess <subcommand> --help`` for the
full argument list of each subcommand.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _find_scripts_dir() -> Path:
    """Locate the repository's ``scripts/`` directory.

    The CLI dispatches to ``scripts/assess_single.py`` and
    ``scripts/assess_batch.py``. In editable installs
    (``pip install -e .``) the installed package lives at
    ``<repo>/src/uniqat``, so the scripts sit two directories up from this
    file. If the scripts directory cannot be found, an actionable error is
    raised.

    Returns
    -------
    pathlib.Path
        Absolute path to the ``scripts/`` directory.

    Raises
    ------
    FileNotFoundError
        If the ``scripts/`` directory cannot be located. This typically
        means UNIQAT was installed from a wheel without the repository
        checkout; clone https://github.com/open-AIMS/UNIQAT for the full
        toolkit.
    """
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent.parent / "scripts",
        pkg_dir.parent.parent.parent / "scripts",
        Path.cwd() / "scripts",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "assess_single.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the UNIQAT scripts/ directory. The uniqat-assess "
        "command requires a checkout of the UNIQAT repository. Clone "
        "https://github.com/open-AIMS/UNIQAT and install with "
        "`pip install -e .`."
    )


def _dispatch(script_name: str, forwarded_args: list[str]) -> int:
    """Invoke a reference script in ``scripts/`` with the forwarded arguments.

    Uses :func:`runpy.run_path` so the script's own ``argparse`` shows the
    expected ``--help`` text and the script's error messages are preserved
    verbatim.

    Parameters
    ----------
    script_name : str
        File name of the reference script (for example ``assess_single.py``).
    forwarded_args : list[str]
        Arguments to pass through to the script's ``main()``.

    Returns
    -------
    int
        Exit code. ``0`` on success, non-zero on error.
    """
    script_path = _find_scripts_dir() / script_name
    saved_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *forwarded_args]
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = saved_argv


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``uniqat-assess`` console script.

    Parameters
    ----------
    argv : list[str] or None
        Arguments to parse. If ``None``, reads from ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code. ``0`` on success, non-zero on error.
    """
    parser = argparse.ArgumentParser(
        prog="uniqat-assess",
        description=(
            "UNIQAT assessment command-line interface. "
            "Use `uniqat-assess single --help` or `uniqat-assess batch --help` "
            "for the full argument list of each subcommand."
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser(
        "single",
        help="Assess a single image (forwards to scripts/assess_single.py).",
        add_help=False,
    )
    subparsers.add_parser(
        "batch",
        help="Assess a directory of images (forwards to scripts/assess_batch.py).",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        return 2

    subcommand = argv[0]
    forwarded = argv[1:]

    if subcommand in {"-h", "--help"}:
        parser.print_help()
        return 0
    if subcommand == "single":
        return _dispatch("assess_single.py", forwarded)
    if subcommand == "batch":
        return _dispatch("assess_batch.py", forwarded)

    parser.error(f"Unknown subcommand: {subcommand!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
