"""Verify every UNIQAT console-script entry point exits 0 on ``--help``."""

from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def test_uniqat_assess_help():
    result = _run([sys.executable, "-m", "uniqat.cli", "--help"])
    assert result.returncode == 0, result.stderr
    assert "uniqat-assess" in result.stdout


def test_uniqat_assess_single_help():
    result = _run([sys.executable, "-m", "uniqat.cli", "single", "--help"])
    assert result.returncode == 0, result.stderr


def test_uniqat_assess_batch_help():
    result = _run([sys.executable, "-m", "uniqat.cli", "batch", "--help"])
    assert result.returncode == 0, result.stderr


def test_uniqat_train_help():
    result = _run([sys.executable, "-m", "uniqat.cli_train", "--help"])
    assert result.returncode == 0, result.stderr


def test_uniqat_web_help():
    result = _run([sys.executable, "-m", "uniqat.cli_web", "--help"])
    assert result.returncode == 0, result.stderr
