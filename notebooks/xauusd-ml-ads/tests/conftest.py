"""Shared pytest fixtures.

Tests live outside the `src` package but import from it; Poetry's `packages`
declaration ensures `src` is importable when running `poetry run pytest`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root."""
    return PROJECT_ROOT
