"""Smoke tests for micromagic packages and modules."""

import micromagic
import micromagic.evaluation
import micromagic.tournaments


def test_package_import() -> None:
    """Ensure micromagic package can be imported and has version set."""
    assert micromagic.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    """Ensure micromagic subpackages can be imported."""
    assert micromagic.tournaments is not None
    assert micromagic.evaluation is not None
