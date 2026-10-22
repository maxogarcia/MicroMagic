"""Integration smoke tests."""

import micromagic


def test_integration_environment_smoke() -> None:
    """Verify integration test framework runs and accesses package attributes."""
    assert hasattr(micromagic, "__version__")
    assert micromagic.__version__ == "0.1.0"
