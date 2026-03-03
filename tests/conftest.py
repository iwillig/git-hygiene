"""Shared pytest configuration."""


def pytest_addoption(parser):
    """Add --run-mlx CLI flag to pytest."""
    parser.addoption(
        "--run-mlx",
        action="store_true",
        default=False,
        help="Run integration tests that require a local MLX model (Apple Silicon)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "mlx: requires a local MLX model (Apple Silicon)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-mlx"):
        return
    import pytest

    skip = pytest.mark.skip(reason="Need --run-mlx option to run")
    for item in items:
        if "mlx" in item.keywords:
            item.add_marker(skip)
