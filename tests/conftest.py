"""Shared pytest configuration."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-ollama",
        action="store_true",
        default=False,
        help="Run integration tests that require a local Ollama server",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ollama: marks tests that require a running Ollama server (deselect with '-m \"not ollama\"')",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-ollama"):
        return
    skip_ollama = pytest.mark.skip(reason="need --run-ollama option to run")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip_ollama)
