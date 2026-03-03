"""
Integration tests that run against a real local MLX model.

These are SKIPPED by default.  Run them explicitly with:

    pipenv run pytest tests/test_integration_mlx.py -v --run-mlx

Or to also pick a specific model:

    MLX_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        pipenv run pytest tests/test_integration_mlx.py -v --run-mlx

The first run will download the model (~278 MB for Qwen2.5-0.5B-Instruct-4bit).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set env vars lint_commits needs at import time
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("REPO", "owner/repo")
os.environ.setdefault("PR_NUMBER", "1")

# Default to a small model for CI-friendliness
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlx_model_name():
    return os.environ.get("MLX_MODEL", DEFAULT_MLX_MODEL)


@pytest.fixture(scope="module")
def mlx_model(mlx_model_name):
    """Load the MLX model once per test module."""
    try:
        from llm_mlx import MlxModel
    except ImportError:
        pytest.skip("llm-mlx is not installed (pip install llm-mlx)")

    return MlxModel(mlx_model_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.mlx
class TestMlxStructureCheck:
    """Run the real structure-check prompt against a local MLX model."""

    def _check(self, model, message: str) -> dict:
        """Send the structure prompt and parse the JSON response."""
        from lint_local import STRUCTURE_PROMPT

        prompt = STRUCTURE_PROMPT.format(message=message)
        response = model.prompt(prompt)
        text = response.text().strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])

        return json.loads(text)

    def test_good_commit(self, mlx_model):
        """A well-formed commit should get few/no issues and a high score."""
        result = self._check(
            mlx_model,
            "Add rate limiting to the login endpoint\n\n"
            "Without rate limiting, the login endpoint is vulnerable to brute-force\n"
            "attacks. This adds a 5-request-per-minute limit per IP address using\n"
            "Redis as the backing store.",
        )
        assert "issues" in result
        assert "score" in result
        assert isinstance(result["issues"], list)
        assert result["score"] >= 6

    def test_vague_commit(self, mlx_model):
        """A vague one-word commit should be flagged."""
        result = self._check(mlx_model, "updates")
        assert "issues" in result
        assert len(result["issues"]) > 0

    def test_returns_valid_json(self, mlx_model):
        """Even for edge cases the model should return parseable JSON."""
        result = self._check(mlx_model, "fix stuff")
        assert isinstance(result, dict)
        assert "issues" in result
        assert "score" in result


@pytest.mark.mlx
class TestMlxEndToEnd:
    """End-to-end: run lint_local.check_structure with a real model."""

    def test_check_structure_returns_list(self, mlx_model_name):
        """check_structure should return a list of strings."""
        from lint_local import check_structure

        issues = check_structure("misc changes", mlx_model_name)
        assert isinstance(issues, list)
        # A vague commit like "misc changes" should produce at least one issue
        assert len(issues) >= 1
        assert all(isinstance(i, str) for i in issues)

    def test_check_structure_clean_commit(self, mlx_model_name):
        """A good commit should return few or no issues."""
        from lint_local import check_structure

        issues = check_structure(
            "Refactor user authentication module\n\n"
            "The previous implementation mixed session management with password\n"
            "hashing. This separates concerns to improve testability.",
            mlx_model_name,
        )
        assert isinstance(issues, list)
        # Allow 0-2 minor issues for a well-formed commit
        assert len(issues) <= 2
