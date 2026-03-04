"""
Integration tests that run against a real local Ollama model via the llm library.

These are SKIPPED by default.  Run them explicitly with:

    pipenv run pytest tests/test_integration_ollama.py -v --run-ollama

Prerequisites:
    1. Ollama must be running: ollama serve
    2. Pull a model: ollama pull qwen2.5:0.5b
    3. Install plugins: pip install llm llm-ollama

Or to use a specific model:

    OLLAMA_MODEL=tinyllama \
        pipenv run pytest tests/test_integration_ollama.py -v --run-ollama
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

DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ollama_model_id():
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


@pytest.fixture(scope="module")
def ollama_model(ollama_model_id):
    """Load the model once per test module via the llm library."""
    from lint_local import get_llm_model
    return get_llm_model(ollama_model_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.ollama
class TestOllamaStructureCheck:
    """Run the real structure-check prompt against a local Ollama model."""

    def _check(self, model, message: str) -> dict:
        """Send the commit message with the system prompt and parse the JSON response."""
        from lint_local import SYSTEM_PROMPT, _parse_llm_json

        response = model.prompt(message, system=SYSTEM_PROMPT)
        return _parse_llm_json(response.text())

    def test_good_commit(self, ollama_model):
        """A well-formed commit that explains why should score high."""
        result = self._check(
            ollama_model,
            "Add rate limiting to the login endpoint\n\n"
            "Without rate limiting, the login endpoint is vulnerable to brute-force\n"
            "attacks. This adds a 5-request-per-minute limit per IP address using\n"
            "Redis as the backing store.",
        )
        assert "explains_why" in result
        assert "score" in result
        assert "feedback" in result
        assert result["explains_why"] is True
        assert result["score"] >= 6

    def test_vague_commit(self, ollama_model):
        """A vague one-word commit should be flagged as not explaining why."""
        result = self._check(ollama_model, "updates")
        assert "explains_why" in result
        assert result["explains_why"] is False
        assert result["score"] <= 5

    def test_returns_valid_json(self, ollama_model):
        """Even for edge cases the model should return parseable JSON."""
        result = self._check(ollama_model, "fix stuff")
        assert isinstance(result, dict)
        assert "explains_why" in result
        assert "score" in result
        assert "feedback" in result
        assert "suggestion" in result


@pytest.mark.ollama
class TestOllamaEndToEnd:
    """End-to-end: run lint_local.check_structure with a real model."""

    def test_check_structure_returns_dict(self, ollama_model):
        """check_structure should return a dict with the expected keys."""
        from lint_local import check_structure

        result = check_structure("misc changes", model=ollama_model)
        assert isinstance(result, dict)
        assert "explains_why" in result
        assert "score" in result
        assert "feedback" in result
        assert result["explains_why"] is False

    def test_check_structure_clean_commit(self, ollama_model):
        """A good commit should explain why and score high."""
        from lint_local import check_structure

        result = check_structure(
            "Refactor user authentication module\n\n"
            "The previous implementation mixed session management with password\n"
            "hashing. This separates concerns to improve testability.",
            model=ollama_model,
        )
        assert isinstance(result, dict)
        assert result["explains_why"] is True
        assert result["score"] >= 6
