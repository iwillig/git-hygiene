"""
Integration tests that run against a real local Ollama model.

These are SKIPPED by default.  Run them explicitly with:

    pipenv run pytest tests/test_integration_ollama.py -v --run-ollama

Prerequisites:
    1. Ollama must be running: ollama serve
    2. Pull a model: ollama pull qwen2.5:0.5b

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
DEFAULT_API_BASE = "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ollama_model():
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


@pytest.fixture(scope="module")
def api_base():
    return os.environ.get("OLLAMA_API_BASE", DEFAULT_API_BASE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.ollama
class TestOllamaStructureCheck:
    """Run the real structure-check prompt against a local Ollama model."""

    def _check(self, model: str, api_base: str, message: str) -> dict:
        """Send the structure prompt and parse the JSON response."""
        from lint_local import STRUCTURE_PROMPT, llm_chat

        prompt = STRUCTURE_PROMPT.format(message=message)
        text = llm_chat(prompt, model, api_base).strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])

        return json.loads(text)

    def test_good_commit(self, ollama_model, api_base):
        """A well-formed commit should get few/no issues and a high score."""
        result = self._check(
            ollama_model,
            api_base,
            "Add rate limiting to the login endpoint\n\n"
            "Without rate limiting, the login endpoint is vulnerable to brute-force\n"
            "attacks. This adds a 5-request-per-minute limit per IP address using\n"
            "Redis as the backing store.",
        )
        assert "issues" in result
        assert "score" in result
        assert isinstance(result["issues"], list)
        assert result["score"] >= 6

    def test_vague_commit(self, ollama_model, api_base):
        """A vague one-word commit should be flagged."""
        result = self._check(ollama_model, api_base, "updates")
        assert "issues" in result
        assert len(result["issues"]) > 0

    def test_returns_valid_json(self, ollama_model, api_base):
        """Even for edge cases the model should return parseable JSON."""
        result = self._check(ollama_model, api_base, "fix stuff")
        assert isinstance(result, dict)
        assert "issues" in result
        assert "score" in result


@pytest.mark.ollama
class TestOllamaEndToEnd:
    """End-to-end: run lint_local.check_structure with a real model."""

    def test_check_structure_returns_list(self, ollama_model, api_base):
        """check_structure should return a list of strings."""
        from lint_local import check_structure

        issues = check_structure("misc changes", ollama_model, api_base)
        assert isinstance(issues, list)
        assert len(issues) >= 1
        assert all(isinstance(i, str) for i in issues)

    def test_check_structure_clean_commit(self, ollama_model, api_base):
        """A good commit should return few or no issues."""
        from lint_local import check_structure

        issues = check_structure(
            "Refactor user authentication module\n\n"
            "The previous implementation mixed session management with password\n"
            "hashing. This separates concerns to improve testability.",
            ollama_model,
            api_base,
        )
        assert isinstance(issues, list)
        assert len(issues) <= 2
