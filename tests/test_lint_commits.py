"""Unit tests for lint_commits.py — grammar & structure helpers."""

from __future__ import annotations

import json
import os
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set required env vars before importing the module
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("REPO", "owner/repo")
os.environ.setdefault("PR_NUMBER", "1")

import llm  # noqa: E402
import lint_commits  # noqa: E402


# ---------------------------------------------------------------------------
# Grammar check
# ---------------------------------------------------------------------------


class TestCheckGrammar:
    def test_no_issues(self):
        """LanguageTool returns no matches → empty list."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp):
            issues = lint_commits.check_grammar("Fix the broken build")
            assert issues == []

    def test_with_issues(self):
        """Issues are extracted properly from LanguageTool response."""
        match = {
            "message": "Possible spelling mistake",
            "context": {"text": "Fix teh build", "offset": 4, "length": 3},
            "replacements": [{"value": "the"}, {"value": "tech"}],
            "rule": {"id": "MORFOLOGIK_RULE_EN_US"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": [match]}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp):
            issues = lint_commits.check_grammar("Fix teh build")
            assert len(issues) == 1
            assert issues[0]["message"] == "Possible spelling mistake"
            assert issues[0]["replacements"] == ["the", "tech"]
            assert issues[0]["rule"] == "MORFOLOGIK_RULE_EN_US"


# ---------------------------------------------------------------------------
# _load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_returns_registered_model(self):
        """Standard llm.get_model path works for registered models."""
        mock_model = MagicMock()
        with patch("lint_commits.llm.get_model", return_value=mock_model):
            result = lint_commits._load_model("gpt-4o-mini")
            assert result is mock_model

    def test_falls_back_to_mlx_model(self):
        """When llm.get_model fails, fall back to MlxModel."""
        mock_model = MagicMock()
        mock_mlx_module = MagicMock()
        mock_mlx_module.MlxModel.return_value = mock_model

        with patch("lint_commits.llm.get_model", side_effect=llm.UnknownModelError("nope")):
            with patch.dict("sys.modules", {"llm_mlx": mock_mlx_module}):
                result = lint_commits._load_model("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
                assert result is mock_model
                mock_mlx_module.MlxModel.assert_called_once_with(
                    "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
                )

    def test_raises_when_no_mlx_plugin(self):
        """RuntimeError if model unknown and llm-mlx not installed."""
        with patch("lint_commits.llm.get_model", side_effect=llm.UnknownModelError("nope")):
            with patch.dict("sys.modules", {"llm_mlx": None}):
                with pytest.raises(RuntimeError, match="not registered"):
                    lint_commits._load_model("mlx-community/some-model")


# ---------------------------------------------------------------------------
# Structure check
# ---------------------------------------------------------------------------


class TestCheckStructure:
    def test_clean_response(self):
        """LLM returns clean JSON with issues."""
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps(
            {"issues": ["Subject is vague"], "score": 4}
        )
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        with patch("lint_commits._load_model", return_value=mock_model):
            issues = lint_commits.check_structure("updates")
            assert issues == ["Subject is vague"]

    def test_no_issues(self):
        """LLM finds no problems."""
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps({"issues": [], "score": 9})
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        with patch("lint_commits._load_model", return_value=mock_model):
            issues = lint_commits.check_structure(
                "Add user avatar upload\n\nUsers requested the ability to upload custom avatars."
            )
            assert issues == []

    def test_markdown_fenced_response(self):
        """LLM wraps response in markdown fences — still parsed."""
        raw = '```json\n{"issues": ["Missing body"], "score": 5}\n```'
        mock_response = MagicMock()
        mock_response.text.return_value = raw
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        with patch("lint_commits._load_model", return_value=mock_model):
            issues = lint_commits.check_structure("Fix bug")
            assert issues == ["Missing body"]

    def test_llm_failure(self):
        """If the LLM call explodes, we get an error string back."""
        with patch("lint_commits._load_model", side_effect=RuntimeError("timeout")):
            issues = lint_commits.check_structure("something")
            assert len(issues) == 1
            assert "timeout" in issues[0]


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_no_issues(self):
        report = lint_commits.build_report([])
        assert "All Commits Look Good" in report

    def test_with_grammar_issue(self):
        ci = lint_commits.CommitIssue(
            sha="abc12345",
            message="Fix teh build",
            grammar_issues=[
                {
                    "message": "Spelling mistake",
                    "replacements": ["the"],
                    "rule": "SPELL",
                    "context": "Fix teh build",
                    "offset": 4,
                    "length": 3,
                }
            ],
        )
        report = lint_commits.build_report([ci])
        assert "Issues Found" in report
        assert "abc12345" in report
        assert "Spelling mistake" in report

    def test_with_structure_issue(self):
        ci = lint_commits.CommitIssue(
            sha="def67890",
            message="stuff",
            structure_issues=["Subject is vague"],
        )
        report = lint_commits.build_report([ci])
        assert "Subject is vague" in report


# ---------------------------------------------------------------------------
# Ignore patterns
# ---------------------------------------------------------------------------


class TestIgnorePatterns:
    def test_merge_commit_ignored(self):
        """Merge commits should match the default ignore pattern."""
        pattern = re.compile(r"^Merge\s")
        assert pattern.search("Merge branch 'main' into feature")

    def test_normal_commit_not_ignored(self):
        pattern = re.compile(r"^Merge\s")
        assert not pattern.search("Fix login timeout")
