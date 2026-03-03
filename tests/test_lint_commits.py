"""Unit tests for lint_commits.py -- grammar & structure helpers."""

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

import lint_commits  # noqa: E402


# ---------------------------------------------------------------------------
# Grammar check
# ---------------------------------------------------------------------------


class TestCheckGrammar:
    def test_no_issues(self):
        """LanguageTool returns no matches -> empty list."""
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
# llm_chat
# ---------------------------------------------------------------------------


class TestLlmChat:
    def test_sends_correct_request(self):
        """Verify the request payload sent to the OpenAI-compatible API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp) as mock_post:
            result = lint_commits.llm_chat("Hi", "qwen2.5:0.5b", "http://localhost:11434/v1")

        assert result == "Hello!"
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "qwen2.5:0.5b"
        assert payload["messages"][0]["content"] == "Hi"

    def test_includes_auth_header(self):
        """API key is passed as a Bearer token."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp) as mock_post:
            lint_commits.llm_chat("Hi", "gpt-4o-mini", "https://api.openai.com/v1", "sk-test")

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer sk-test"

    def test_no_auth_header_when_empty(self):
        """No Authorization header when api_key is empty."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp) as mock_post:
            lint_commits.llm_chat("Hi", "qwen2.5:0.5b", "http://localhost:11434/v1", "")

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Structure check
# ---------------------------------------------------------------------------


class TestCheckStructure:
    def test_clean_response(self):
        """LLM returns clean JSON with issues."""
        llm_response = json.dumps({"issues": ["Subject is vague"], "score": 4})
        with patch("lint_commits.llm_chat", return_value=llm_response):
            issues = lint_commits.check_structure("updates")
            assert issues == ["Subject is vague"]

    def test_no_issues(self):
        """LLM finds no problems."""
        llm_response = json.dumps({"issues": [], "score": 9})
        with patch("lint_commits.llm_chat", return_value=llm_response):
            issues = lint_commits.check_structure(
                "Add user avatar upload\n\nUsers requested the ability to upload custom avatars."
            )
            assert issues == []

    def test_markdown_fenced_response(self):
        """LLM wraps response in markdown fences -- still parsed."""
        llm_response = '```json\n{"issues": ["Missing body"], "score": 5}\n```'
        with patch("lint_commits.llm_chat", return_value=llm_response):
            issues = lint_commits.check_structure("Fix bug")
            assert issues == ["Missing body"]

    def test_llm_failure(self):
        """If the LLM call explodes, we get an error string back."""
        with patch("lint_commits.llm_chat", side_effect=RuntimeError("timeout")):
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
