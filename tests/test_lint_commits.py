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
            "offset": 4,
            "length": 3,
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

    def test_custom_word_suppresses_match(self):
        """A match on a word in the custom dictionary is suppressed."""
        match = {
            "message": "Possible spelling mistake found.",
            "offset": 10,
            "length": 6,
            "context": {"text": "Set up an Ollama server", "offset": 10, "length": 6},
            "replacements": [{"value": "llama"}],
            "rule": {"id": "MORFOLOGIK_RULE_EN_US"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": [match]}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp):
            issues = lint_commits.check_grammar(
                "Set up an Ollama server", custom_words={"ollama"}
            )
            assert issues == []

    def test_custom_word_case_insensitive(self):
        """Custom words matching is case-insensitive."""
        match = {
            "message": "Possible spelling mistake found.",
            "offset": 8,
            "length": 5,
            "context": {"text": "Use the NGINX proxy", "offset": 8, "length": 5},
            "replacements": [],
            "rule": {"id": "MORFOLOGIK_RULE_EN_US"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": [match]}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp):
            issues = lint_commits.check_grammar(
                "Use the NGINX proxy", custom_words={"nginx"}
            )
            assert issues == []

    def test_non_dictionary_word_still_reported(self):
        """Words not in the custom dictionary are still reported."""
        match = {
            "message": "Possible spelling mistake found.",
            "offset": 4,
            "length": 3,
            "context": {"text": "Fix teh build", "offset": 4, "length": 3},
            "replacements": [{"value": "the"}],
            "rule": {"id": "MORFOLOGIK_RULE_EN_US"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": [match]}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp):
            issues = lint_commits.check_grammar(
                "Fix teh build", custom_words={"ollama", "nginx"}
            )
            assert len(issues) == 1

    def test_builtin_words_used_by_default(self):
        """When no custom_words passed, the built-in dictionary is used."""
        # "ollama" is in BUILTIN_WORDS / CUSTOM_WORDS
        match = {
            "message": "Possible spelling mistake found.",
            "offset": 10,
            "length": 6,
            "context": {"text": "Set up an Ollama server", "offset": 10, "length": 6},
            "replacements": [{"value": "llama"}],
            "rule": {"id": "MORFOLOGIK_RULE_EN_US"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": [match]}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp):
            # No custom_words arg -- uses module-level CUSTOM_WORDS
            issues = lint_commits.check_grammar("Set up an Ollama server")
            assert issues == []


# ---------------------------------------------------------------------------
# _extract_flagged_word
# ---------------------------------------------------------------------------


class TestExtractFlaggedWord:
    def test_extracts_word(self):
        text = "Set up an Ollama server"
        match = {"offset": 10, "length": 6}
        assert lint_commits._extract_flagged_word(match, text) == "Ollama"

    def test_returns_empty_on_bad_offset(self):
        match = {"offset": 100, "length": 3}
        assert lint_commits._extract_flagged_word(match, "short") == ""

    def test_returns_empty_on_zero_length(self):
        match = {"offset": 0, "length": 0}
        assert lint_commits._extract_flagged_word(match, "text") == ""


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
        # With no system prompt, only one message
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hi"

    def test_system_prompt_sent_as_system_message(self):
        """When a system_prompt is provided, it is the first message."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_commits.requests.post", return_value=mock_resp) as mock_post:
            lint_commits.llm_chat(
                "commit msg", "qwen2.5:0.5b", "http://localhost:11434/v1",
                system_prompt="You are a reviewer.",
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are a reviewer."
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "commit msg"

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
    def test_good_commit(self):
        """LLM recognises a commit that explains why."""
        llm_response = json.dumps({
            "explains_why": True,
            "score": 9,
            "feedback": "Good commit message that explains the motivation.",
            "suggestion": None,
        })
        with patch("lint_commits.llm_chat", return_value=llm_response):
            result = lint_commits.check_structure(
                "Add user avatar upload\n\nUsers requested the ability to upload custom avatars."
            )
            assert result["explains_why"] is True
            assert result["score"] == 9
            assert result["suggestion"] is None

    def test_poor_commit(self):
        """LLM flags a commit that does not explain why."""
        llm_response = json.dumps({
            "explains_why": False,
            "score": 2,
            "feedback": "This commit only states what changed, not why.",
            "suggestion": "Fix typo in README\n\nThe API endpoint URL had a trailing slash that caused 404 errors.",
        })
        with patch("lint_commits.llm_chat", return_value=llm_response):
            result = lint_commits.check_structure("updates")
            assert result["explains_why"] is False
            assert result["score"] == 2
            assert result["suggestion"] is not None

    def test_markdown_fenced_response(self):
        """LLM wraps response in markdown fences -- still parsed."""
        inner = json.dumps({
            "explains_why": False, "score": 3,
            "feedback": "Missing context.", "suggestion": None,
        })
        llm_response = f"```json\n{inner}\n```"
        with patch("lint_commits.llm_chat", return_value=llm_response):
            result = lint_commits.check_structure("Fix bug")
            assert result["score"] == 3

    def test_llm_failure(self):
        """If the LLM call explodes, we get an error dict back."""
        with patch("lint_commits.llm_chat", side_effect=RuntimeError("timeout")):
            result = lint_commits.check_structure("something")
            assert result["explains_why"] is False
            assert result["score"] == 0
            assert "timeout" in result["feedback"]

    def test_system_prompt_is_used(self):
        """check_structure passes the system prompt to llm_chat."""
        llm_response = json.dumps({
            "explains_why": True, "score": 8,
            "feedback": "Good.", "suggestion": None,
        })
        with patch("lint_commits.llm_chat", return_value=llm_response) as mock_chat:
            lint_commits.check_structure("some commit")

        call_kwargs = mock_chat.call_args
        # system_prompt should be passed as a keyword argument
        assert "system_prompt" in call_kwargs.kwargs
        assert "explains the WHY" in call_kwargs.kwargs["system_prompt"]


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
            structure_issues=["This commit does not explain why."],
            score=2,
            suggestion="Fix login timeout\n\nThe previous 5s timeout was too short for slow networks.",
        )
        report = lint_commits.build_report([ci])
        assert "does not explain why" in report
        assert "2/10" in report
        assert "Fix login timeout" in report


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
