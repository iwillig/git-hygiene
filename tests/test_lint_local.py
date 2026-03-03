"""Unit tests for lint_local.py -- local CLI runner."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set required env vars that lint_commits expects (it gets imported transitively)
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("REPO", "owner/repo")
os.environ.setdefault("PR_NUMBER", "1")

import lint_local  # noqa: E402


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------


class TestGitLog:
    def test_parses_commits(self):
        """Commits are parsed from git log output."""
        fake_output = "abc123\x00Fix the build\x00def456\x00Add feature\n\nSome body\x00"
        mock_result = MagicMock()
        mock_result.stdout = fake_output
        mock_result.returncode = 0

        with patch("lint_local.subprocess.run", return_value=mock_result) as mock_run:
            commits = lint_local.git_log(last_n=2)

        assert len(commits) == 2
        assert commits[0]["sha"] == "abc123"
        assert commits[0]["message"] == "Fix the build"
        assert commits[1]["sha"] == "def456"
        assert "Add feature" in commits[1]["message"]
        assert "Some body" in commits[1]["message"]

        # Verify git command included -n flag
        cmd = mock_run.call_args[0][0]
        assert "-n" in cmd
        assert "2" in cmd

    def test_revision_range(self):
        """Revision range is passed to git log."""
        mock_result = MagicMock()
        mock_result.stdout = "abc123\x00Fix\x00"
        mock_result.returncode = 0

        with patch("lint_local.subprocess.run", return_value=mock_result) as mock_run:
            commits = lint_local.git_log(revision_range="main..HEAD")

        cmd = mock_run.call_args[0][0]
        assert "main..HEAD" in cmd

    def test_empty_output(self):
        """Empty git log returns empty list."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 0

        with patch("lint_local.subprocess.run", return_value=mock_result):
            commits = lint_local.git_log(last_n=5)
            assert commits == []


# ---------------------------------------------------------------------------
# check_grammar (local version)
# ---------------------------------------------------------------------------


class TestCheckGrammarLocal:
    def test_calls_languagetool(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_local.requests.post", return_value=mock_resp) as mock_post:
            issues = lint_local.check_grammar(
                "Fix the build",
                "https://api.languagetool.org/v2",
                "en-US",
            )

        assert issues == []
        call_args = mock_post.call_args
        assert "https://api.languagetool.org/v2/check" in call_args[0]

    def test_custom_url(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_local.requests.post", return_value=mock_resp) as mock_post:
            lint_local.check_grammar("Fix", "https://my-lt.example.com/v2", "de-DE")

        call_url = mock_post.call_args[0][0]
        assert call_url == "https://my-lt.example.com/v2/check"
        call_data = mock_post.call_args[1].get("data") or mock_post.call_args.kwargs.get("data")
        assert call_data["language"] == "de-DE"

    def test_custom_word_suppresses_match(self):
        """Words in the custom dictionary are filtered out."""
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

        with patch("lint_local.requests.post", return_value=mock_resp):
            issues = lint_local.check_grammar(
                "Set up an Ollama server",
                "https://api.languagetool.org/v2",
                "en-US",
                custom_words={"ollama"},
            )
            assert issues == []

    def test_builtin_words_used_by_default(self):
        """Built-in dictionary is used when custom_words is not passed."""
        text = "Add Ollama support"
        match = {
            "message": "Possible spelling mistake found.",
            "offset": 4,
            "length": 6,
            "context": {"text": text, "offset": 4, "length": 6},
            "replacements": [{"value": "llama"}],
            "rule": {"id": "MORFOLOGIK_RULE_EN_US"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"matches": [match]}
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_local.requests.post", return_value=mock_resp):
            # No custom_words arg -- uses BUILTIN_WORDS which includes "ollama"
            issues = lint_local.check_grammar(
                text, "https://api.languagetool.org/v2", "en-US"
            )
            assert issues == []

    def test_non_dictionary_word_still_reported(self):
        """Misspellings not in the dictionary are still flagged."""
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

        with patch("lint_local.requests.post", return_value=mock_resp):
            issues = lint_local.check_grammar(
                "Fix teh build",
                "https://api.languagetool.org/v2",
                "en-US",
                custom_words={"ollama"},
            )
            assert len(issues) == 1


# ---------------------------------------------------------------------------
# llm_chat (local version)
# ---------------------------------------------------------------------------


class TestLlmChatLocal:
    def test_calls_ollama_api(self):
        """Verify request is sent to the correct endpoint."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "response text"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("lint_local.requests.post", return_value=mock_resp) as mock_post:
            result = lint_local.llm_chat("prompt", "qwen2.5:0.5b", "http://localhost:11434/v1")

        assert result == "response text"
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:11434/v1/chat/completions"


# ---------------------------------------------------------------------------
# check_structure (local version)
# ---------------------------------------------------------------------------


class TestCheckStructureLocal:
    def test_returns_issues_from_llm(self):
        """LLM response with issues is parsed correctly."""
        llm_response = json.dumps({"issues": ["Missing body"], "score": 5})
        with patch("lint_local.llm_chat", return_value=llm_response):
            issues = lint_local.check_structure("stuff", "qwen2.5:0.5b", "http://localhost:11434/v1")
            assert issues == ["Missing body"]

    def test_no_issues(self):
        llm_response = json.dumps({"issues": [], "score": 9})
        with patch("lint_local.llm_chat", return_value=llm_response):
            issues = lint_local.check_structure("Fix typo", "qwen2.5:0.5b", "http://localhost:11434/v1")
            assert issues == []

    def test_error_returns_error_string(self):
        """If the model call fails, an error string is returned."""
        with patch("lint_local.llm_chat", side_effect=RuntimeError("boom")):
            issues = lint_local.check_structure("stuff", "qwen2.5:0.5b", "http://localhost:11434/v1")
            assert len(issues) == 1
            assert "boom" in issues[0]


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = lint_local.parse_args([])
        assert args.last == 5
        assert args.range is None
        assert args.grammar_only is False
        assert args.structure_only is False
        assert args.model == lint_local.DEFAULT_MODEL
        assert args.api_key == ""
        assert args.api_base == lint_local.DEFAULT_API_BASE

    def test_range(self):
        args = lint_local.parse_args(["--range", "main..HEAD"])
        assert args.range == "main..HEAD"

    def test_last(self):
        args = lint_local.parse_args(["--last", "10"])
        assert args.last == 10

    def test_grammar_only(self):
        args = lint_local.parse_args(["--grammar-only"])
        assert args.grammar_only is True

    def test_structure_only(self):
        args = lint_local.parse_args(["--structure-only"])
        assert args.structure_only is True

    def test_custom_model(self):
        args = lint_local.parse_args(["--model", "tinyllama"])
        assert args.model == "tinyllama"

    def test_api_key(self):
        args = lint_local.parse_args(["--api-key", "sk-test123"])
        assert args.api_key == "sk-test123"

    def test_api_base(self):
        args = lint_local.parse_args(["--api-base", "https://api.openai.com/v1"])
        assert args.api_base == "https://api.openai.com/v1"

    def test_custom_ignore_patterns(self):
        args = lint_local.parse_args(["--ignore-pattern", "^WIP", "--ignore-pattern", "^fixup!"])
        assert args.ignore_pattern == ["^WIP", "^fixup!"]

    def test_custom_words(self):
        args = lint_local.parse_args(["--custom-word", "Shortcut", "--custom-word", "Clubhouse"])
        assert args.custom_word == ["Shortcut", "Clubhouse"]

    def test_custom_words_default(self):
        args = lint_local.parse_args([])
        assert args.custom_word is None


# ---------------------------------------------------------------------------
# main() integration (mocked externals)
# ---------------------------------------------------------------------------


class TestMainLocal:
    def _mock_commits(self):
        return [
            {"sha": "aaa111", "message": "Fix typo in README"},
            {"sha": "bbb222", "message": "Merge branch 'main' into feature"},
            {"sha": "ccc333", "message": "stuff"},
        ]

    def test_main_skips_merge_commits(self, capsys):
        llm_response = json.dumps({"issues": [], "score": 9})

        with (
            patch("lint_local.git_log", return_value=self._mock_commits()),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.llm_chat", return_value=llm_response),
        ):
            lint_local.main(["--last", "3"])

        out = capsys.readouterr().out
        assert "skipped" in out.lower()

    def test_main_grammar_only_skips_llm(self):
        mock_grammar = MagicMock(return_value=[])

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "Fix typo"}]),
            patch("lint_local.check_grammar", mock_grammar),
            patch("lint_local.check_structure") as mock_structure,
        ):
            lint_local.main(["--grammar-only"])

        mock_grammar.assert_called_once()
        mock_structure.assert_not_called()

    def test_main_structure_only_skips_grammar(self):
        llm_response = json.dumps({"issues": [], "score": 9})

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "Fix typo"}]),
            patch("lint_local.check_grammar") as mock_grammar,
            patch("lint_local.llm_chat", return_value=llm_response),
        ):
            lint_local.main(["--structure-only"])

        mock_grammar.assert_not_called()

    def test_main_returns_1_on_issues(self):
        llm_response = json.dumps({"issues": ["Subject is vague"], "score": 3})

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "stuff"}]),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.llm_chat", return_value=llm_response),
        ):
            rc = lint_local.main(["--last", "1"])

        assert rc == 1

    def test_main_returns_0_when_clean(self):
        llm_response = json.dumps({"issues": [], "score": 10})

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "Fix typo in README"}]),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.llm_chat", return_value=llm_response),
        ):
            rc = lint_local.main(["--last", "1"])

        assert rc == 0
