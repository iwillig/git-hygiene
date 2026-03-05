"""Unit tests for lint_local.py -- local CLI runner."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set required env vars that lint_commits expects (it gets imported transitively)
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("REPO", "owner/repo")
os.environ.setdefault("PR_NUMBER", "1")

import lint_local

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
            lint_local.git_log(revision_range="main..HEAD")

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
            issues = lint_local.check_grammar(text, "https://api.languagetool.org/v2", "en-US")
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
# get_llm_model
# ---------------------------------------------------------------------------


class TestGetLlmModelLocal:
    def test_loads_model(self):
        mock_model = MagicMock()
        with patch("lint_local.llm.get_model", return_value=mock_model) as mock_get:
            result = lint_local.get_llm_model("qwen2.5:0.5b")
        mock_get.assert_called_once_with("qwen2.5:0.5b")
        assert result is mock_model

    def test_sets_api_key(self):
        mock_model = MagicMock()
        mock_model.key = ""
        with patch("lint_local.llm.get_model", return_value=mock_model):
            result = lint_local.get_llm_model("gpt-4o-mini", api_key="sk-test")
        assert result.key == "sk-test"


# ---------------------------------------------------------------------------
# check_structure (local version)
# ---------------------------------------------------------------------------


def _mock_model(response_text: str) -> MagicMock:
    """Create a mock llm.Model that returns the given text."""
    mock_response = MagicMock()
    mock_response.text.return_value = response_text
    mock_model = MagicMock()
    mock_model.prompt.return_value = mock_response
    return mock_model


class TestCheckStructureLocal:
    def test_good_commit(self):
        """LLM response for a good commit is parsed correctly."""
        llm_response = json.dumps(
            {
                "explains_why": True,
                "score": 9,
                "feedback": "Clear motivation provided.",
                "suggestion": None,
            }
        )
        model = _mock_model(llm_response)
        result = lint_local.check_structure(
            "Fix typo\n\nThe URL had a trailing slash.", model=model
        )
        assert result["explains_why"] is True
        assert result["score"] == 9

    def test_poor_commit(self):
        """LLM flags a vague commit."""
        llm_response = json.dumps(
            {
                "explains_why": False,
                "score": 2,
                "feedback": "Does not explain why.",
                "suggestion": "Describe the reason.",
            }
        )
        model = _mock_model(llm_response)
        result = lint_local.check_structure("stuff", model=model)
        assert result["explains_why"] is False
        assert result["suggestion"] is not None

    def test_error_returns_error_dict(self):
        """If the model call fails, an error dict is returned."""
        mock_model = MagicMock()
        mock_model.prompt.side_effect = RuntimeError("boom")
        result = lint_local.check_structure("stuff", model=mock_model)
        assert result["explains_why"] is False
        assert result["score"] == 0
        assert "boom" in result["feedback"]

    def test_system_prompt_passed(self):
        """check_structure passes the system prompt to model.prompt."""
        llm_response = json.dumps(
            {
                "explains_why": True,
                "score": 8,
                "feedback": "Good.",
                "suggestion": None,
            }
        )
        model = _mock_model(llm_response)
        lint_local.check_structure("some commit", model=model)

        model.prompt.assert_called_once()
        call_kwargs = model.prompt.call_args
        assert "system" in call_kwargs.kwargs
        assert "explains the WHY" in call_kwargs.kwargs["system"]

    def test_falls_back_to_model_id(self):
        """When no model object given, loads by model_id."""
        llm_response = json.dumps(
            {
                "explains_why": True,
                "score": 8,
                "feedback": "Good.",
                "suggestion": None,
            }
        )
        model = _mock_model(llm_response)
        with patch("lint_local.get_llm_model", return_value=model) as mock_get:
            lint_local.check_structure("commit msg", model_id="tinyllama")
        mock_get.assert_called_once_with("tinyllama", "")


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

    def test_custom_ignore_patterns(self):
        args = lint_local.parse_args(["--ignore-pattern", "^WIP", "--ignore-pattern", "^fixup!"])
        assert args.ignore_pattern == ["^WIP", "^fixup!"]

    def test_custom_words(self):
        args = lint_local.parse_args(["--custom-word", "Shortcut", "--custom-word", "Clubhouse"])
        assert args.custom_word == ["Shortcut", "Clubhouse"]

    def test_custom_words_default(self):
        args = lint_local.parse_args([])
        assert args.custom_word is None

    def test_no_api_base_arg(self):
        """--api-base was removed; the llm library handles provider URLs."""
        with pytest.raises(SystemExit):
            lint_local.parse_args(["--api-base", "http://example.com"])


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

    def _good_response(self):
        return json.dumps(
            {
                "explains_why": True,
                "score": 9,
                "feedback": "Good commit.",
                "suggestion": None,
            }
        )

    def _bad_response(self):
        return json.dumps(
            {
                "explains_why": False,
                "score": 2,
                "feedback": "Does not explain why the change was made.",
                "suggestion": "stuff\n\nExplain the reason here.",
            }
        )

    def _mock_model_with(self, response_text: str) -> MagicMock:
        return _mock_model(response_text)

    def test_main_skips_merge_commits(self, capsys):
        model = self._mock_model_with(self._good_response())

        with (
            patch("lint_local.git_log", return_value=self._mock_commits()),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.get_llm_model", return_value=model),
        ):
            lint_local.main(["--last", "3"])

        out = capsys.readouterr().out
        assert "skipped" in out.lower()

    def test_main_grammar_only_skips_llm(self):
        mock_grammar = MagicMock(return_value=[])

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "Fix typo"}]),
            patch("lint_local.check_grammar", mock_grammar),
            patch("lint_local.get_llm_model") as mock_get_model,
        ):
            lint_local.main(["--grammar-only"])

        mock_grammar.assert_called_once()
        mock_get_model.assert_not_called()

    def test_main_structure_only_skips_grammar(self):
        model = self._mock_model_with(self._good_response())

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "Fix typo"}]),
            patch("lint_local.check_grammar") as mock_grammar,
            patch("lint_local.get_llm_model", return_value=model),
        ):
            lint_local.main(["--structure-only"])

        mock_grammar.assert_not_called()

    def test_main_returns_1_on_issues(self):
        model = self._mock_model_with(self._bad_response())

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "stuff"}]),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.get_llm_model", return_value=model),
        ):
            rc = lint_local.main(["--last", "1"])

        assert rc == 1

    def test_main_returns_0_when_clean(self):
        clean = json.dumps(
            {
                "explains_why": True,
                "score": 10,
                "feedback": "",
                "suggestion": None,
            }
        )
        model = self._mock_model_with(clean)

        with (
            patch(
                "lint_local.git_log",
                return_value=[{"sha": "aaa111", "message": "Fix typo in README"}],
            ),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.get_llm_model", return_value=model),
        ):
            rc = lint_local.main(["--last", "1"])

        assert rc == 0

    def test_main_returns_0_for_high_score_with_positive_feedback(self):
        """Regression test: positive feedback on a 10/10 commit should not fail.

        Bug: feedback was always treated as an issue, even positive feedback.
        Fix: only treat feedback as an issue if score < 7 or explains_why is False.
        """
        high_score_with_positive_feedback = json.dumps(
            {
                "explains_why": True,
                "score": 10,
                "feedback": "The commit explains the reason for the change, which is excellent.",
                "suggestion": None,
            }
        )
        model = self._mock_model_with(high_score_with_positive_feedback)

        with (
            patch(
                "lint_local.git_log",
                return_value=[
                    {
                        "sha": "8eed9917",
                        "message": "Allow a looser definition of what a good git message means",
                    }
                ],
            ),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.get_llm_model", return_value=model),
        ):
            rc = lint_local.main(["--last", "1"])

        # Should return 0 (success) because score is 10/10
        assert rc == 0
