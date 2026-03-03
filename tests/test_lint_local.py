"""Unit tests for lint_local.py — local CLI runner."""

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

import llm  # noqa: E402
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


# ---------------------------------------------------------------------------
# check_structure (local version) — MLX model loading path
# ---------------------------------------------------------------------------


class TestCheckStructureLocal:
    def test_uses_llm_get_model_first(self):
        """Standard llm.get_model path when model is registered."""
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps({"issues": [], "score": 9})
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        with patch("lint_local._try_load_mlx_model", return_value=mock_model):
            issues = lint_local.check_structure("Fix typo", "gpt-4o-mini")
            assert issues == []

    def test_returns_issues_from_llm(self):
        """LLM response with issues is parsed correctly."""
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps(
            {"issues": ["Missing body"], "score": 5}
        )
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        with patch("lint_local._try_load_mlx_model", return_value=mock_model):
            issues = lint_local.check_structure(
                "stuff", "mlx-community/Llama-3.2-3B-Instruct-4bit"
            )
            assert issues == ["Missing body"]

    def test_error_returns_error_string(self):
        """If the model call fails, an error string is returned."""
        with patch("lint_local._try_load_mlx_model", side_effect=RuntimeError("boom")):
            issues = lint_local.check_structure("stuff", "mlx-community/some-model")
            assert len(issues) == 1
            assert "boom" in issues[0]


# ---------------------------------------------------------------------------
# _try_load_mlx_model
# ---------------------------------------------------------------------------


class TestTryLoadMlxModel:
    def test_returns_registered_model(self):
        mock_model = MagicMock()
        with patch("lint_local.llm.get_model", return_value=mock_model):
            result = lint_local._try_load_mlx_model("gpt-4o-mini")
            assert result is mock_model

    def test_falls_back_to_mlx(self):
        """When llm.get_model raises UnknownModelError, fall back to MlxModel."""
        mock_model = MagicMock()

        def fake_get_model(name):
            raise llm.UnknownModelError(name)

        # Mock at the _try_load_mlx_model level — patch the import inside the function
        with patch("lint_local.llm.get_model", side_effect=fake_get_model):
            # The function does a dynamic import of llm_mlx — mock that
            mock_mlx_module = MagicMock()
            mock_mlx_module.MlxModel.return_value = mock_model
            with patch.dict("sys.modules", {"llm_mlx": mock_mlx_module}):
                result = lint_local._try_load_mlx_model(
                    "mlx-community/Llama-3.2-3B-Instruct-4bit"
                )
                assert result is mock_model
                mock_mlx_module.MlxModel.assert_called_once_with(
                    "mlx-community/Llama-3.2-3B-Instruct-4bit"
                )

    def test_raises_when_no_mlx_plugin(self):
        """Raises RuntimeError if model unknown and llm-mlx not available."""
        def fake_get_model(name):
            raise llm.UnknownModelError(name)

        with patch("lint_local.llm.get_model", side_effect=fake_get_model):
            # Simulate llm_mlx not being installed
            with patch.dict("sys.modules", {"llm_mlx": None}):
                with pytest.raises(RuntimeError, match="not registered"):
                    lint_local._try_load_mlx_model("mlx-community/some-model")


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
        assert args.api_key is None

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
        args = lint_local.parse_args(["--model", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"])
        assert args.model == "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

    def test_api_key(self):
        args = lint_local.parse_args(["--api-key", "sk-test123"])
        assert args.api_key == "sk-test123"

    def test_custom_ignore_patterns(self):
        args = lint_local.parse_args(["--ignore-pattern", "^WIP", "--ignore-pattern", "^fixup!"])
        assert args.ignore_pattern == ["^WIP", "^fixup!"]


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
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps({"issues": [], "score": 9})
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        mock_grammar = MagicMock(return_value=[])

        with (
            patch("lint_local.git_log", return_value=self._mock_commits()),
            patch("lint_local.check_grammar", mock_grammar),
            patch("lint_local.llm.get_model", return_value=mock_model),
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
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps({"issues": [], "score": 9})
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "Fix typo"}]),
            patch("lint_local.check_grammar") as mock_grammar,
            patch("lint_local.llm.get_model", return_value=mock_model),
        ):
            lint_local.main(["--structure-only"])

        mock_grammar.assert_not_called()

    def test_main_returns_1_on_issues(self):
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps(
            {"issues": ["Subject is vague"], "score": 3}
        )
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "stuff"}]),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.llm.get_model", return_value=mock_model),
        ):
            rc = lint_local.main(["--last", "1"])

        assert rc == 1

    def test_main_returns_0_when_clean(self):
        mock_response = MagicMock()
        mock_response.text.return_value = json.dumps({"issues": [], "score": 10})
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response

        with (
            patch("lint_local.git_log", return_value=[{"sha": "aaa111", "message": "Fix typo in README"}]),
            patch("lint_local.check_grammar", return_value=[]),
            patch("lint_local.llm.get_model", return_value=mock_model),
        ):
            rc = lint_local.main(["--last", "1"])

        assert rc == 0
