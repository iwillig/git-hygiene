#!/usr/bin/env python3
"""
Git Hygiene — lint commit messages for grammar and structure quality.

Grammar:      LanguageTool API
Structure:    LLM via the llm library (supports Ollama, OpenAI, and many other providers)
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import llm
import requests

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["REPO"]  # owner/repo
PR_NUMBER = os.environ["PR_NUMBER"]
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:0.5b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
ENABLE_GRAMMAR = os.environ.get("ENABLE_GRAMMAR", "false").lower() == "true"
LANGUAGETOOL_URL = os.environ.get("LANGUAGETOOL_URL", "https://api.languagetool.org/v2")
LANGUAGETOOL_LANGUAGE = os.environ.get("LANGUAGETOOL_LANGUAGE", "en-US")
IGNORE_PATTERNS_RAW = os.environ.get("IGNORE_PATTERNS", "")
CUSTOM_WORDS_RAW = os.environ.get("CUSTOM_WORDS", "")
FAIL_ON_ERROR = os.environ.get("FAIL_ON_ERROR", "true").lower() == "true"

IGNORE_PATTERNS: list[re.Pattern] = []
for line in IGNORE_PATTERNS_RAW.strip().splitlines():
    line = line.strip()
    if line:
        IGNORE_PATTERNS.append(re.compile(line))

# Built-in dictionary of common dev/tool names that LanguageTool flags as
# spelling mistakes.  Users can extend this via the custom-words input.
BUILTIN_WORDS: set[str] = {
    "ollama",
    "llama",
    "llm",
    "openai",
    "langchain",
    "github",
    "gitlab",
    "bitbucket",
    "dockerfile",
    "kubernetes",
    "kubectl",
    "redis",
    "postgres",
    "postgresql",
    "mongodb",
    "nginx",
    "fastapi",
    "graphql",
    "grpc",
    "protobuf",
    "webpack",
    "vite",
    "eslint",
    "pytest",
    "mypy",
    "ruff",
    "pipenv",
    "pipfile",
    "pyproject",
    "toml",
    "yaml",
    "json",
    "env",
    "dotenv",
    "cli",
    "api",
    "url",
    "http",
    "https",
    "ssh",
    "tcp",
    "dns",
    "ci",
    "cd",
    "pr",
    "sha",
    "repo",
    "repos",
    "refactor",
    "refactored",
    "refactoring",
    "linter",
    "linting",
    "config",
    "configs",
    "middleware",
    "frontend",
    "backend",
    "monorepo",
    "codebase",
    "README",
    "changelog",
    "pre-commit",
    "deps",
    "dev",
    "devs",
    "param",
    "params",
    "auth",
    "authn",
    "authz",
    "oauth",
    "async",
    "await",
    "goroutine",
    "mutex",
    "stdin",
    "stdout",
    "stderr",
    "args",
    "kwargs",
    "enum",
    "enums",
    "struct",
    "structs",
    "tuple",
    "tuples",
    "bool",
    "int",
    "str",
    "dict",
    "dataclass",
    "namespace",
    "namespaces",
    "runtime",
    "subprocess",
    "plugin",
    "plugins",
    "serializer",
    "deserializer",
    "endpoint",
    "endpoints",
    "webhook",
    "webhooks",
    "cron",
    "regex",
    "noop",
    "wip",
    "fixup",
}

CUSTOM_WORDS: set[str] = {w.lower() for w in BUILTIN_WORDS}
for line in CUSTOM_WORDS_RAW.strip().splitlines():
    word = line.strip()
    if word:
        CUSTOM_WORDS.add(word.lower())

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CommitIssue:
    sha: str
    message: str
    grammar_issues: list[dict] = field(default_factory=list)
    structure_issues: list[str] = field(default_factory=list)
    score: int | None = None
    suggestion: str | None = None

    @property
    def has_issues(self) -> bool:
        return bool(self.grammar_issues or self.structure_issues)


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

GH_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
GH_API = "https://api.github.com"


def fetch_pr_commits() -> list[dict]:
    """Return the list of commits on the pull request."""
    url = f"{GH_API}/repos/{REPO}/pulls/{PR_NUMBER}/commits"
    commits = []
    page = 1
    while True:
        resp = requests.get(url, headers=GH_HEADERS, params={"per_page": 100, "page": page})
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        commits.extend(data)
        page += 1
    return commits


def post_pr_comment(body: str) -> None:
    """Post (or update) a PR comment with the lint results."""
    marker = "<!-- git-hygiene-bot -->"
    body_with_marker = f"{marker}\n{body}"

    # Look for an existing comment to update
    url = f"{GH_API}/repos/{REPO}/issues/{PR_NUMBER}/comments"
    resp = requests.get(url, headers=GH_HEADERS, params={"per_page": 100})
    resp.raise_for_status()
    for comment in resp.json():
        if marker in (comment.get("body") or ""):
            patch_url = comment["url"]
            requests.patch(patch_url, headers=GH_HEADERS, json={"body": body_with_marker})
            return

    # No existing comment — create one
    requests.post(url, headers=GH_HEADERS, json={"body": body_with_marker})


# ---------------------------------------------------------------------------
# LanguageTool grammar check
# ---------------------------------------------------------------------------


def _extract_flagged_word(match: dict, text: str) -> str:
    """Extract the word that LanguageTool flagged from the original text."""
    offset = match.get("offset", 0)
    length = match.get("length", 0)
    if offset >= 0 and length > 0 and offset + length <= len(text):
        return text[offset : offset + length]
    return ""


def check_grammar(text: str, custom_words: set[str] | None = None) -> list[dict]:
    """Return a list of grammar issues from LanguageTool.

    Matches where the flagged word appears in *custom_words* (case-insensitive)
    are silently dropped.
    """
    if custom_words is None:
        custom_words = CUSTOM_WORDS

    resp = requests.post(
        f"{LANGUAGETOOL_URL}/check",
        data={
            "text": text,
            "language": LANGUAGETOOL_LANGUAGE,
            "enabledOnly": "false",
        },
        timeout=30,
    )
    resp.raise_for_status()
    matches = resp.json().get("matches", [])
    issues = []
    for m in matches:
        # Check if the flagged word is in the custom dictionary
        flagged = _extract_flagged_word(m, text)
        if flagged and flagged.lower() in custom_words:
            continue

        issues.append(
            {
                "message": m.get("message", ""),
                "context": m.get("context", {}).get("text", ""),
                "offset": m.get("context", {}).get("offset", 0),
                "length": m.get("context", {}).get("length", 0),
                "replacements": [r["value"] for r in m.get("replacements", [])[:3]],
                "rule": m.get("rule", {}).get("id", ""),
            }
        )
    return issues


# ---------------------------------------------------------------------------
# LLM structure check (via the llm library)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a git commit message reviewer. Your task is to evaluate if a "
    "commit message explains the WHY behind the change, not just the WHAT.\n"
    "\n"
    "A good commit message should:\n"
    "- Explain the reason for the change\n"
    "- Describe the problem being solved or the benefit being added\n"
    "- Provide context for future developers\n"
    "\n"
    "A poor commit message only describes what was changed without explaining why.\n"
    "\n"
    "Examples:\n"
    "- BAD (no why): 'updates', 'fixes', 'misc changes' - these are too vague\n"
    "- GOOD (has why): 'Add rate limiting to prevent brute-force attacks'\n"
    "- GOOD (has why): 'Refactor auth module to improve testability'\n"
    "\n"
    "Analyze the commit message and respond with a JSON object containing:\n"
    '- "explains_why": boolean - true if the message explains why the change was made\n'
    '- "score": number from 0-10 - how well it explains the why (10 being excellent)\n'
    '- "feedback": string - specific feedback on the message (can be positive for high scores)\n'
    '- "suggestion": string or null - ONLY provide a suggested rewrite if score < 7. '
    "For scores 7 and above, this MUST be null.\n"
    "\n"
    "Be fair in your evaluation:\n"
    "- Vague one-word commits get 0-3\n"
    "- Commits that only describe WHAT get 3-6\n"
    "- Commits that explain WHY get 7-10\n"
    "- Don't be overly critical if the WHY is clear, even if it could be more detailed"
)


def get_llm_model(model_id: str, api_key: str = "") -> llm.Model:
    """Load a model via the llm library. Sets the API key if provided."""
    model = llm.get_model(model_id)
    if api_key and hasattr(model, "key"):
        model.key = api_key
    return model


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Strip optional markdown fences and parse JSON from an LLM response.

    Handles common issues with small models producing malformed JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()

    # Try to parse as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Small models sometimes produce invalid JSON with unescaped quotes
        # Try some basic fixes
        import re

        # Fix common issue: "word" inside a string value (should be \"word\")
        # This is a heuristic - may not catch all cases
        lines = text.split("\n")
        fixed_lines = []
        for line in lines:
            # If line contains a key-value pair with quotes in the value
            if '": "' in line and line.count('"') > 4:
                # Try to fix unescaped quotes in string values
                # Match pattern: "key": "value with "quotes" in it"
                match = re.match(r'(\s*"[^"]+"\s*:\s*")(.*?)("\s*[,}]?\s*)$', line)
                if match:
                    prefix, value, suffix = match.groups()
                    # Escape any quotes in the value
                    value = value.replace('"', '\\"')
                    line = prefix + value + suffix
            fixed_lines.append(line)
        text = "\n".join(fixed_lines)
        return json.loads(text)


def check_structure(message: str, model: llm.Model | None = None) -> dict[str, Any]:
    """Use an LLM to evaluate whether a commit message explains *why*.

    Returns a dict with keys: explains_why, score, feedback, suggestion.
    On failure returns a dict with feedback containing the error.
    """
    if model is None:
        model = get_llm_model(LLM_MODEL, LLM_API_KEY)

    try:
        response = model.prompt(message, system=SYSTEM_PROMPT)
        text = response.text()
        result = _parse_llm_json(text)

        # Enforce the rule: only suggest rewrites for scores < 7
        # Small models sometimes don't follow instructions perfectly
        if result.get("score", 0) >= 7:
            result["suggestion"] = None

        return result
    except Exception as exc:
        print(f"WARNING: LLM structure check failed: {exc}", file=sys.stderr)
        return {
            "explains_why": False,
            "score": 0,
            "feedback": f"LLM analysis error: {exc}",
            "suggestion": None,
        }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_report(results: list[CommitIssue]) -> str:
    """Build a Markdown summary suitable for a PR comment."""
    issues_found = any(r.has_issues for r in results)

    lines: list[str] = []
    if issues_found:
        lines.append("## Git Hygiene — Issues Found\n")
    else:
        lines.append("## Git Hygiene — All Commits Look Good!\n")
        return "\n".join(lines)

    for r in results:
        if not r.has_issues:
            continue
        short_sha = r.sha[:8]
        subject = r.message.split("\n", 1)[0][:72]
        lines.append(f"### `{short_sha}` — {subject}\n")

        if r.grammar_issues:
            lines.append("**Grammar issues:**\n")
            for gi in r.grammar_issues:
                suggestion = ""
                if gi["replacements"]:
                    suggestion = f" -> try: *{', '.join(gi['replacements'])}*"
                lines.append(f"- {gi['message']}{suggestion}  (`{gi['rule']}`)")
            lines.append("")

        if r.structure_issues:
            lines.append("**Structure feedback:**\n")
            for si in r.structure_issues:
                lines.append(f"- {si}")
            if r.score is not None:
                lines.append(f"\nScore: **{r.score}/10**")
            if r.suggestion:
                lines.append(f"\nSuggested rewrite:\n> {r.suggestion}")
            lines.append("")

    lines.append("---")
    lines.append("*Powered by [Git Hygiene](https://github.com/shortcut/git-hygiene)*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"Loading model: {LLM_MODEL}")
    model = get_llm_model(LLM_MODEL, LLM_API_KEY)
    print(f"   Model loaded: {model.model_id}")

    print(f"Fetching commits for PR #{PR_NUMBER} in {REPO} ...")
    commits = fetch_pr_commits()
    print(f"   Found {len(commits)} commit(s).")

    results: list[CommitIssue] = []

    for commit in commits:
        sha = commit["sha"]
        message: str = commit["commit"]["message"]
        subject = message.split("\n", 1)[0]

        # Check ignore patterns
        if any(p.search(subject) for p in IGNORE_PATTERNS):
            print(f"   SKIP {sha[:8]} — skipped (matches ignore pattern)")
            continue

        print(f"   CHECK {sha[:8]} — {subject[:60]}")

        ci = CommitIssue(sha=sha, message=message)

        # Grammar check
        if ENABLE_GRAMMAR:
            try:
                ci.grammar_issues = check_grammar(message)
            except Exception as exc:
                print(f"      WARNING: Grammar check failed: {exc}", file=sys.stderr)

        # Structure check
        result = check_structure(message, model=model)
        ci.score = result.get("score")
        ci.suggestion = result.get("suggestion")
        feedback = result.get("feedback", "")

        # Only treat feedback as an issue if the commit doesn't explain why
        # or if it scores below 7 (our threshold for "good enough")
        explains_why = result.get("explains_why", True)
        score = result.get("score", 0)

        if not explains_why or score < 7:
            if feedback:
                ci.structure_issues = [feedback]
            elif not explains_why:
                ci.structure_issues = ["Commit message does not explain why the change was made"]

        results.append(ci)

    # Build report
    report = build_report(results)
    print("\n" + report)

    # Post PR comment
    try:
        post_pr_comment(report)
        print("\nPosted PR comment.")
    except Exception as exc:
        print(f"\nWARNING: Failed to post PR comment: {exc}", file=sys.stderr)

    # Determine exit code
    has_issues = any(r.has_issues for r in results)
    if has_issues and FAIL_ON_ERROR:
        print("\nIssues found — failing the check.")
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
