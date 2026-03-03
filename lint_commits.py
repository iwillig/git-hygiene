#!/usr/bin/env python3
"""
Git Hygiene — lint commit messages for grammar and structure quality.

Grammar:      LanguageTool API
Structure:    LLM via OpenAI-compatible API (Ollama local or remote provider)
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

import requests

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["REPO"]  # owner/repo
PR_NUMBER = os.environ["PR_NUMBER"]
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:0.5b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "http://localhost:11434/v1")
USE_LOCAL_MODEL = os.environ.get("USE_LOCAL_MODEL", "true").lower() == "true"
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
        return text[offset:offset + length]
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
# LLM structure check (OpenAI-compatible API)
# ---------------------------------------------------------------------------

STRUCTURE_PROMPT = """\
You are a senior software engineer reviewing a git commit message.

Evaluate the commit message below and return a JSON object with exactly these keys:
- "issues": a list of short strings describing problems (empty list if none)
- "score": an integer 1-10 (10 = perfect)

Check for:
1. Does the subject line summarise WHAT changed? (imperative mood preferred)
2. Is there a body that explains WHY the change was made (motivation / context)?
   A one-line fix like "Fix typo" is acceptable — but non-trivial changes need a body.
3. Is the subject <= 72 characters?
4. Is the subject separated from the body by a blank line (if a body exists)?
5. Does the message avoid vague wording like "fix stuff", "updates", "misc changes"?

Return ONLY the JSON object, no markdown fences.

Commit message:
\"\"\"
{message}
\"\"\"
"""


def llm_chat(prompt: str, model: str, api_base: str, api_key: str = "") -> str:
    """Send a chat completion request to an OpenAI-compatible API and return the response text."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    resp = requests.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def check_structure(message: str) -> list[str]:
    """Use an LLM to evaluate commit message structure. Returns a list of issue strings."""
    prompt = STRUCTURE_PROMPT.format(message=message)

    try:
        text = llm_chat(prompt, LLM_MODEL, LLM_API_BASE, LLM_API_KEY).strip()

        # Strip markdown fences if the model wraps them
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        data = json.loads(text)
        return data.get("issues", [])
    except Exception as exc:
        print(f"WARNING: LLM structure check failed: {exc}", file=sys.stderr)
        return [f"LLM analysis error: {exc}"]


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
                    suggestion = f' -> try: *{", ".join(gi["replacements"])}*'
                lines.append(f'- {gi["message"]}{suggestion}  (`{gi["rule"]}`)')
            lines.append("")

        if r.structure_issues:
            lines.append("**Structure issues:**\n")
            for si in r.structure_issues:
                lines.append(f"- {si}")
            lines.append("")

    lines.append("---")
    lines.append("*Powered by [Git Hygiene](https://github.com/shortcut/git-hygiene)*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if USE_LOCAL_MODEL:
        print(f"Using local Ollama model: {LLM_MODEL}")
    else:
        print(f"Using remote model: {LLM_MODEL} at {LLM_API_BASE}")

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
        try:
            ci.grammar_issues = check_grammar(message)
        except Exception as exc:
            print(f"      WARNING: Grammar check failed: {exc}", file=sys.stderr)

        # Structure check
        ci.structure_issues = check_structure(message)

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
