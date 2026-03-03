#!/usr/bin/env python3
"""
Git Hygiene — local CLI runner.

Lint commit messages from the current git repository without GitHub.
Ideal for local development with mlx-lm models on Apple Silicon.

Usage:
    # Lint the last 5 commits using a local MLX model:
    python lint_local.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Lint commits on a branch compared to main:
    python lint_local.py --range main..HEAD

    # Lint a specific number of recent commits:
    python lint_local.py --last 10

    # Skip the LLM check and only do grammar:
    python lint_local.py --grammar-only

    # Skip grammar and only do LLM structure check:
    python lint_local.py --structure-only

    # Use a remote model instead:
    python lint_local.py --model gpt-4o-mini --api-key sk-...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

import llm
import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LANGUAGETOOL_URL = "https://api.languagetool.org/v2"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_IGNORE_PATTERNS = [r"^Merge\s", r"^Revert\s"]

# ---------------------------------------------------------------------------
# Data model (shared with lint_commits.py)
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
# Git helpers
# ---------------------------------------------------------------------------


def git_log(revision_range: str | None = None, last_n: int | None = None) -> list[dict]:
    """
    Return commits from the local git repo.

    Each dict has keys: "sha" and "message".
    """
    cmd = ["git", "log", "--format=%H%x00%B%x00"]
    if revision_range:
        cmd.append(revision_range)
    elif last_n:
        cmd.extend(["-n", str(last_n)])
    else:
        cmd.extend(["-n", "5"])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = result.stdout.strip()
    if not raw:
        return []

    commits = []
    # Split on the null-byte record separator.  Each record is "sha\0message\0"
    entries = raw.split("\0")
    # entries come in pairs: [sha, message, sha, message, ...] with a trailing empty
    i = 0
    while i + 1 < len(entries):
        sha = entries[i].strip()
        message = entries[i + 1].strip()
        if sha and message:
            commits.append({"sha": sha, "message": message})
        i += 2

    return commits


# ---------------------------------------------------------------------------
# LanguageTool grammar check
# ---------------------------------------------------------------------------


def check_grammar(text: str, lt_url: str, language: str) -> list[dict]:
    """Return a list of grammar issues from LanguageTool."""
    resp = requests.post(
        f"{lt_url}/check",
        data={
            "text": text,
            "language": language,
            "enabledOnly": "false",
        },
        timeout=30,
    )
    resp.raise_for_status()
    matches = resp.json().get("matches", [])
    issues = []
    for m in matches:
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
# LLM structure check
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
3. Is the subject ≤ 72 characters?
4. Is the subject separated from the body by a blank line (if a body exists)?
5. Does the message avoid vague wording like "fix stuff", "updates", "misc changes"?

Return ONLY the JSON object, no markdown fences.

Commit message:
\"\"\"
{message}
\"\"\"
"""


def _try_load_mlx_model(model_name: str) -> llm.Model:
    """
    Attempt to load a model.  If it looks like an MLX community model,
    try the llm-mlx MlxModel class directly (avoids needing to pre-register
    the model with `llm mlx download-model`).
    """
    # First try the standard llm registry (works for registered / API models)
    try:
        return llm.get_model(model_name)
    except llm.UnknownModelError:
        pass

    # Fall back to direct MlxModel instantiation
    try:
        from llm_mlx import MlxModel
        return MlxModel(model_name)
    except ImportError:
        raise RuntimeError(
            f"Model '{model_name}' is not registered with llm and the llm-mlx "
            "plugin is not installed.  Install it with: pip install llm-mlx"
        )


def check_structure(message: str, model_name: str, api_key: str | None = None) -> list[str]:
    """Use an LLM to evaluate commit message structure."""
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        os.environ.setdefault("ANTHROPIC_API_KEY", api_key)

    prompt = STRUCTURE_PROMPT.format(message=message)

    try:
        model = _try_load_mlx_model(model_name)
        response = model.prompt(prompt)
        text = response.text().strip()
        # Strip markdown fences if the model wraps them
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        data = json.loads(text)
        return data.get("issues", [])
    except Exception as exc:
        print(f"⚠️  LLM structure check failed: {exc}", file=sys.stderr)
        return [f"LLM analysis error: {exc}"]


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

# ANSI colours
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_report(results: list[CommitIssue]) -> None:
    """Pretty-print results to the terminal."""
    issues_found = any(r.has_issues for r in results)

    if not issues_found:
        print(f"\n{GREEN}{BOLD}✅ All commits look good!{RESET}\n")
        return

    print(f"\n{RED}{BOLD}🧹 Git Hygiene — Issues Found{RESET}\n")

    for r in results:
        if not r.has_issues:
            continue
        short_sha = r.sha[:8]
        subject = r.message.split("\n", 1)[0][:72]
        print(f"{CYAN}{BOLD}{short_sha}{RESET} — {subject}")

        if r.grammar_issues:
            print(f"  {YELLOW}Grammar:{RESET}")
            for gi in r.grammar_issues:
                suggestion = ""
                if gi["replacements"]:
                    suggestion = f' → try: {", ".join(gi["replacements"])}'
                print(f"    • {gi['message']}{suggestion}  ({gi['rule']})")

        if r.structure_issues:
            print(f"  {YELLOW}Structure:{RESET}")
            for si in r.structure_issues:
                print(f"    • {si}")

        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lint git commit messages locally for grammar and structure quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --model mlx-community/Llama-3.2-3B-Instruct-4bit
  %(prog)s --range main..HEAD --grammar-only
  %(prog)s --last 10 --model gpt-4o-mini --api-key sk-...
""",
    )
    # Commit selection
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--range",
        metavar="REV_RANGE",
        help="Git revision range (e.g. main..HEAD, HEAD~5..HEAD)",
    )
    g.add_argument(
        "--last",
        type=int,
        default=5,
        metavar="N",
        help="Number of recent commits to lint (default: 5)",
    )

    # Check selection
    p.add_argument("--grammar-only", action="store_true", help="Only run the grammar check")
    p.add_argument("--structure-only", action="store_true", help="Only run the LLM structure check")

    # LLM
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    p.add_argument("--api-key", help="API key for remote LLM providers (not needed for local MLX models)")

    # LanguageTool
    p.add_argument(
        "--languagetool-url",
        default=DEFAULT_LANGUAGETOOL_URL,
        help=f"LanguageTool API base URL (default: {DEFAULT_LANGUAGETOOL_URL})",
    )
    p.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code for LanguageTool (default: {DEFAULT_LANGUAGE})",
    )

    # Ignore patterns
    p.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        metavar="REGEX",
        help="Regex patterns for commit subjects to skip (repeatable). "
        "Defaults to skipping Merge and Revert commits.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ignore_patterns = [
        re.compile(p)
        for p in (args.ignore_pattern if args.ignore_pattern else DEFAULT_IGNORE_PATTERNS)
    ]

    # Fetch commits
    if args.range:
        print(f"🔍 Fetching commits in range {args.range} …")
        commits = git_log(revision_range=args.range)
    else:
        print(f"🔍 Fetching last {args.last} commit(s) …")
        commits = git_log(last_n=args.last)

    print(f"   Found {len(commits)} commit(s).\n")

    if not commits:
        print("No commits to lint.")
        return 0

    results: list[CommitIssue] = []

    for commit in commits:
        sha = commit["sha"]
        message = commit["message"]
        subject = message.split("\n", 1)[0]

        if any(p.search(subject) for p in ignore_patterns):
            print(f"   ⏭️  {sha[:8]} — skipped (matches ignore pattern)")
            continue

        print(f"   🔎 {sha[:8]} — {subject[:60]}")

        ci = CommitIssue(sha=sha, message=message)

        # Grammar check
        if not args.structure_only:
            try:
                ci.grammar_issues = check_grammar(message, args.languagetool_url, args.language)
            except Exception as exc:
                print(f"      ⚠️  Grammar check failed: {exc}", file=sys.stderr)

        # Structure check
        if not args.grammar_only:
            ci.structure_issues = check_structure(message, args.model, args.api_key)

        results.append(ci)

    print_report(results)

    has_issues = any(r.has_issues for r in results)
    return 1 if has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
