#!/usr/bin/env python3
"""
Git Hygiene — local CLI runner.

Lint commit messages from the current git repository without GitHub.
Uses the llm library for LLM inference (Ollama via llm-ollama, OpenAI built-in, etc.).

Usage:
    # Lint the last 5 commits using a local Ollama model:
    python lint_local.py --model qwen2.5:0.5b

    # Lint commits on a branch compared to main:
    python lint_local.py --range main..HEAD

    # Lint a specific number of recent commits:
    python lint_local.py --last 10

    # Skip the LLM check and only do grammar:
    python lint_local.py --grammar-only

    # Skip grammar and only do LLM structure check:
    python lint_local.py --structure-only

    # Use a remote model instead (e.g. OpenAI):
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
DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_IGNORE_PATTERNS = [r"^Merge\s", r"^Revert\s"]

# Built-in dictionary of common dev/tool names that LanguageTool flags as
# spelling mistakes.  Users can extend this via the --custom-words flag.
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
    entries = raw.split("\0")
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


def _extract_flagged_word(match: dict, text: str) -> str:
    """Extract the word that LanguageTool flagged from the original text."""
    offset = match.get("offset", 0)
    length = match.get("length", 0)
    if offset >= 0 and length > 0 and offset + length <= len(text):
        return text[offset:offset + length]
    return ""


def check_grammar(
    text: str, lt_url: str, language: str, custom_words: set[str] | None = None
) -> list[dict]:
    """Return a list of grammar issues from LanguageTool.

    Matches where the flagged word appears in *custom_words* (case-insensitive)
    are silently dropped.
    """
    if custom_words is None:
        custom_words = {w.lower() for w in BUILTIN_WORDS}

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


def _parse_llm_json(text: str) -> dict:
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
        lines = text.split('\n')
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
        text = '\n'.join(fixed_lines)
        return json.loads(text)


def check_structure(message: str, model: llm.Model | None = None, model_id: str = "", api_key: str = "") -> dict:
    """Use an LLM to evaluate whether a commit message explains *why*.

    Pass either a pre-loaded *model* or a *model_id* (+ optional *api_key*)
    to load one on the fly.

    Returns a dict with keys: explains_why, score, feedback, suggestion.
    On failure returns a dict with feedback containing the error.
    """
    if model is None:
        model = get_llm_model(model_id or DEFAULT_MODEL, api_key)

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
        print(f"\n{GREEN}{BOLD}All commits look good!{RESET}\n")
        return

    print(f"\n{RED}{BOLD}Git Hygiene — Issues Found{RESET}\n")

    for r in results:
        if not r.has_issues:
            continue
        short_sha = r.sha[:8]
        subject = r.message.split("\n", 1)[0][:72]
        print(f"{CYAN}{BOLD}{short_sha}{RESET} -- {subject}")

        if r.grammar_issues:
            print(f"  {YELLOW}Grammar:{RESET}")
            for gi in r.grammar_issues:
                suggestion = ""
                if gi["replacements"]:
                    suggestion = f' -> try: {", ".join(gi["replacements"])}'
                print(f"    - {gi['message']}{suggestion}  ({gi['rule']})")

        if r.structure_issues:
            print(f"  {YELLOW}Feedback:{RESET}")
            for si in r.structure_issues:
                print(f"    - {si}")
            if r.score is not None:
                print(f"  Score: {r.score}/10")
            if r.suggestion:
                print(f"  {YELLOW}Suggested rewrite:{RESET}")
                print(f"    {r.suggestion}")

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
  %(prog)s --model qwen2.5:0.5b
  %(prog)s --range main..HEAD --grammar-only
  %(prog)s --last 10 --model gpt-4o-mini --api-base https://api.openai.com/v1 --api-key sk-...
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
    p.add_argument(
        "--enable-grammar",
        action="store_true",
        help="Enable the LanguageTool grammar checker (disabled by default)",
    )
    p.add_argument("--grammar-only", action="store_true", help="Only run the grammar check (implies --enable-grammar)")
    p.add_argument("--structure-only", action="store_true", help="Only run the LLM structure check")

    # LLM (via the llm library -- model discovery handled by llm + plugins)
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name as known to the llm library (default: {DEFAULT_MODEL}). "
        "Ollama models require the llm-ollama plugin. "
        "Run 'llm models' to see available models.",
    )
    p.add_argument("--api-key", default="", help="API key for remote LLM providers")

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

    # Custom dictionary
    p.add_argument(
        "--custom-word",
        action="append",
        default=None,
        metavar="WORD",
        help="Extra words to add to the spell-check dictionary (repeatable). "
        "A built-in list of common dev/tool names is always included.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ignore_patterns = [
        re.compile(p)
        for p in (args.ignore_pattern if args.ignore_pattern else DEFAULT_IGNORE_PATTERNS)
    ]

    # Build custom words set
    custom_words = {w.lower() for w in BUILTIN_WORDS}
    if args.custom_word:
        for w in args.custom_word:
            custom_words.add(w.lower())

    # Load LLM model once (skip if grammar-only)
    model = None
    if not args.grammar_only:
        print(f"Loading model: {args.model}")
        model = get_llm_model(args.model, args.api_key)
        print(f"   Model loaded: {model.model_id}")

    # Fetch commits
    if args.range:
        print(f"Fetching commits in range {args.range} ...")
        commits = git_log(revision_range=args.range)
    else:
        print(f"Fetching last {args.last} commit(s) ...")
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
            print(f"   SKIP {sha[:8]} -- skipped (matches ignore pattern)")
            continue

        print(f"   CHECK {sha[:8]} -- {subject[:60]}")

        ci = CommitIssue(sha=sha, message=message)

        # Grammar check (disabled by default; --enable-grammar or --grammar-only to turn on)
        run_grammar = args.enable_grammar or args.grammar_only
        if run_grammar and not args.structure_only:
            try:
                ci.grammar_issues = check_grammar(
                    message, args.languagetool_url, args.language, custom_words
                )
            except Exception as exc:
                print(f"      WARNING: Grammar check failed: {exc}", file=sys.stderr)

        # Structure check
        if not args.grammar_only:
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

    print_report(results)

    has_issues = any(r.has_issues for r in results)
    return 1 if has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
