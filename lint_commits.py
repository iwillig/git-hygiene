#!/usr/bin/env python3
"""
Git Hygiene — lint commit messages for grammar and structure quality.

Grammar:      LanguageTool API
Structure:    LLM library (any provider, including local MLX models)
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

import llm
import requests

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["REPO"]  # owner/repo
PR_NUMBER = os.environ["PR_NUMBER"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
USE_LOCAL_MODEL = os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true"
LANGUAGETOOL_URL = os.environ.get("LANGUAGETOOL_URL", "https://api.languagetool.org/v2")
LANGUAGETOOL_LANGUAGE = os.environ.get("LANGUAGETOOL_LANGUAGE", "en-US")
IGNORE_PATTERNS_RAW = os.environ.get("IGNORE_PATTERNS", "")
FAIL_ON_ERROR = os.environ.get("FAIL_ON_ERROR", "true").lower() == "true"

IGNORE_PATTERNS: list[re.Pattern] = []
for line in IGNORE_PATTERNS_RAW.strip().splitlines():
    line = line.strip()
    if line:
        IGNORE_PATTERNS.append(re.compile(line))

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


def check_grammar(text: str) -> list[dict]:
    """Return a list of grammar issues from LanguageTool."""
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
# LLM model loading
# ---------------------------------------------------------------------------


def _load_model(model_name: str) -> llm.Model:
    """
    Load an LLM model.

    1. Try the standard llm plugin registry (OpenAI, Anthropic, etc.)
    2. If that fails and we're in local mode, try direct MlxModel instantiation
       via the llm-mlx plugin (no pre-registration needed).
    """
    try:
        return llm.get_model(model_name)
    except llm.UnknownModelError:
        pass

    # Fall back to direct MlxModel instantiation for local MLX models
    try:
        from llm_mlx import MlxModel
        return MlxModel(model_name)
    except ImportError:
        raise RuntimeError(
            f"Model '{model_name}' is not registered with llm and the llm-mlx "
            "plugin is not installed.  Either set use-local-model: true (which "
            "installs llm-mlx automatically), or use a registered model name."
        )


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


def check_structure(message: str) -> list[str]:
    """Use an LLM to evaluate commit message structure. Returns a list of issue strings."""
    # Configure API key for remote models.  The `llm` library reads keys
    # from its own key store, but in CI we inject them via env vars.
    if LLM_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", LLM_API_KEY)
        os.environ.setdefault("ANTHROPIC_API_KEY", LLM_API_KEY)

    prompt = STRUCTURE_PROMPT.format(message=message)

    try:
        model = _load_model(LLM_MODEL)
        response = model.prompt(prompt)
        text = response.text().strip()
        # Strip markdown fences if the model wraps them anyway
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
# Markdown report
# ---------------------------------------------------------------------------


def build_report(results: list[CommitIssue]) -> str:
    """Build a Markdown summary suitable for a PR comment."""
    issues_found = any(r.has_issues for r in results)

    lines: list[str] = []
    if issues_found:
        lines.append("## 🧹 Git Hygiene — Issues Found\n")
    else:
        lines.append("## ✅ Git Hygiene — All Commits Look Good!\n")
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
                    suggestion = f' → try: *{", ".join(gi["replacements"])}*'
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
        print(f"🧠 Using local MLX model: {LLM_MODEL}")
    else:
        print(f"🌐 Using remote model: {LLM_MODEL}")

    print(f"🔍 Fetching commits for PR #{PR_NUMBER} in {REPO} …")
    commits = fetch_pr_commits()
    print(f"   Found {len(commits)} commit(s).")

    results: list[CommitIssue] = []

    for commit in commits:
        sha = commit["sha"]
        message: str = commit["commit"]["message"]
        subject = message.split("\n", 1)[0]

        # Check ignore patterns
        if any(p.search(subject) for p in IGNORE_PATTERNS):
            print(f"   ⏭️  {sha[:8]} — skipped (matches ignore pattern)")
            continue

        print(f"   🔎 {sha[:8]} — {subject[:60]}")

        ci = CommitIssue(sha=sha, message=message)

        # Grammar check
        try:
            ci.grammar_issues = check_grammar(message)
        except Exception as exc:
            print(f"      ⚠️  Grammar check failed: {exc}", file=sys.stderr)

        # Structure check
        ci.structure_issues = check_structure(message)

        results.append(ci)

    # Build report
    report = build_report(results)
    print("\n" + report)

    # Post PR comment
    try:
        post_pr_comment(report)
        print("\n💬 Posted PR comment.")
    except Exception as exc:
        print(f"\n⚠️  Failed to post PR comment: {exc}", file=sys.stderr)

    # Determine exit code
    has_issues = any(r.has_issues for r in results)
    if has_issues and FAIL_ON_ERROR:
        print("\n❌ Issues found — failing the check.")
        return 1

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
