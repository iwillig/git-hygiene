# Git Hygiene

A GitHub Action that lints git commit messages on pull requests for:

- **Grammar issues** — via the [LanguageTool](https://languagetool.org/) API
- **Structure quality** — via an LLM (checks that commits explain *why*, not just *what*)

When issues are found the action **posts a PR comment** summarising them and **fails the check**.

## Quick Start

### Option A: Local model — no API key needed (recommended)

Run a small LLM directly on the GitHub Actions runner. Zero cost, fully
private — no data ever leaves the runner.

```yaml
name: Git Hygiene

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write

jobs:
  lint-commits:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: shortcut/git-hygiene@main
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          use-local-model: "true"
          llm-model: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

This installs [mlx-lm](https://github.com/ml-explore/mlx-lm) (CPU backend)
and the [llm-mlx](https://github.com/simonw/llm-mlx) plugin, downloads the
model (~278 MB), and runs inference on the runner. MLX supports Linux via its
CPU backend, so this works on standard `ubuntu-latest` GitHub Actions runners.

> **Performance note:** CPU inference on a 2-core runner takes roughly 30–60 s
> per commit with the 0.5B model. For PRs with many commits, consider a remote
> model or a [larger runner](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners).

### Option B: Remote API model

Uses a cloud LLM provider — faster, but requires an API key secret.

```yaml
jobs:
  lint-commits:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: shortcut/git-hygiene@main
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          llm-model: "gpt-4o-mini"
          llm-api-key: ${{ secrets.OPENAI_API_KEY }}
```

## Inputs

| Input                   | Required | Default                              | Description                                                             |
| ----------------------- | -------- | ------------------------------------ | ----------------------------------------------------------------------- |
| `github-token`          | yes      | `${{ github.token }}`                | GitHub token (needs `pull-requests: write` for comments)                |
| `use-local-model`       | no       | `false`                              | Run the LLM locally on the runner via mlx-lm (no API key needed)       |
| `llm-model`             | no       | `gpt-4o-mini`                        | Model name — any [llm](https://llm.datasette.io/) model or MLX HF repo |
| `llm-api-key`           | no*      | —                                    | API key for a remote LLM provider (*required unless `use-local-model`)  |
| `languagetool-url`      | no       | `https://api.languagetool.org/v2`    | LanguageTool API base URL (point to self-hosted if desired)             |
| `languagetool-language` | no       | `en-US`                              | Language code for grammar checking                                     |
| `ignore-patterns`       | no       | `^Merge\s` / `^Revert\s`            | Newline-separated regexes — matching commit subjects are skipped       |
| `fail-on-error`         | no       | `true`                               | Set to `false` to post a comment without failing the check             |

### MLX Models for `use-local-model`

Any [mlx-community](https://huggingface.co/mlx-community) model works. Recommended for CI:

| Model | Size | Speed (2-core runner) | Notes |
|---|---|---|---|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 278 MB | ~30–60 s/commit | **Best for CI** — smallest, fastest |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 1.8 GB | ~2–5 min/commit | Better quality, needs more RAM |

## How It Works

![Architecture diagram showing the Git Hygiene sequence flow](docs/architecture.png)

<details>
<summary>Diagram source (PlantUML)</summary>

The diagram above is generated from [`docs/architecture.puml`](docs/architecture.puml).
Regenerate it with:

```bash
plantuml -tpng docs/architecture.puml -o ../docs/
```
</details>

### Step by step:

1. A developer opens or updates a pull request.
2. GitHub triggers the composite action (`action.yml`) on the runner.
3. The action sets up Python, installs dependencies (and `mlx-lm[cpu]` + `llm-mlx` if `use-local-model` is enabled).
4. `lint_commits.py` fetches all PR commits via the GitHub API.
5. For each commit (skipping those matching ignore patterns):
   - **Grammar check** — sends the message to the LanguageTool API and collects spelling/grammar matches.
   - **Structure check** — loads a model (local MLX or remote API) and prompts it to evaluate subject line, body, and overall quality. The response is parsed as JSON.
6. Results are aggregated into a Markdown report.
7. The report is posted (or updated) as a PR comment.
8. The action exits with code 1 (failing the check) if any issues were found and `fail-on-error` is true.

## Using a Different Remote LLM Provider

The action uses Simon Willison's [llm](https://llm.datasette.io/) library which supports many providers:

**OpenAI:**
```yaml
llm-model: "gpt-4o-mini"
llm-api-key: ${{ secrets.OPENAI_API_KEY }}
```

**Anthropic** (install the plugin in a prior step):
```yaml
- run: pip install llm-anthropic
- uses: shortcut/git-hygiene@main
  with:
    llm-model: "claude-3-haiku-20240307"
    llm-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Self-Hosted LanguageTool

Run LanguageTool on your own infrastructure and point the action at it:

```yaml
languagetool-url: "https://lt.internal.example.com/v2"
```

## Local Development (Apple Silicon)

You can run git-hygiene locally against your current git repo using MLX with
full GPU acceleration on Apple Silicon — no API keys required.

### Setup

```bash
# Install dependencies (includes llm-mlx and mlx-lm)
pipenv install --dev

# Download a small MLX model (~278 MB)
pipenv run llm mlx download-model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

### Run

```bash
# Lint the last 5 commits with the default MLX model
pipenv run python lint_local.py

# Lint commits on a branch compared to main
pipenv run python lint_local.py --range main..HEAD

# Lint the last 10 commits
pipenv run python lint_local.py --last 10

# Use a specific MLX model
pipenv run python lint_local.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Only grammar check (no LLM)
pipenv run python lint_local.py --grammar-only

# Only structure check (no LanguageTool)
pipenv run python lint_local.py --structure-only

# Use a remote model instead (e.g. OpenAI)
pipenv run python lint_local.py --model gpt-4o-mini --api-key sk-...
```

### CLI Options

| Flag | Description |
|---|---|
| `--range REV_RANGE` | Git revision range (e.g. `main..HEAD`) |
| `--last N` | Number of recent commits to lint (default: 5) |
| `--model MODEL` | LLM model name (default: `mlx-community/Llama-3.2-3B-Instruct-4bit`) |
| `--api-key KEY` | API key for remote providers (not needed for local MLX models) |
| `--grammar-only` | Only run the grammar check |
| `--structure-only` | Only run the LLM structure check |
| `--languagetool-url URL` | Custom LanguageTool API URL |
| `--language CODE` | Language for grammar checking (default: `en-US`) |
| `--ignore-pattern REGEX` | Regex for subjects to skip (repeatable) |

### Recommended Models (Local)

| Model | Size | Notes |
|---|---|---|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 278 MB | Fastest, good for quick checks |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 1.8 GB | Default, good balance |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 4 GB | Best quality |

## Development

```bash
# Install dependencies
pipenv install --dev

# Run unit tests (mocked, fast)
pipenv run pytest tests/ -v

# Run integration tests with a real local MLX model (requires Apple Silicon)
pipenv run pytest tests/test_integration_mlx.py -v --run-mlx

# Use a specific model for integration tests
MLX_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    pipenv run pytest tests/test_integration_mlx.py -v --run-mlx
```

## License

MIT
