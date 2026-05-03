# AI Engineering Journey

Hi, I'm Alperen. This repository documents my journey to become a production-oriented AI Engineer.

## Goal

Build practical skills in:

- Python engineering
- Machine learning
- Deep learning
- RAG systems
- LLM applications
- AI agents
- MLOps
- Cloud deployment
- Evaluation and monitoring

## Learning Format

Each day follows this structure:

1. Learn the concept
2. Build a small artifact
3. Commit the work
4. Document progress

## Current Progress

### Month 1 — Foundations

- Shell and Git basics
- Python engineering
- Linear algebra basics
- Pandas, SQL, and EDA
- Titanic mini ML project

## Repository Philosophy

This is not a passive course repo.
Every folder should contain proof of work: scripts, notebooks, tests, reports, demos, or deployment artifacts.

## Local Git Hooks

This repository includes Git hooks under `.githooks`.

After cloning the repository, enable hooks with:

```bash
git config core.hooksPath .githooks
```

The current hook validates commit messages using the Conventional Commits format:

```text
type(scope): short description
```

Example:

```text
docs(repo): add contributing guide
```

## Current Artifacts

- `toolbox/`: shell scripts and CLI notes
- `labs/lin_alg/vec.py`: pure Python vector operations and cosine similarity
- `labs/lin_alg/text_similarity.py`: word-count based text similarity demo
- `labs/lin_alg/debug_examples.py`: handled debugging examples

## Weekly Summaries

- [Week 1 — Foundations](docs/week1_summary.md)

## Useful Commands

Run Week 1 checks:

```bash
./toolbox/scripts/run_week1_checks.sh
```

### Type Checking

- `pyproject.toml` configures mypy in strict mode.
- `examples/w2d2_type_check_demo.py` demonstrates typed DailyLog usage.

### Type Checking

- `pyproject.toml` configures mypy in strict mode.
- `examples/w2d2_type_check_demo.py` demonstrates typed DailyLog usage.
