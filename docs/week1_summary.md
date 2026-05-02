# Week 1 Summary — Foundations

## Goal

Build a basic engineering workflow for the AI Engineering journey:

- shell usage
- Git and GitHub workflow
- repository policies
- vector similarity fundamentals
- debugging and profiling habits

## Completed Work

### W1D1 — Shell and Toolbox

Created a small toolbox structure:

- `toolbox/scripts/find_todos.sh`
- `toolbox/notes/cli-cheatsheet.md`

Learned:

- relative vs absolute paths
- grep/find/cat basics
- executable shell scripts
- basic Git commit workflow

### W1D2 — Branch and Pull Request Workflow

Practiced professional Git workflow:

- create feature branch
- commit focused change
- push branch
- open pull request
- squash merge
- delete feature branch

### W1D3 — Repository Policies

Added repository standards:

- `.github/CONTRIBUTING.md`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.githooks/commit-msg`

The commit hook validates Conventional Commit messages.

### W1D4 — Vector Similarity

Implemented pure Python vector utilities:

- `dot`
- `norm`
- `cosine_similarity`

Also added a simple word-count based text similarity demo.

Key idea:

```text
text -> vector -> cosine similarity -> similarity score
```

This is a simplified version of the retrieval logic used in RAG systems.

### W1D5 — Debugging and Profiling

Added:

- logging
- handled error examples
- simple benchmarking
- cProfile usage

Key idea:

```text
You cannot improve what you do not measure.
```

## Main Artifacts

- `toolbox/scripts/find_todos.sh`
- `toolbox/scripts/run_week1_checks.sh`
- `toolbox/notes/cli-cheatsheet.md`
- `labs/lin_alg/vec.py`
- `labs/lin_alg/text_similarity.py`
- `labs/lin_alg/debug_examples.py`
- `.github/CONTRIBUTING.md`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.githooks/commit-msg`

## How to Run Week 1 Checks

```bash
./toolbox/scripts/run_week1_checks.sh
```

## Reflection

This week established the foundation for working like an engineer:

- write small scripts
- commit cleanly
- use feature branches
- open pull requests
- document artifacts
- add basic tests/checks
- log and profile code behavior

The most important mindset shift:

```text
Learning is not complete until it produces a visible artifact.
```
