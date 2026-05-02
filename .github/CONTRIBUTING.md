# Contributing Guide

This repository documents my AI Engineering learning journey.

## Branch Naming

Use short and descriptive branch names:

- feature/w1d2-git-workflow
- fix/w1d2-close-markdown-codeblock
- docs/readme-update
- chore/repo-policy

## Commit Message Format

Use Conventional Commits:

type(scope): short description

Examples:

- feat(rag): add document chunker
- fix(api): handle empty query
- docs(readme): update setup guide
- test(vector): add cosine similarity tests
- chore(repo): add commit hook

Allowed types:

- feat
- fix
- docs
- style
- refactor
- test
- chore
- ci
- build
- perf

## Pull Request Rules

Before opening a PR:

1. Run relevant scripts or tests.
2. Check `git diff`.
3. Keep changes small and focused.
4. Write a clear PR title.
5. Explain what changed and why.

## Main Branch Rule

Do not commit directly to `main` for feature work.

Use:

1. Create a feature branch.
2. Commit changes.
3. Push branch.
4. Open PR.
5. Merge after review.
