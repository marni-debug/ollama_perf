# Agent workflow

## Git

- Never commit to `main`. Do not merge into `main` unless the user explicitly says so.
- New task: create `feat/<short-name>` from current `origin/main` (or continue the existing feature branch for that task).
- One task per branch. Do not mix unrelated work.
- After the task is done: push the feature branch and open a Pull Request into `main`. The user decides whether to merge.
- Do not push `main`.
