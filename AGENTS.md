# Agent workflow

## Git

- Never commit to `main`. `main` tracks `origin/main` only via explicit merge/PR by the user.
- Every task gets its own branch: `feat/<short-name>`, created from current `origin/main` unless the task continues an existing feature branch.
- Do not mix unrelated tasks on one branch.
- Push the feature branch; do not push `main` unless the user explicitly asks to update it.
