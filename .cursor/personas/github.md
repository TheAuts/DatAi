---
description: Dual-save GitHub sync from the local DatAi checkout
alwaysApply: true
---

# GitHub dual-save

Source of truth is this machine checkout, not a cloud clone:

- Path: `C:\Users\andre\Desktop\Stinky\DatAi`
- Remote: `origin` → `https://github.com/TheAuts/DatAi.git`
- Working branch: `main`

Rules:

- Sync first: `git fetch origin` and `git status` before edits. Prefer a clean tree.
- Stay on `main` unless already on a branch Andrew asked for. Do not open `cursor/*` branches or pull requests unless `git push` to the current branch is rejected.
- Atomic commits: one logical change per commit. Do not mix feature work with workflow or bugfix.
- Dual-save: after `git commit`, GitHub must match local. Push immediately with `git push -u origin HEAD`. A `post-commit` hook also pushes; if it fails, push manually before finishing.
- Never leave the branch ahead of `origin`. After each save, `HEAD` and `origin/<branch>` should be the same commit.
- On conflict, auth failure, or rejected push: stop and report. Do not force-push unless Andrew asks.
