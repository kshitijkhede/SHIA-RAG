# Pushing this project to GitHub

Your `Rag/` folder is already a git repository. Follow these steps in a terminal
opened **inside the `Rag/` folder** (`cd ~/Desktop/Rag`).

## 0. One-time: make sure git knows who you are
```bash
git config user.name  "Kshitij Khede"
git config user.email "<your-github-email>"
```

## 1. Keep copyrighted papers OUT of the public repo
The updated `.gitignore` already excludes `research_papers/` and `New Paper/`.
If those folders were committed earlier, untrack them (files stay on your disk):
```bash
git rm -r --cached research_papers "New Paper" 2>/dev/null || true
```

## 2. Safety check — make sure no API key is committed
```bash
git grep -nEi "AIza|AQ\.|sk-[A-Za-z0-9]|api[_-]?key\s*=" -- . ':!*.md' || echo "clean"
```
If anything shows up, remove/replace it before pushing.

## 3. Stage and commit
```bash
git add -A
git commit -m "SHIA-RAG: major project code, benchmarks, and mid-semester report"
```

## 4a. Create the repo and push  (if you have the GitHub CLI `gh`)
```bash
gh auth status || gh auth login          # if not already logged in
gh repo create kshitijkhede/SHIA-RAG --public --source=. --remote=origin --push
```

## 4b. OR push manually (no `gh`)
1. Create an empty repo on github.com named **SHIA-RAG** (do NOT add a README there).
2. Then:
```bash
git branch -M main
git remote add origin https://github.com/kshitijkhede/SHIA-RAG.git
git push -u origin main
```
(When prompted for a password, use a **Personal Access Token**, not your account password.)

## Notes
- Use `--public` or `--private` in step 4a depending on what you want.
- If you *do* own the rights to the papers and want them included, delete the
  `research_papers/` and `New Paper/` lines from `.gitignore` before step 3.
