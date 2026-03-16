# Push To GitHub

From the repository root:

```bash
git remote add origin git@github.com:<YOUR_GITHUB_USER>/conditional_drifting_models.git
git push -u origin main
```

If you prefer HTTPS:

```bash
git remote add origin https://github.com/<YOUR_GITHUB_USER>/conditional_drifting_models.git
git push -u origin main
```

Current local commit:

```text
8ecee1588a823611b241788f35080254ee3ca92d
```

If you want the multi-seed GPU benchmark right after cloning:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_publication_benchmark.py --device cuda:0
```
