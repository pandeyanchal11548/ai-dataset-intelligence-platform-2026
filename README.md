# AI Dataset Intelligence Platform

A platform for uploading, profiling, and understanding datasets (CSV/Excel),
with a Python/FastAPI backend and a web frontend.

## Project structure

```
ai-dataset-intelligence-platform/
├── backend/
│   ├── app/
│   │   ├── loaders/          # Dataset loading & profiling (done)
│   │   │   ├── dataset_loader.py
│   │   │   ├── schema.py
│   │   │   └── exceptions.py
│   │   ├── api/               # FastAPI routes (next)
│   │   ├── core/              # Config, settings (next)
│   │   ├── models/            # Pydantic / DB models (next)
│   │   └── utils/
│   ├── tests/
│   │   └── test_dataset_loader.py
│   └── requirements.txt
├── frontend/                  # UI (React/Vite scaffold to be added)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── styles/
│   └── public/
├── data/
│   ├── raw/                   # Uploaded/original files (gitignored)
│   ├── processed/             # Cleaned outputs (gitignored)
│   └── samples/                # Small example files safe to commit
├── docs/
├── notebooks/                  # Exploration notebooks
├── scripts/
│   └── setup_env.sh
└── .gitignore
```

## 1. Git

Repository is initialized. Suggested first commit:

```bash
git add .
git commit -m "chore: scaffold project structure, env setup, dataset loader"
```

## 2. Environment setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

This creates a `.venv`, upgrades pip, and installs everything in
`backend/requirements.txt` (pandas, openpyxl/xlrd, FastAPI, pytest, etc.).

## 3. Dataset Loader module

`backend/app/loaders/dataset_loader.py` exposes `DatasetLoader`:

```python
from app.loaders.dataset_loader import DatasetLoader

loader = DatasetLoader()
df, profile = loader.load("data/samples/sample_sales.csv")

print(profile.to_dict())
```

It performs:

- **File validation** — extension whitelist (`.csv`, `.tsv`, `.txt`, `.xlsx`,
  `.xls`, `.xlsm`), existence, non-empty, size-limit, and corruption/parsing
  error handling.
- **Column detection** — fixes blank/`Unnamed:` headers, de-duplicates
  column names, drops fully-empty rows/columns, and records every fix as a
  human-readable warning in the returned profile.
- **Type inference** — classifies each column as `integer`, `float`,
  `boolean`, `datetime`, `categorical`, or `text`, independent of the raw
  pandas dtype, plus null counts/percentages, unique counts, and sample
  values — ready to drive frontend column-type badges.

Run the tests (after `pip install -r backend/requirements.txt`):

```bash
cd backend
pytest tests/ -v
```

## Roadmap (next steps)

- [ ] FastAPI upload endpoint (`POST /datasets/upload`) wrapping `DatasetLoader`
- [ ] Frontend scaffold (React + Vite/Tailwind) with an upload + profile view
- [ ] Persistent storage layer for uploaded datasets
- [ ] Basic dataset statistics / visualization endpoints
