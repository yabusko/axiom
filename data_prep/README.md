# data_prep — Corpus building & cleaning

Exploratory and productionized steps that turn the raw `paperlists/` archives
into a clean, model-ready corpus.

| File               | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `flow.ipynb`       | Exploratory notebook: crawl `paperlists/`, profile text artifacts, derive the cleaning rules. |
| `clean_dataset.py` | Productionized cleaner encoding those rules (dedupe, drop admin rows, normalize text). |

## Usage

Run from the **repo root**, writing into `data/` (the whole `data/` directory
is git-ignored):

```bash
python data_prep/clean_dataset.py --input data/FULL_DATASET.jsonl --output data/cleaned_dataset.jsonl
```

## Data files (not committed)

`data/FULL_DATASET.jsonl` and `data/cleaned_dataset.jsonl` (~78 MB each,
23,002 real papers — ACL/EMNLP/COLING/NAACL 2020-2025, see `docs/DECISIONS.md`)
are git-ignored and regenerated rather than committed. They contain author
emails and affiliations scraped from the source paperlists — never publish
them outside this machine.

If you don't have the raw `paperlists/` archives, pull the files directly from
the team's old `main` history instead of regenerating from scratch:

```bash
git show origin/main:FULL_DATASET.jsonl > data/FULL_DATASET.jsonl
git show origin/main:cleaned_dataset.jsonl > data/cleaned_dataset.jsonl
```

`axiom/config.EVAL_DATASET_PATH` and `scripts/ingest_eval_corpus.py` read
`data/FULL_DATASET.jsonl` (the **uncleaned** file) by default — see OD31 in
`docs/DECISIONS.md` for why the eval corpus still builds from that one, not
`cleaned_dataset.jsonl`.
