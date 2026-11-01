"""Human-vs-AI (or human-vs-human) label agreement via Cohen's kappa (E3).

Reads two independently-produced label sets for the same items and reports
how well they agree -- e.g. a human reviewer's pass over
`eval/ndcg_queries_v2.json` against the original AI-drafted labels (save a
copy of the file before running `scripts/label_cli.py review`, then compare
the before/after copies with this script), or two different annotators'
label files. Pure Python + `sklearn.metrics.cohen_kappa_score` (already a
pinned dependency, requirements.txt) -- no LLM call, no new dependency.

Each input file is either:
  - the same shape `scripts/label_cli.py`/`scripts/eval_ndcg.py` already use
    (`{"queries": [{"query": "...", "judgments": {paper_id: relevance}, ...},
    ...]}`) -- the natural choice when comparing two snapshots of
    `ndcg_queries_v2.json`. Items are keyed by `"<query text>::<paper_id>"`
    (not list position) so the two files can list queries in a different
    order and still line up correctly.
  - a flat JSON object (`{"<item_id>": <label>, ...}`), for label sets that
    aren't tied to the nDCG query schema.
  - a 2-column CSV with a header row (`id,label`) -- same flat shape, for
    labels kept in a spreadsheet rather than JSON.

Only items present in BOTH files are compared (an item judged in just one of
the two tells you nothing about agreement); the report says how many were
dropped for that reason so a wildly small overlap is visible, not silent.

Usage (from repo root):
    python scripts/label_agreement.py eval/ndcg_queries_v2.json.before eval/ndcg_queries_v2.json.after
    python scripts/label_agreement.py annotator_a.csv annotator_b.csv --out eval/label_agreement.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# Loading -- JSON (ndcg_queries_v2 shape or flat) / CSV
# ---------------------------------------------------------------------------
def _flatten_ndcg_v2(data: dict) -> dict[str, int]:
    """`{"queries": [{"query": ..., "judgments": {paper_id: relevance}}, ...]}`
    -> `{"<query>::<paper_id>": relevance}`. Keying by query TEXT (not list
    index) so two files can list queries in a different order."""
    flat: dict[str, int] = {}
    for q in data.get("queries", []):
        query_text = q.get("query", "")
        for paper_id, relevance in q.get("judgments", {}).items():
            flat[f"{query_text}::{paper_id}"] = int(relevance)
    return flat


def _load_json_labels(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "queries" in data:
        return _flatten_ndcg_v2(data)
    if isinstance(data, dict):
        return {str(k): int(v) for k, v in data.items()}
    raise ValueError(f"{path}: expected a JSON object (flat {{id: label}} or "
                      f"{{'queries': [...]}}), got {type(data).__name__}")


def _load_csv_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return labels
    header = [c.strip().lower() for c in rows[0]]
    body = rows[1:] if header in (["id", "label"], ["item_id", "label"]) else rows
    for row in body:
        if not row or len(row) < 2:
            continue
        item_id, label = row[0].strip(), row[1].strip()
        if not item_id:
            continue
        labels[item_id] = int(label)
    return labels


def load_labels(path: Path) -> dict[str, int]:
    """Dispatch by extension: `.csv` -> 2-column `id,label` (header
    optional); anything else -> JSON, either the nDCG v2 query/judgments
    shape or a flat `{id: label}` object."""
    if path.suffix.lower() == ".csv":
        return _load_csv_labels(path)
    return _load_json_labels(path)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
def compute_agreement(labels_a: dict[str, int], labels_b: dict[str, int]) -> dict:
    """Cohen's kappa over the items judged in BOTH label sets.

    `n_common` is the comparison's actual sample size; `only_in_a`/`only_in_b`
    count items dropped because only one side judged them (reported, not
    silently discarded, so a thin overlap -- e.g. a reviewer who only
    touched a handful of items -- is visible in the output). `kappa` is
    `float('nan')` (sklearn's own result, not an error here) if there's no
    label variance to measure agreement over, e.g. fewer than 2 common items
    or every common item got the same label on both sides.
    """
    common = sorted(set(labels_a) & set(labels_b))
    only_a = sorted(set(labels_a) - set(labels_b))
    only_b = sorted(set(labels_b) - set(labels_a))

    a_values = [labels_a[k] for k in common]
    b_values = [labels_b[k] for k in common]

    agreement_rate = (
        sum(1 for x, y in zip(a_values, b_values) if x == y) / len(common)
        if common else 0.0
    )
    kappa = cohen_kappa_score(a_values, b_values) if len(common) >= 2 else float("nan")

    return {
        "n_common": len(common),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "agreement_rate": agreement_rate,
        "kappa": kappa,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("label_file_a", type=str)
    ap.add_argument("label_file_b", type=str)
    ap.add_argument("--out", type=str, default=None,
                     help="write the report as JSON to this path (optional)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path_a, path_b = Path(args.label_file_a), Path(args.label_file_b)

    labels_a = load_labels(path_a)
    labels_b = load_labels(path_b)
    report = compute_agreement(labels_a, labels_b)

    print(f"n_common={report['n_common']}  only_in_a={report['n_only_a']}  "
          f"only_in_b={report['n_only_b']}")
    print(f"agreement_rate={report['agreement_rate']:.3f}  kappa={report['kappa']:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[report] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
