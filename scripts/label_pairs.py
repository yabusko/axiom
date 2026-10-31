"""Blind human-labeling CLI for `eval/enrichment_pairs.csv` (B3, stage-2 fixes).

Shows only `entity`/`kind`/`candidate` -- never `cos`/`margin` -- so the
annotator judges each (entity, candidate) pair on its merits, not on the
model's own confidence score. Rows are walked in a fixed (seeded) shuffle,
not the file's cosine-sorted order, for the same reason. `cos`/`margin` stay
in the CSV throughout (needed afterward to calibrate against the labels);
only the terminal display hides them.

Plain stdlib (csv + argparse + input()) -- no new dependency, matching
scripts/label_cli.py's existing scaffold discipline.

Keys: 1 = candidate is the same concept as entity. 0 = it isn't.
? = unsure (excluded from calibration). Saves the whole file after every
row, so a Ctrl-C mid-session loses nothing and a rerun resumes exactly
where it left off (already-labeled rows are skipped).

Subcommands (see --help):
    status                          progress summary, no stdin (testable)
    label --annotator ID            interactive walkthrough (human-only)
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "eval" / "enrichment_pairs.csv"
FIELDNAMES = ["paper_id", "entity", "kind", "evidence_span", "candidate", "cos", "margin",
              "label", "annotator"]
VALID_LABELS = {"0", "1", "?"}
SHUFFLE_SEED = 1234  # fixed, not re-randomized per run -- same order every resume


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------
def load_pairs(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("annotator", "")
    return rows


def save_pairs(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Pure logic (testable without stdin)
# ---------------------------------------------------------------------------
def shuffled_order(n: int, seed: int = SHUFFLE_SEED) -> list[int]:
    """A fixed permutation of range(n) -- same order on every run, so a
    resumed session walks unlabeled rows in the same sequence as the first."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def is_labeled(row: dict) -> bool:
    return (row.get("label") or "").strip() in VALID_LABELS


def remaining_order(rows: list[dict], seed: int = SHUFFLE_SEED) -> list[int]:
    """Shuffled row indices that still need a label -- the resume logic."""
    return [i for i in shuffled_order(len(rows), seed) if not is_labeled(rows[i])]


def progress_summary(rows: list[dict]) -> str:
    labeled = [r for r in rows if is_labeled(r)]
    counts = {"1": 0, "0": 0, "?": 0}
    for r in labeled:
        counts[r["label"].strip()] += 1
    return (f"{len(labeled)}/{len(rows)} labeled "
            f"(1={counts['1']}, 0={counts['0']}, ?={counts['?']}); "
            f"{len(rows) - len(labeled)} remaining")


# ---------------------------------------------------------------------------
# Interactive labeling (human-only; not exercised by tests)
# ---------------------------------------------------------------------------
def label_loop(rows: list[dict], path: Path, annotator: str) -> None:  # pragma: no cover
    order = remaining_order(rows)
    total_remaining = len(order)
    if total_remaining == 0:
        print(f"Nothing left to label. {progress_summary(rows)}")
        return

    print(f"{total_remaining} pairs left to label. 1=match, 0=no match, ?=unsure. Ctrl-C to pause.\n")
    for n, idx in enumerate(order, start=1):
        row = rows[idx]
        print(f"[{n}/{total_remaining}] entity={row['entity']!r}  kind={row['kind']}  "
              f"candidate={row['candidate']!r}")
        while True:
            raw = input("  1 / 0 / ? > ").strip()
            if raw in VALID_LABELS:
                break
            print("  please enter 1, 0, or ?")
        row["label"] = raw
        row["annotator"] = annotator
        save_pairs(path, rows)

    print(f"\nDone. {progress_summary(rows)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=str, default=str(DEFAULT_PATH),
                     help="path to enrichment_pairs.csv (default: eval/enrichment_pairs.csv)")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="progress summary, no interaction")

    label = sub.add_parser("label", help="interactive blind labeling walkthrough")
    label.add_argument("--annotator", type=str, default="yahor")

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.file)
    rows = load_pairs(path)

    if args.command == "status":
        print(progress_summary(rows))
        return 0

    if args.command == "label":  # pragma: no cover
        label_loop(rows, path, args.annotator)
        return 0

    return 1  # pragma: no cover -- argparse's `required=True` makes this unreachable


if __name__ == "__main__":
    raise SystemExit(main())
