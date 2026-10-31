"""Calibrate the B1 embedding-match threshold/margin from labeled pairs (B3, stage-2 fixes).

Reads `eval/enrichment_pairs.csv` after `scripts/ai_label_enrichment_pairs.py` has filled
in its `label` column (agreed AI labels only -- disagreements and "unsure" rows have an
empty `label` and are excluded here automatically), grid-sweeps (tau, delta) over the
observed cos/margin range, and picks the point with the highest recall among those meeting
a precision floor (OD23's own rule: "highest recall at precision >= 0.9").

Also reports how many of a given run's `new_concept` entities (the ones that fell through
the cascade without a match) would flip to a real embedding match under the chosen
threshold -- recomputed fresh against the real ontology/phrase-encoder, not estimated,
since that's the number that actually matters for deciding whether to act on this.

Usage (from repo root):
    python scripts/calibrate_enrichment_threshold.py --min-precision 0.9
    python scripts/calibrate_enrichment_threshold.py --min-precision 0.9 --run-id <run_id>
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config, db

PAIRS_PATH = Path(__file__).resolve().parent.parent / "eval" / "enrichment_pairs.csv"
REPORT_PATH = Path(__file__).resolve().parent.parent / "eval" / "enrichment_threshold_calibration.md"


def load_labeled_rows(path: Path = PAIRS_PATH) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("label") in ("0", "1")]


def precision_recall(rows: list[dict], tau: float, delta: float) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        is_match = r["label"] == "1"
        predicted = float(r["cos"]) >= tau and float(r["margin"]) >= delta
        if predicted and is_match:
            tp += 1
        elif predicted and not is_match:
            fp += 1
        elif not predicted and is_match:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tau": round(tau, 4), "delta": round(delta, 4), "precision": precision,
            "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sweep(rows: list[dict], *, tau_step: float = 0.01, delta_step: float = 0.01,
          tau_range: tuple[float, float] = (0.70, 1.00),
          delta_range: tuple[float, float] = (0.00, 0.30)) -> list[dict]:
    points = []
    tau = tau_range[0]
    while tau <= tau_range[1] + 1e-9:
        delta = delta_range[0]
        while delta <= delta_range[1] + 1e-9:
            points.append(precision_recall(rows, tau, delta))
            delta += delta_step
        tau += tau_step
    return points


def best_operating_point(points: list[dict], min_precision: float) -> dict:
    """Highest recall among points meeting the precision floor; if none clear the floor,
    the single highest-precision point available (reported honestly as not meeting the
    target, not silently substituted)."""
    pool = [p for p in points if p["precision"] >= min_precision]
    if not pool:
        return max(points, key=lambda p: (p["precision"], p["recall"]))
    return max(pool, key=lambda p: (p["recall"], p["precision"]))


async def count_flips(run_id: str, tau: float, delta: float) -> dict:
    """Recompute cos/margin for every distinct new_concept entity in `run_id` against the
    real ontology/phrase-encoder and count how many would newly clear (tau, delta)."""
    import numpy as np
    from axiom.agents import enrichment

    conn = db.connect(config.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT DISTINCT name, kind FROM paper_concepts WHERE run_id = ? AND cosine = -1.0",
            (run_id,),
        ).fetchall()
        encoder = enrichment.PhraseEncoder()
        ontology = enrichment.build_ontology(conn, encoder)
    finally:
        conn.close()

    async def top1_top2(name: str) -> tuple[float, float]:
        raw = await asyncio.to_thread(encoder.encode, [name])
        vec = enrichment._normalize_rows(np.array(raw, dtype=float))
        sims = (vec @ ontology.vectors.T)[0]
        order = np.argsort(sims)[::-1]
        top1 = max(-1.0, min(1.0, float(sims[int(order[0])])))
        top2 = max(-1.0, min(1.0, float(sims[int(order[1])]))) if len(order) > 1 else -1.0
        return top1, top1 - top2

    results = await asyncio.gather(*[top1_top2(r["name"]) for r in rows])
    flips = sum(1 for cos, margin in results if cos >= tau and margin >= delta)
    return {"total_new_concept": len(rows), "flips": flips}


def write_report(path: Path, *, chosen: dict, min_precision: float, n_labeled: int,
                  flips: dict | None) -> None:
    lines = [
        "# Enrichment embedding-match threshold calibration (B3, stage-2 fixes)",
        "",
        f"- Source labels: `eval/enrichment_pairs.csv`, AI-labeled (sonnet-5-5 + opus-5-5 "
        "agreement; disagreements and unsure excluded) -- no human review.",
        f"- Labeled pairs used: {n_labeled}",
        f"- Rule: highest recall at precision >= {min_precision}",
        f"- Chosen: tau={chosen['tau']}, delta={chosen['delta']} "
        f"(precision={chosen['precision']:.3f}, recall={chosen['recall']:.3f}, "
        f"tp={chosen['tp']} fp={chosen['fp']} fn={chosen['fn']} tn={chosen['tn']})",
    ]
    if flips is not None:
        lines.append(
            f"- Of {flips['total_new_concept']} `new_concept` entities in the reference run, "
            f"{flips['flips']} would flip to a real embedding match under this threshold."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-precision", type=float, default=0.9)
    ap.add_argument("--run-id", type=str, default=None,
                     help="run_id to recompute new_concept flip counts against (optional)")
    ap.add_argument("--file", type=str, default=str(PAIRS_PATH))
    args = ap.parse_args(argv)

    rows = load_labeled_rows(Path(args.file))
    if len(rows) < 2:
        print(f"[calibrate] only {len(rows)} usable labeled pair(s) -- not enough to calibrate.")
        return 1

    points = sweep(rows)
    chosen = best_operating_point(points, args.min_precision)
    print(f"[calibrate] {len(rows)} labeled pairs")
    print(f"[calibrate] chosen: tau={chosen['tau']} delta={chosen['delta']} "
          f"precision={chosen['precision']:.3f} recall={chosen['recall']:.3f} "
          f"(tp={chosen['tp']} fp={chosen['fp']} fn={chosen['fn']} tn={chosen['tn']})")

    flips = None
    if args.run_id:
        flips = asyncio.run(count_flips(args.run_id, chosen["tau"], chosen["delta"]))
        print(f"[calibrate] {flips['flips']}/{flips['total_new_concept']} new_concept "
              f"entities would flip to a real match under this threshold")

    write_report(REPORT_PATH, chosen=chosen, min_precision=args.min_precision,
                 n_labeled=len(rows), flips=flips)
    print(f"[calibrate] wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
