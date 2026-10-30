"""Human review/override CLI for `eval/ndcg_queries_v2.json` (R9).

Plain stdlib (argparse + input()) -- no new dependency, matching the repo's
existing scaffold discipline (OD17: `scripts/` + markdown/json reports, no
notebooks, no new pins without a docs/DECISIONS.md entry).

The v2 file's judgments are AI-drafted (pooled top-10 across every retrieval
mode, graded by the `fast` tier -- see that file's `_provenance`). This CLI
is how a human reviews/overrides individual labels; every override is
recorded in that judgment's `judgments_meta` entry as
`{"provenance": "human-reviewed", "annotator": <id>, "note": <str|None>}`,
never by silently overwriting the AI-drafted provenance tag.

Subcommands (see --help):
    list                                           one line per query
    show --query IDX                                full judgment table for one query
    set --query IDX --paper ID --relevance N
        --annotator ID [--note TEXT]                override/add one judgment
    review --annotator ID                           interactive walkthrough (human-only)

`list`/`show`/`set` are plain argparse -- fully offline-testable, no stdin.
`review` is the actual "a person reviews labels" workflow (input() loop);
it is intentionally never exercised by the automated test suite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom import config


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------
def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Pure logic (testable without stdin)
# ---------------------------------------------------------------------------
def query_summary_lines(data: dict) -> list[str]:
    lines = []
    for i, q in enumerate(data.get("queries", [])):
        n = len(q.get("judgments", {}))
        pooled = ",".join(q.get("pooled_from", [])) or "?"
        lines.append(f"[{i}] ({n} judgments, pooled_from={pooled}) {q['query']!r}")
    return lines


def query_detail_lines(data: dict, query_idx: int) -> list[str]:
    q = _get_query(data, query_idx)
    lines = [f"[{query_idx}] {q['query']!r}"]
    meta = q.get("judgments_meta", {})
    for paper_id, relevance in q.get("judgments", {}).items():
        m = meta.get(paper_id, {})
        prov = m.get("provenance", "AI-drafted")
        annotator = m.get("annotator")
        tag = f"{prov}" + (f" by {annotator}" if annotator else "")
        lines.append(f"    {paper_id}: {relevance}  [{tag}]")
    return lines


def _get_query(data: dict, query_idx: int) -> dict:
    queries = data.get("queries", [])
    if not (0 <= query_idx < len(queries)):
        raise ValueError(f"query index {query_idx} out of range (0..{len(queries) - 1})")
    return queries[query_idx]


def set_judgment(data: dict, *, query_idx: int, paper_id: str, relevance: int,
                  annotator: str, note: str | None = None,
                  scale_max: int = config.EVAL_V2_SCALE_MAX) -> None:
    """Override (or add) one (query, paper) judgment, recording the annotator.

    Mutates `data` in place. Raises ValueError on an out-of-range query index,
    a non-integer/out-of-scale relevance, or a missing/blank annotator id --
    callers (the CLI, tests) decide how to surface that.
    """
    if not annotator or not annotator.strip():
        raise ValueError("annotator id is required for a human override")
    if not isinstance(relevance, int) or not (0 <= relevance <= scale_max):
        raise ValueError(f"relevance must be an int in 0..{scale_max}, got {relevance!r}")
    q = _get_query(data, query_idx)
    q.setdefault("judgments", {})[paper_id] = relevance
    q.setdefault("judgments_meta", {})[paper_id] = {
        "provenance": "human-reviewed",
        "annotator": annotator,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Interactive review (human-only; not exercised by tests)
# ---------------------------------------------------------------------------
def review_loop(data: dict, path: Path, annotator: str) -> None:  # pragma: no cover
    """Walk every query/judgment, prompting for a relevance override.

    Enter (blank input) keeps the current value unchanged (no override
    recorded); typing a new int in range records a human override and saves
    immediately, so progress survives a Ctrl-C mid-review.
    """
    scale_max = data.get("_scale_max", config.EVAL_V2_SCALE_MAX)
    for qi, q in enumerate(data.get("queries", [])):
        print(f"\n[{qi}] {q['query']!r}")
        for paper_id, relevance in list(q.get("judgments", {}).items()):
            raw = input(f"    {paper_id} (current={relevance}, AI-drafted; "
                        f"0..{scale_max} or Enter to keep): ").strip()
            if not raw:
                continue
            try:
                new_val = int(raw)
                note = input("    note (optional): ").strip() or None
                set_judgment(data, query_idx=qi, paper_id=paper_id, relevance=new_val,
                             annotator=annotator, note=note, scale_max=scale_max)
                save(path, data)
                print(f"    -> saved {paper_id}={new_val} (annotator={annotator})")
            except ValueError as exc:
                print(f"    skipped: {exc}")
    print(f"\nReview complete. {path} saved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=str, default=str(config.EVAL_V2_QUERIES_PATH),
                     help="path to ndcg_queries_v2.json (default: config.EVAL_V2_QUERIES_PATH)")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list all queries with judgment counts")

    show = sub.add_parser("show", help="show one query's full judgment table")
    show.add_argument("--query", type=int, required=True)

    setp = sub.add_parser("set", help="override/add one judgment")
    setp.add_argument("--query", type=int, required=True)
    setp.add_argument("--paper", type=str, required=True)
    setp.add_argument("--relevance", type=int, required=True)
    setp.add_argument("--annotator", type=str, required=True)
    setp.add_argument("--note", type=str, default=None)

    review = sub.add_parser("review", help="interactive walkthrough (human-only)")
    review.add_argument("--annotator", type=str, required=True)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.file)
    data = load(path)

    if args.command == "list":
        for line in query_summary_lines(data):
            print(line)
        return 0

    if args.command == "show":
        try:
            for line in query_detail_lines(data, args.query):
                print(line)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "set":
        try:
            set_judgment(data, query_idx=args.query, paper_id=args.paper,
                         relevance=args.relevance, annotator=args.annotator,
                         note=args.note, scale_max=data.get("_scale_max", config.EVAL_V2_SCALE_MAX))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        save(path, data)
        print(f"saved {args.paper}={args.relevance} for query {args.query} "
              f"(annotator={args.annotator}) -> {path}")
        return 0

    if args.command == "review":  # pragma: no cover
        review_loop(data, path, args.annotator)
        return 0

    return 1  # pragma: no cover -- argparse's `required=True` makes this unreachable


if __name__ == "__main__":
    raise SystemExit(main())
