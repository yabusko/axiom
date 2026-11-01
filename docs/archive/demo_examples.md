# Demo Examples (Task 8.3)

Concrete, reproducible outputs from the **actual built system**, run against
the demo corpus (`scripts/bootstrap_synthetic.py`'s 30-paper synthetic NLP
corpus, 2021–2025). Every number below came from a real run on
2026-07-04 — none of it is illustrative/fabricated. Reproduce any of it with
the commands shown.

> **What this is not:** a demo of PBI 5's hypothesis pitch (3-step Dead-End/
> Fertile-Frontier evaluation with badges) — that pipeline isn't built (see
> `docs/DECISIONS.md` OD9/OD14). What follows is what actually runs today:
> search, trends (OD10), research gaps (OD9), reading-list summaries (OD14),
> and the retrieval eval (OD11).

---

## 1. Search

```bash
streamlit run app/streamlit_app.py   # 🔍 Search tab
```

Query: `parameter efficient fine-tuning with low-rank adapters`

| Score | Title | Year · Venue |
|---|---|---|
| 0.033 | QLoRA: Efficient Finetuning of Quantized Language Models | 2023 · NeurIPS |
| 0.032 | LoRA: Low-Rank Adaptation of Large Language Models | 2021 · ICLR |
| 0.031 | Composable Adapters for Continual Instruction Tuning | 2025 · EMNLP |

(Hybrid dense+sparse retrieval, RRF-fused — see `docs/DECISIONS.md` for why
these scores are small: they're rank-agreement scores, not cosine similarity.)

---

## 2. Trending concepts (OD10 velocity engine)

```bash
streamlit run app/streamlit_app.py   # 📈 Trending tab
# or: python -c "from axiom import db, velocity; ..."
```

Corpus split: **prior window 2021–2023** (17 papers) → **recent window
2024–2025** (13 papers).

**Top risers** (velocity `+9.59`, all `0 → 1` papers — the demo corpus is
small enough that most "risers" are single-paper concepts; the *ranking
mechanism* is what's being demonstrated, not statistical significance at this
scale):
- Knowledge graph, Search, Continual learning, Uncertainty estimation, Probing

**Top fading** (velocity `-9.20` to `-10.20`):
- Text embeddings, Prompting, In-context learning, Citation analysis, Cross-lingual transfer

> **Honest caveat:** at 30 papers, per-concept counts are tiny (mostly 0-2),
> so every ranked keyword is flagged `low_confidence` (`recent_count < 5`) —
> exactly the confidence warning the backlog's acceptance criteria call for.
> The mechanism is real; the statistical power isn't, until the corpus scales
> up (T2.3).

---

## 3. Research gaps (OD9) — 2 examples

```bash
streamlit run app/streamlit_app.py   # 🕸️ Citation graph tab → Research gaps view
```

The corpus resolves into 8 Louvain communities. Top 2 candidate gaps by
`gap_score = centroid_cosine / (1 + inter_community_citations)`:

**Gap #1** — score 0.956
> **Parameter-efficient fine-tuning · Low-rank adaptation · Instruction tuning · Alignment** (7 papers, e.g. `LoRA`, `Direct Preference Optimization`)
> ⟷ **Low-resource NLP · Data augmentation · NER · Cross-lingual transfer** (4 papers, e.g. `Cross-Lingual Transfer for Low-Resource NER`)
>
> Semantic similarity **0.956**, **0** inter-community citations. Reading: PEFT
> methods and low-resource/multilingual NLP are topically close (both about
> adapting large models efficiently) but this corpus shows zero papers citing
> across that boundary — a candidate opening (e.g. "parameter-efficient
> adaptation specifically for low-resource languages").

**Gap #2** — score 0.951
> **Retrieval-augmented generation · Question answering · Hallucination · Information retrieval** (6 papers)
> ⟷ **Dense retrieval · Text embeddings · Scientific NLP · Contrastive learning** (4 papers, e.g. `Scientific Document Representations from Citation-Aware Pretraining`)
>
> Semantic similarity **0.951**, **0** inter-community citations. Reading: RAG
> work and the dense-retrieval/embeddings literature it's built on don't cite
> each other in this corpus — surprising, since RAG *depends on* retrieval
> quality; a candidate direction is explicitly connecting RAG failure modes to
> retriever/embedding choices.

> **Honest caveat (already logged as a known risk):** every candidate pair in
> this corpus has **0 inter-citations** and centroid cosines cluster tightly
> (0.94–0.96) — the expected signature of a single-topic, 30-paper corpus, not
> a validated finding. These are framed as **candidates**, never proven gaps.
> Discrimination sharpens with a denser/multi-topic corpus (T2.3).

---

## 4. Reading list + local-LLM summary (OD13/OD14)

```bash
streamlit run app/streamlit_app.py   # 📚 Reading list tab
# bookmark a paper from Search, then click "🧠 Summarize"
```

Real output for *"Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks"* (summarized locally via Ollama, `qwen2.5:7b`, no API key/cost):

- Combines pretrained generator with dense retriever for NLP tasks. _(cites `S0001`)_
- Attends to retrieved documents during decoding for grounded generation. _(cites `S0001`)_
- Outperforms parametric baselines on open-domain QA and fact verification. _(cites `S0001`)_

Every bullet paraphrases a claim actually present in the abstract — no
invented numbers or outside knowledge (enforced by the summarizer's system
prompt, `axiom/summarize.py`).

---

## 5. Keyword canonicalization (T3.2/OD14)

```bash
python scripts/canonicalize_concepts.py
```

Ad-hoc verification (real Ollama call, before the full corpus run):

```
Input:  LoRA, Low-Rank Adaptation, Machine translation
Output: {"groups": [
    {"canonical": "LoRA", "members": ["LoRA", "Low-Rank Adaptation"]},
    {"canonical": "Machine translation", "members": ["Machine translation"]}
]}
```

Correctly merges the true synonym pair and correctly leaves the unrelated
concept alone. Running it over the actual 30-paper demo corpus's 51 distinct
concepts produced **zero merges** — a true negative: this hand-authored
corpus has no duplicate concept labels to begin with. The merge behavior
itself is verified above; it activates automatically once a real corpus has
true duplicates (e.g. real OpenAlex data tagging both "LoRA" and "Low-Rank
Adaptation" on different papers about the same method).

---

## 6. Retrieval quality (nDCG@10, OD11)

```bash
python scripts/ingest_eval_corpus.py    # one-time: samples 1,500 real ACL/EMNLP/COLING/NAACL papers
python scripts/eval_ndcg.py --report eval/report.md
```

**Mean nDCG@10: 0.453** (hybrid) vs **0.345** (dense-only) — hybrid retrieval
measurably wins, full breakdown in `eval/report.md`.

> **Honest caveat:** below the backlog's ≥0.65 target, and the relevance
> judgments (`eval/ndcg_queries.json`) are single-pass and AI-drafted, not the
> "2 annotators" the acceptance criteria specify. Treat this as a working demo
> signal, not a validated quality claim — see the file's own `_provenance`
> field.

---

## What's demonstrably *not* here

- **PBI 5** — no hypothesis pitch, no Dead-End/Fertile-Frontier badges, no
  HITL review queue. The gap detector above (§3) is corpus-level
  (community-pairs), not a per-hypothesis evaluator.
- **React frontend** — everything above runs in Streamlit; FastAPI
  (`api/main.py`) exposes the same capabilities over REST but has no UI
  consumer yet.
- **Calibrated thresholds** — `gap_score` and `velocity` rankings are
  uncalibrated; no labeled established/dead/hot topic set exists (T8.1).
- **Human-rated gap quality** — the 1-5 rating study needs real human raters
  (T8.2), not something this package can substitute.
