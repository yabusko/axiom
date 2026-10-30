"""SPECTER2 abstract encoder.

Committed team decision: allenai/specter2_base, 768-dim. If no GPU is available
we fall back to CPU and emit a clear warning — we never silently switch models.

Encoding recipe is the canonical SPECTER one: concatenate title and abstract
with the tokenizer's [SEP] token and take the [CLS] token of the last hidden
state as the document embedding.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from axiom import config


def _pick_device() -> str:
    """CUDA > Apple MPS > CPU. Warn (don't switch models) when no GPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    warnings.warn(
        "No GPU detected — running SPECTER2 (allenai/specter2_base) on CPU. "
        "Encoding will be slower but the model is unchanged.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"


class Specter2Encoder:
    """Lazy-loaded SPECTER2 encoder. Construct once, reuse for all encode calls."""

    def __init__(self, model_id: str = config.MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or _pick_device()
        load_from = self._resolve_local_snapshot(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        self.model = AutoModel.from_pretrained(load_from).to(self.device)
        self.model.eval()
        # Decide how queries are embedded (see _resolve_query_path). Documents
        # are ALWAYS encoded with the plain base model via encode(), so existing
        # corpus points stay valid and never need re-encoding for this change.
        self.query_path = self._resolve_query_path()
        print(f"[embed] query path: {self.query_path}")

    @staticmethod
    def _resolve_local_snapshot(model_id: str) -> str:
        """Local cache dir if already downloaded, else the bare repo id.

        `from_pretrained(repo_id, local_files_only=True)` still calls
        `has_file()` internally, which hits the network regardless of
        `local_files_only` (a known gap in this pinned transformers version)
        -- on a flaky/proxied connection that raises before ever reaching the
        cache. Passing a literal local directory instead skips hub resolution
        entirely. `snapshot_download(local_files_only=True)` does respect the
        flag and raises cleanly (no network call) when nothing is cached.
        """
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=model_id, local_files_only=True)
        except Exception:
            return model_id

    def _resolve_query_path(self) -> str:
        """Prefer the SPECTER2 ad-hoc query adapter; fall back to expansion.

        The adapter is the proper fix for query/document asymmetry, but the
        `adapters` library pins a newer transformers than our committed
        transformers==4.40.2. We attempt the import so the code is adapter-ready,
        but under the current pins this fails and we use query expansion — which
        needs no new dependency. See config.QUERY_EXPANSION_TEMPLATE.
        """
        try:
            import adapters  # noqa: F401  (only present if pins were bumped)
        except Exception:
            return "expansion"
        # TODO(P2): with `adapters` installed (pin bump), load
        # allenai/specter2_adhoc_query for queries + allenai/specter2 (proximity)
        # for documents, and re-encode the corpus so both sides align.
        return "adapter"

    @torch.no_grad()
    def encode(
        self,
        titles: Sequence[str],
        abstracts: Sequence[str] | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> list[list[float]]:
        """Encode papers into 768-dim vectors.

        titles/abstracts are parallel sequences. If abstracts is None, titles
        are treated as standalone text (used for query encoding).
        """
        if abstracts is None:
            texts = list(titles)
        else:
            sep = self.tokenizer.sep_token or "[SEP]"
            texts = [f"{t or ''}{sep}{a or ''}" for t, a in zip(titles, abstracts)]

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            output = self.model(**inputs)
            # [CLS] token = first position of last hidden state.
            cls = output.last_hidden_state[:, 0, :]
            vectors.extend(cls.cpu().tolist())
        return vectors

    def encode_query(self, query: str) -> list[float]:
        """Encode a single free-text query into one 768-dim vector.

        Improves on naively embedding the raw query string (which treats a terse
        query as a document). Under the "expansion" path we wrap the query in an
        abstract-shaped sentence so it lands nearer the document region; under
        the "adapter" path (if ever enabled) the raw query is used as-is.
        """
        if self.query_path == "expansion":
            text = config.QUERY_EXPANSION_TEMPLATE.format(query=query)
        else:
            text = query
        return self.encode([text], abstracts=None)[0]
