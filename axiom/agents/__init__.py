"""Agentic extraction pipeline (OD19-OD23, docs/agentic/ARCHITECTURE.md).

Router-worker agents: coordinator -> extraction -> enrichment -> linking,
with a shared Pydantic v2 contract layer (contracts.py) and a deterministic
verifier (verifier.py) driving a bounded self-correction loop. Additive to
the rest of `axiom/*` -- see ARCHITECTURE.md for the module map and data flow.
"""
