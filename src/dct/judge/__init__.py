"""PDCT-utility judge subsystem (P1.3a substrate).

This package is the foundation layer:
- SQLite schema + migrations
- Atomic queue (enqueue, claim, commit)
- Daemon adapter (build_judge_payload)
- Stub codex worker scaffold

What is NOT in P1.3a:
- Era detection (deferred to plan v4 with multi-scale support)
- Live codex invocation
- Gold-set CI (era-dependent)
- pdct_report.py wiring
"""
__all__: list[str] = []
