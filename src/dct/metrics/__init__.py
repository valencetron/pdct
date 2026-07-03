"""dct.metrics — CLI for prelim PDCT metrics (tokens, utility, ablation).

Invocation:
  python -m dct.metrics tokens   [--days N]
  python -m dct.metrics utility  [--days N]
  python -m dct.metrics ablation [--days N]

Reads logs from $PDCT_LOGS_DIR (default: <repo>/logs):
  - measurement.jsonl  (per-turn canonical row, schema_version=2;
                        v2 adds graph_nodes/graph_edges — build 85)
  - utility.jsonl      (kind=turn rows + kind=followup rows)
"""
