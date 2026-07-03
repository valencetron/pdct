---
date: 2026-01-05
concepts: [vector-database-selection, embedding-storage, pgvector, retrieval-infrastructure]
tags: [distillation, example]
gist: Ada and the assistant compared vector databases and chose pgvector for the project
---
# Choosing a Vector Database

Ada asked which vector store to use for the recipe-search side project. The assistant compared pgvector, a hosted service, and a local FAISS index. Decision: pgvector — the project already runs Postgres, the corpus is small (under 100k rows), and operational simplicity beats raw ANN speed at this scale.

## Key Points
- pgvector chosen for operational simplicity; corpus under 100k embeddings
- Hosted vector services rejected: cost and data-residency concerns
- FAISS rejected: no persistence story without extra plumbing
- Revisit if corpus exceeds 1M rows or p95 query latency exceeds 100ms
