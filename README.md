# cited-docs-agent

Ask questions about your documents and get answers where **every sentence has a source**.
Built with LangGraph, Qdrant and Claude.

> Work in progress. Stage 1 of 3 is done: documents are split into chunks, embedded and
> stored in Qdrant, and you can search them by meaning.

## How it works (so far)

```
documents (.md, .txt, .pdf)
   │  chunking.py: split into pieces, one section per piece, keep file + section + page
   ▼
chunks ──► embedding model (BGE-small, runs locally) ──► vectors
                                                            │
                                                            ▼
                                       Qdrant vector database (local folder)
                                                            ▲
question ──► same embedding model ──► vector ───────────────┘  closest chunks, with scores
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m docsagent ingest sample_docs
python -m docsagent search "How many leave days can I carry over?"
python -m pytest
```

The sample documents in `sample_docs/` describe a fictional company, Northwind Labs.

## Roadmap

- [x] Stage 1: chunking, local embeddings, Qdrant storage, semantic search
- [ ] Stage 2: LangGraph agent (retrieve, answer with citations, remove unsupported sentences)
- [ ] Stage 3: evaluation set and score, CI, Docker
