# cited-docs-agent

[![CI](https://github.com/kartikmalhotra/cited-docs-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmalhotra/cited-docs-agent/actions/workflows/ci.yml)

Ask questions about your documents and get answers where **every sentence has a source**.
Sentences that can't be backed by a source are removed before you see them.

**Stack:** LangGraph (agent workflow) · Qdrant (vector database) · Claude (LLM) ·
BGE-small embeddings and an MS MARCO cross-encoder reranker (both run locally) · FastAPI · Docker · GitHub Actions

```
$ python -m docsagent ask "How many leave days can I carry over, and until when?"
Answer: You can carry over up to 5 unused days of annual leave into the next calendar year [1].
Carried-over days must be used by 31 March, or they expire [1].

Sources:
  [1] leave-policy.md › Carry-over  (match 0.81)

Steps: retrieve → generate → check
```
*(Illustrative output. Run it yourself to see real scores.)*

## How it works

```mermaid
flowchart LR
    Q([question]) --> R[retrieve<br/>Qdrant search → injection screen → rerank]
    R -->|good match| G[generate<br/>Claude answers, citing n after every sentence]
    R -->|weak match, first time| W[rewrite<br/>turn the question into a search query]
    W --> R
    G --> C[check<br/>remove unsupported sentences]
    C --> A([answer + sources])
```

1. **Chunking** (`docsagent/chunking.py`): documents (.md, .txt, .pdf) are split into pieces,
   one section per piece, each remembering its file, section and page.
2. **Embedding and search** (`docsagent/vectorstore.py`): a small open-source model (BGE-small,
   runs locally, no API key) turns each piece into a 384-number vector; Qdrant stores them and
   returns the pieces closest in meaning to a question (cosine similarity).
3. **Reranking** (`docsagent/rerank.py`): vector search fetches 8 candidates; a cross-encoder,
   which reads the question and each chunk together, picks the best 4.
4. **Prompt-injection screen** (`docsagent/security.py`): retrieved chunks that look like
   instructions to an AI ("ignore previous instructions…", fake `SYSTEM:` lines, requests to
   reveal the prompt) are removed before the model sees them, and the answer says which source
   was ignored and why. The system prompt also tells the model that sources are data, never
   instructions, and the model has no tools while answering.
5. **Agent** (`docsagent/agent.py`): a LangGraph state graph. If the best match is weak it
   rewrites the question once and searches again; then Claude answers from the numbered pieces
   at temperature 0.
6. **Source check** (`docsagent/citations.py`), the output guardrail against hallucination. A sentence is removed if
   - it cites nothing,
   - it cites a source that doesn't exist, or
   - it contains a number (5 days, ₹2,500, 31 March) that isn't in the source it cites.

   If nothing survives, the answer is "I couldn't find this in the documents."

## Evaluation

`evals/questions.jsonl` has 18 questions about the sample documents: 15 with known answers
and 3 the documents can't answer. Each run scores four things:

| Check | Question it answers |
|---|---|
| retrieval | Did search find the right section? |
| answer | Does the final answer contain the key facts? |
| citation | Does the answer cite the right section? |
| refusal | For unanswerable questions, did it say so instead of guessing? |

```bash
python -m docsagent eval --retrieval-only   # free, no API key
python -m docsagent eval                    # full agent, uses Claude
```

CI scores retrieval on every push, with and without the reranker, and shows both reports on
the run page. Besides "right section in the top 4", it reports **top-1**: was the right
section ranked first?

**Measured so far:** retrieval found the right section for **15 of 15** answerable questions
(BGE-small embeddings + Qdrant, measured in CI). Full-agent scores (answer, citation, refusal)
come from `python -m docsagent eval` with an API key.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m docsagent ingest sample_docs
python -m docsagent search "How many leave days can I carry over?"

export ANTHROPIC_API_KEY=...       # your own key; never commit it
python -m docsagent ask "How many leave days can I carry over, and until when?"

python -m pytest
```

As an API:

```bash
uvicorn docsagent.api:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How fast must on-call acknowledge a SEV1 page?"}'
```

With Docker:

```bash
docker build -t cited-docs-agent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY cited-docs-agent
```

Settings (environment variables): `DOCSAGENT_MODEL` (default `claude-haiku-5-5`) and
`DOCSAGENT_EMBED_MODEL` (default `BAAI/bge-small-en-v1.5`).

The sample documents in `sample_docs/` describe a fictional company, Northwind Labs.

## Tests

The agent, source check, evaluation scoring and API are tested with a fake search and a fake
LLM, so tests run offline, cost nothing and give the same result every time.

## Design choices

- **One section per chunk** instead of fixed-size windows, so a citation points to one clear
  section and a chunk never mixes two topics.
- **A rule-based source check after the model**, not a second model call: it is free, instant
  and predictable, and it catches the most common hallucination (made-up numbers).
- **Local embeddings and reranker**: no cost per document and no document text sent out for
  indexing; only the question and the few retrieved pieces go to Claude.
- **Retrieve wide, then rerank**: cheap vector search for recall, an accurate cross-encoder for
  precision, measured side by side in CI.
- **Defence in depth against prompt injection**: screen the sources, instruct the model, verify
  the output, and give the model no tools to misuse.

## Limits and next steps

- The source check verifies citations and numbers, not the meaning of every sentence. Next: an
  LLM-as-judge check for unsupported claims without numbers.
- The injection screen is rule-based, so a cleverly worded attack can slip past it; the
  instruction and output layers are there for that case. Next: a small classifier model.
- Search is meaning-only. Next: hybrid search (keyword + vector) for names and codes.
