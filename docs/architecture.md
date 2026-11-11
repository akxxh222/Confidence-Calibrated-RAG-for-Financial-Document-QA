# Architecture

CC-RAG answers questions from indexed SEC 10-K and 10-Q filings. The web app
combines dense retrieval with repeated LLM generation and a learned correctness
classifier. It returns `answer`, `hedge`, or `refuse` with source references.

## Code map

| Directory/file | Responsibility | First file to read |
| --- | --- | --- |
| `app/api/` | Flask routes, JSON schema, HTML/CSS/JavaScript | `app/api/app.py` |
| `app/ingestion/` | SEC downloads, HTML extraction, section-aware chunks | `app/ingestion/chunker.py` |
| `app/retrieval/` | Gemini embeddings and pgvector cosine search | `app/retrieval/retriever.py` |
| `app/generation/` | Grounded prompts, retries, parallel samples, agreement | `app/generation/self_consistency.py` |
| `app/calibration/` | Feature vector, classifier persistence, training | `app/calibration/model.py` |
| `db/` | Shared database connection and SQL schema | `db/schema.sql` |
| `scripts/` | Database initialization and corpus ingestion entry points | `scripts/ingest_sample.py` |
| `eval/` | Development seed, evaluation datasets, baselines, experiment code | `eval/run_experiment.py` |
| `models/` | Saved v0 classifier and provenance metadata | `models/calibration_model_v0.json` |
| `tests/` | Offline unit tests and mocked API checks | `tests/test_api_smoke.py` |
| `.github/workflows/` | Lint/test, container build, image publishing | `.github/workflows/ci.yml` |

## Offline ingestion

```text
SEC EDGAR -> download HTML/submission metadata -> extract readable text
          -> detect sections -> split within each section
          -> Gemini embeddings -> PostgreSQL documents + chunks
```

The extractor retains table rows as pipe-delimited text. The chunker recognizes
MD&A, financial statements, footnotes, and market risk; it supports annual and
quarterly Item numbers and some broken/spaced header forms. Content before the
first recognized header is kept as `unlabeled`. The secondary split uses a
400-word proxy for tokens and 50-word overlap, not an exact tokenizer. It avoids
crossing detected section boundaries but can still split large tables.

Embeddings use `gemini-embedding-2` with 768 dimensions and explicit vector
normalization. `db/schema.sql` stores `VECTOR(768)`. Changing dimensions requires
a planned migration and re-embedding; changing the embedding model also requires
re-indexing even if the vector length stays the same.

## Request lifecycle

```text
Browser -> POST /ask -> embed question -> pgvector top-k passages
        -> five grounded Gemini generations (up to three concurrently)
        -> pairwise agreement -> [top1, spread, consistency]
        -> saved logistic classifier -> confidence and routing -> JSON -> page
```

The retriever uses `1 - cosine_distance` and returns five chunks by default.
The small-corpus configuration probes all 100 IVFFlat lists. It does not
currently use hybrid keyword retrieval or a cross-encoder reranker.

Each generation receives the same question and retrieved context, with
temperature 0.7. Samples retain submission order regardless of network completion
order. The first usable sample is the primary answer. Transient provider failures
can be tolerated when at least three samples succeed; quota/configuration failures
stop further batches. Parallelism reduces waiting time, not API usage.

## Score meanings

| Signal | Definition | Limitation |
| --- | --- | --- |
| Top match | Highest retrieved cosine similarity | Similarity is not a probability of correctness |
| Ranking spread | Top-1 similarity minus top-3 similarity | A small gap can mean several equally useful passages |
| Self-consistency | Pairwise agreement across generated samples | Agreement can be consistently wrong |
| Answer confidence | Saved classifier estimate from the three features | Small development seed; requires revalidation after model changes |

Numeric agreement compares extracted financial figures within a 2% relative
tolerance, accounting for common currency/unit forms and skipping plain years
when possible. With fewer than two numeric samples, scoring falls back to mean
pairwise cosine similarity of answer embeddings. Numeric extraction currently
focuses on one figure per sample, so agreement is not full claim verification.

The artifact's current thresholds are 0.50 for answer and 0.20 for hedge.
A hedge prepends a qualification to the generated answer; it does not
necessarily mean the answer is partial. Refusal returns a fixed explanation.
Empty retrieval refuses immediately without generation. More details and the
generation-model change caveat are in [Calibration](calibration.md).

## API surface

| Route | Purpose |
| --- | --- |
| `GET /` | Filing Research Console |
| `GET /health` | Process liveness; no database/provider check |
| `POST /ask` | Full calibrated pipeline used by the page |
| `POST /api/query` | One-generation RAG with retrieval metrics |

Both POST routes accept `{"question": "..."}`. The calibrated JSON response
contains `question`, `mode`, `answer`, `confidence_score`, `retrieval_confidence`,
`spread`, `self_consistency_score`, and `sources`. Each source includes chunk ID,
document ID, detected section, optional page reference, and similarity.

Provider quota errors return HTTP 429 and deadline failures HTTP 504. Other
handled Gemini API failures return HTTP 503. The frontend renders the JSON error
and checks content type before parsing, so an unexpected HTML response does not
appear as a JSON syntax error. See [Operations](operations.md).

## Storage

PostgreSQL contains `documents`, `chunks`, `eval_questions`, and `eval_results`.
The online pipeline reads `documents`/`chunks`; evaluation currently uses JSONL
files rather than the evaluation SQL tables. Canonical SEC source URLs and fiscal
periods are populated during ingestion. `page_ref` can be absent for HTML filings.
Docker persists the database in the Compose-managed `ccrag_pgdata` named volume.
