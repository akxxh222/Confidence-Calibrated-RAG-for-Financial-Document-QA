# CC-RAG: Confidence-Calibrated RAG for Financial Documents

Retrieval-augmented QA over SEC filings that decides to **answer / hedge / refuse**
based on a learned combination of retrieval confidence and generation
self-consistency, instead of a fixed similarity threshold.

## Repo layout

```
cc-rag/
├── app/
│   ├── ingestion/        # Person A — corpus download + section-aware chunking
│   │   ├── edgar_fetch.py
│   │   └── chunker.py
│   ├── retrieval/        # Person B — embeddings, pgvector search, reranking
│   │   ├── embed.py
│   │   └── retriever.py
│   ├── generation/       # Person B — LLM calls + self-consistency sampling
│   │   ├── generate.py
│   │   └── self_consistency.py
│   ├── calibration/      # Person B — the core novel mechanism
│   │   ├── features.py
│   │   ├── model.py
│   │   └── train.py
│   └── api/              # Person A — Flask REST layer
│       ├── app.py
│       └── schemas.py
├── db/
│   ├── schema.sql
│   └── connection.py
├── eval/                 # Person C — test set + comparison experiment
│   ├── test_set/
│   │   ├── answerable.jsonl
│   │   ├── false_premise.jsonl
│   │   └── unanswerable.jsonl
│   ├── baselines.py
│   ├── run_experiment.py
│   └── metrics.py
├── tests/                # CI unit tests
├── scripts/               # one-off setup scripts
├── .github/workflows/ci.yml
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Build order (matches the 4-week plan)

1. **Ingestion → Storage → Retrieval** (Week 1) — get raw text into pgvector and
   a naive top-k search working end to end before touching calibration at all.
2. **Generation + self-consistency + calibration v0** (Week 2) — wire in the LLM,
   sample N answers, compute agreement, train a first-pass calibration model on
   a small hand-labeled seed set.
3. **Eval test set + comparison experiment** (Week 3) — the part that makes this
   a paper, not a demo. Build the 3-category set, run all three system variants,
   compute hallucination-rate/coverage with bootstrap CIs.
4. **Ablation + polish + writeup** (Week 4).

## Local dev

```bash
cp .env.example .env          # fill in DB creds + LLM API key
docker compose up -d db       # postgres + pgvector only, for local dev
pip install -r requirements.txt
python scripts/init_db.py     # creates tables from db/schema.sql
python scripts/ingest_sample.py   # pulls a couple of filings to sanity-check the pipeline
flask --app app.api.app run --debug
```
