# CC-RAG: Confidence-Calibrated RAG for Financial Documents

Retrieval-augmented QA over SEC filings that decides to **answer / hedge / refuse**
based on a learned combination of retrieval confidence and generation
self-consistency, instead of a fixed similarity threshold.

## Repo layout

```
cc-rag/
├── app/
│   ├── ingestion/        # Person A — corpus download + HTML→text + chunking
│   │   ├── edgar_fetch.py
│   │   ├── html_extract.py
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
python -m venv .venv          # use an isolated environment
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
docker compose up -d db       # postgres + pgvector only, for local dev
python -m scripts.init_db     # creates tables from db/schema.sql (idempotent)
python -m app.ingestion.edgar_fetch --tickers AAPL --forms 10-K --limit 1
python -m app.ingestion.html_extract --input data/raw_filings --out data/clean_text
python -m scripts.ingest_sample --file data/clean_text/<filing>.txt --company AAPL
flask --app app.api.app run --debug
```
