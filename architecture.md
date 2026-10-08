# CC-RAG — System Architecture

Confidence-Calibrated RAG for Financial Document QA. The system answers
questions about SEC filings and returns one of three modes — **answer**,
**hedge**, or **refuse** — based on a learned calibration model that combines
retrieval confidence with generation self-consistency, instead of a fixed
similarity-score cutoff.

> **Status (2026-10-08):** Milestone 0 complete and verified (empty pipeline
> runs end-to-end against live Docker Postgres and the live Gemini API).
> Milestone 1 (real corpus ingestion) in progress: EDGAR identity configured,
> first filing fetch pending.

## 1. Stack summary

| Layer | Technology |
|---|---|
| Corpus | SEC EDGAR filings (10-K / 10-Q); earnings-call transcripts deferred |
| Extraction | BeautifulSoup 4 + lxml (HTML → plain text) |
| Chunking | Custom section-aware chunker (400-token cap, 50-token overlap) |
| Embeddings | Google Gemini `gemini-embedding-2` @ 768 dims (`google-genai` SDK) |
| Vector store | PostgreSQL 15 + pgvector — Docker, host port **5433** |
| Generation | Google Gemini `gemini-3.8-flash` (`models.generate_content`) |
| Calibration | scikit-learn LogisticRegression (shallow-MLP swap-in) over 3 signals |
| API | Flask + gunicorn — `POST /ask`, `GET /health` |
| Eval | 3-category test set, bootstrap-CI metrics, JSONL experiment flow |
| CI/CD | GitHub Actions: flake8 + pytest → Docker build → ghcr.io publish |

## 2. Data flow

```
        INGESTION (offline, once per filing)
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ SEC EDGAR   │──▶│ edgar_fetch  │──▶│ html_extract │──▶│ chunker      │
│ 10-K / 10-Q │   │ (HTML into   │   │ (HTML→text;  │   │ (section-    │
│             │   │ data/raw_    │   │ tables become│   │ aware; ≤400  │
│             │   │ filings/)    │   │ "a | b" rows)│   │ tokens)      │
└─────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                               ▼
                 ┌──────────────────────┐   ┌─────────────────────────────┐
                 │ embed.py             │──▶│ Gemini API                  │
                 │ gemini-embedding-2   │   │ models.embed_content        │
                 │ @ 768 dims           │   │ (output_dimensionality=768) │
                 └──────────┬───────────┘   └─────────────────────────────┘
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │ PostgreSQL 15 + pgvector — Docker, host port 5433          │
   │ documents · chunks(text, VECTOR(768)) · eval_questions ·   │
   │ eval_results                                               │
   └─────────────────────────────┬──────────────────────────────┘
                                 │
   QUERY TIME (Flask POST /ask)  ▼
   ① retriever.py ─── top-k cosine search → top1_score, spread
                                 ▼
   ② self_consistency.py ─── N=5 samples @ T=0.7 → agreement score
                                 ▼
   ③ calibration model (logreg) ─── P(correct)
                                 ▼
   ④ decide() ─── ≥0.70 answer · ≥0.40 hedge · <0.40 refuse
                                 ▼
   {mode, answer, confidence_score, retrieval_confidence,
    self_consistency_score, sources}
```

Evaluation runs offline (`eval/run_experiment.py`): the 3-category test set —
answerable / false-premise / unanswerable — is run through three system
variants (always-answer, fixed-threshold, calibrated), hand-judged, and
summarised by `eval/metrics.py` into the hallucination-rate/coverage table
with bootstrap 95% CIs. That table is the paper's core result.

## 3. Query lifecycle (POST /ask)

1. Parse and validate the question (400 if empty).
2. `retrieve()` — embed the query (Gemini), cosine top-k against `chunks`,
   compute `top1_score` and `spread = top1 − top3`. If retrieval is empty the
   endpoint refuses immediately — no LLM call, calibration model not loaded.
3. `run_self_consistency()` — N=5 independent generations at temperature 0.7;
   the first numeric figure of each sample is extracted (handles `$`, `,`,
   `%`, billion/million/B/M scaling) and samples are compared pairwise within
   a 2% relative tolerance → agreement score in [0, 1]. If no sample contains
   a number, the fallback is the mean pairwise dot product of normalized
   embeddings.
4. `CalibrationFeatures(top1_score, spread, self_consistency_score)` →
   `CalibrationModel.predict_proba` → P(correct).
5. `decide()`: P ≥ 0.70 → answer · P ≥ 0.40 → hedge · otherwise refuse.
6. Response JSON: `mode`, final answer text (hedge prepends a
   verify-against-source warning; refuse returns a fixed explanation),
   `confidence_score`, `retrieval_confidence`, `self_consistency_score`,
   `sources` (section + page ref per retrieved chunk).

<!-- ARCHITECTURE-CONTINUES-1 -->
