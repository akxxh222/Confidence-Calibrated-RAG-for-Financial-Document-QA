# Development and Git workflow

## Run Flask while editing

Follow [Getting started](getting-started.md) to create `.venv`, configure `.env`,
and ingest a corpus. For local Flask development, leave PostgreSQL in Docker
and stop the containerized API so port 5000 is available:

```powershell
docker compose stop api
docker compose up -d db
.\.venv\Scripts\python.exe -m flask --app app.api.app run --debug
```

Use `localhost:5433` in the local `DATABASE_URL`. Run commands from the repository
root: dataset paths, SQL initialization, and model loading are currently relative
to that directory. Restart Flask after changing `.env`, because settings and
Gemini clients are cached in the process.

## Verify a change

These commands match CI's Python checks and do not require a Gemini key or a
live database. If `.env` is absent, provide a placeholder database URL for imports:

```powershell
$env:DATABASE_URL = 'postgresql://ccrag:ccrag@localhost:5433/ccrag'
.\.venv\Scripts\python.exe -m flake8 app eval tests --max-line-length=100 --extend-ignore=E203
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

The example overrides the URL only in the current shell. Remove it with
`Remove-Item Env:DATABASE_URL` before using the real `.env` database credentials.
Tests mock provider and database interactions; passing tests do not establish
live quota, corpus availability, or provider performance.

For runtime changes, also build the image with `docker compose build api` and
perform a real `/ask` check against a populated development database. Live checks
consume API quota; inspect retrieved evidence as well as the HTTP status.

## What belongs in Git

| Commit | Keep local/ignored |
| --- | --- |
| Application, SQL, scripts, tests, CI configuration | `.env` and private credential files |
| `.env.example` with placeholders | `.venv`, caches, logs, editor/assistant state |
| Maintained guides under `docs/` and root README | `docs/local/` planning/review notes |
| Reviewed seed/test JSONL and review records | Raw calibration collection checkpoints |
| The approved v0 `.joblib` and matching JSON metadata | Other generated model files, MLflow runs |
| Reproducible evaluation/report code | Downloaded filings and generated reports |

The checked-in classifier pair is deliberate: `/ask` requires it to start
answering. Keep both files together and review provenance when replacing it.
Load joblib artifacts only from trusted sources.

Before making a commit:

```powershell
git status --short
git diff --check
git diff
```

Stage explicit paths, review the staged diff, and use a focused message such as
`fix: preserve sample order during parallel generation` or
`docs: explain fresh-clone setup`. Separate unrelated concerns; honor the team's
per-file commit convention when requested. Avoid blanket staging when local
artifacts or unrelated work are present. Current cleanup does not stage or
publish changes automatically.

## CI/CD

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`:

1. Install requirements on Python 3.11, run Flake8, then run offline Pytest checks.
2. Build a Docker image only after lint and tests pass.
3. Publish the image to GHCR only for a push to `main`, using GitHub's job token.

Publishing an image does not deploy a server or populate its database. Dependencies
currently include some ranges and container tags such as `latest`; builds are
not fully locked. Upgrade dependencies deliberately and verify both the local
Python and container environments when changing SDK versions.

## Where to make changes

- API validation or routing: `app/api/`; request tests in `tests/test_api_smoke.py`.
- Retrieval: `app/retrieval/`; verify embedding dimension and scoring semantics.
- LLM prompting/retries/concurrency: `app/generation/`; preserve sample independence
  and primary-answer ordering.
- Corpus extraction/chunking: `app/ingestion/`; use realistic header/table tests.
- Features, training, thresholds: `app/calibration/`; preserve feature order and
  keep development labels separate from final evaluation.

Update the appropriate guide in the same change so the next engineer can
reproduce the new behavior.
