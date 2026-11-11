# Getting started

## Prerequisites

- Docker Desktop running with Linux containers and Docker Compose.
- Python 3.11 for local tooling and ingestion, matching CI and the container.
- A Gemini API key with generation and embedding model access and available quota.
- A real SEC downloader name and contact email when downloading filings.

Clone the repository and open a terminal in its root. The commands below use
Windows PowerShell; the Python module and Docker commands also work elsewhere.

## Configure once

Create your private environment file only when it is missing:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env`: set `GEMINI_API_KEY`, replace `POSTGRES_PASSWORD`, and use the same
password in `DATABASE_URL`. For local host tools, the URL uses `localhost:5433`;
Compose supplies the API container with a separate URL using `db:5432`.
Set `EDGAR_NAME` and `EDGAR_EMAIL` for ingestion. See
[configuration](operations.md#configuration) for all settings.

Create the local tool environment and start the database:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d db
docker compose exec db pg_isready -U ccrag -d ccrag
```

The last command assumes the example database/user name `ccrag`; adapt both
flags if you changed them. Wait for `accepting connections` before continuing.
On first database creation, Compose applies `db/schema.sql` automatically.
For an existing database whose schema needs the additive initialization:

```powershell
.\.venv\Scripts\python.exe -m scripts.init_db
```

This script does not migrate existing column types. Do not change the
embedding model or dimension as part of first-day setup.

## Ingest a fresh database

Skip this section when using an existing, populated development database.
Downloads use current SEC results, so the exact years can change over time.
The initial corpus configuration is two companies with two annual and three
quarterly filings each, for ten filings in total.

```powershell
.\.venv\Scripts\python.exe -m app.ingestion.edgar_fetch --tickers AAPL MSFT --forms 10-K --limit 2
.\.venv\Scripts\python.exe -m app.ingestion.edgar_fetch --tickers AAPL MSFT --forms 10-Q --limit 3
.\.venv\Scripts\python.exe -m app.ingestion.html_extract --input data/raw_filings --out data/clean_text
.\.venv\Scripts\python.exe -m scripts.ingest_sample --root data/clean_text --raw_root data/raw_filings
```

Embedding requires live Gemini requests. Ingestion skips filings with source
URLs already present in the database. Raw downloads and extracted text stay
in ignored `data/`; they are not shipped in the repository or Docker image.
Downloaded HTML uses
`data/raw_filings/sec-edgar-filings/<ticker>/<form>/<accession>/`.

Check the indexed corpus:

```powershell
docker compose exec db psql -U ccrag -d ccrag -c "SELECT company, doc_type, fiscal_period FROM documents ORDER BY doc_id;"
docker compose exec db psql -U ccrag -d ccrag -c "SELECT count(*) AS chunks, count(embedding) AS embedded FROM chunks;"
```

The counts must be nonzero, with every ingested chunk embedded. Query a fiscal
period present in the first command's output; do not assume the latest download
contains an older demonstration year.

## Run and verify

```powershell
docker compose up -d --build
Invoke-RestMethod http://localhost:5000/health
```

Open [localhost:5000](http://localhost:5000). The web page calls `POST /ask`,
which uses retrieved passages, five live Gemini generations, self-consistency,
and the saved calibration model. Generations run in batches of up to three.

`GET /health` verifies the HTTP process only; a successful question also checks
the database, corpus, API credentials, embedding service, generation service,
and model artifact. Confidence determines routing, so even a supported figure
can receive a qualified answer. See [Architecture](architecture.md).

The calibrated route needs both checked-in `models/calibration_model_v0` files.
Do not retrain them merely to launch the application.

## Stop or reload

```powershell
docker compose down
```

This keeps the database volume. After changing `.env` or code, use
`docker compose up -d --build api` to recreate the application with the change.
Do not use `down -v` for routine shutdown; it deletes the indexed database.
