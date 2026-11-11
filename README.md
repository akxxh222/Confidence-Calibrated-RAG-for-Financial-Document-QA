# CC-RAG

Confidence-calibrated question answering over SEC financial filings.
The application retrieves filing passages, generates answers with Gemini,
and uses retrieval and self-consistency signals to answer, hedge, or refuse.

## Run locally

Install Docker Desktop with Compose. Create `.env` from `.env.example` if it
does not exist, then set your Gemini key and matching database credentials.

```powershell
docker compose up -d --build
```

Open [localhost:5000](http://localhost:5000). A fresh clone starts with an empty
database: follow [Getting started](docs/getting-started.md) to ingest filings
before asking questions. An existing local database is kept in a Docker volume.

## Start here

| Need | Guide |
| --- | --- |
| First day: install, ingest, run, verify | [Getting started](docs/getting-started.md) |
| Locate code and understand a request | [Architecture](docs/architecture.md) |
| Make and review a change | [Development](docs/development.md) |
| Configuration, Docker, errors, database recovery | [Operations](docs/operations.md) |
| Training data, saved classifier, thresholds | [Calibration](docs/calibration.md) |
| Datasets and comparison experiment | [Evaluation](docs/evaluation.md) |

The [documentation index](docs/README.md) lists all maintained guides.
Application code lives in `app/`, SQL in `db/`, evaluation code/data in `eval/`,
setup commands in `scripts/`, and automated checks in `tests/`.

The current runtime uses Gemini 3.5 Flash and five generations in parallel
batches of up to three. Confidence scores come from a small development-seed
classifier trained with an earlier generation model; see the
[calibration limitations](docs/calibration.md#current-artifacts-and-configuration).
