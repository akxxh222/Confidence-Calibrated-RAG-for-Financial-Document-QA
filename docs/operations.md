# Operations runbook

## Configuration

`.env` is private and ignored. `.env.example` is the tracked configuration
reference. Current defaults are:

| Variable | Default/example | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Empty | Google API credential; required for live requests |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Runtime generation model |
| `GEMINI_THINKING_LEVEL` | `low` | Thinking level; must be supported by the chosen model |
| `GENERATION_MAX_OUTPUT_TOKENS` | `1000` | Per-generation output limit |
| `GEMINI_HTTP_TIMEOUT_MS` | `20000` | Generation request timeout |
| `GEMINI_MAX_ATTEMPTS` | `2` | Generation attempts, including the initial call |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Retrieval/answer embedding model |
| `EMBEDDING_DIM` | `768` | Must match `VECTOR(768)` in the database |
| `POSTGRES_USER` / `POSTGRES_DB` | `ccrag` | Database identity and name |
| `POSTGRES_PASSWORD` | `change_me` placeholder | Replace and match the host database URL |
| `DATABASE_URL` | PostgreSQL URL on `localhost:5433` | Local tooling connection; Compose overrides API to `db:5432` |
| `SELF_CONSISTENCY_N` | `5` | Independent generation samples per question |
| `SELF_CONSISTENCY_MIN_SAMPLES` | `3` | Minimum usable samples if transient calls fail |
| `SELF_CONSISTENCY_WORKERS` | `3` | Maximum concurrent generations per question |
| `TOP_K` | `5` | Retrieved chunks per question |
| `IVFFLAT_PROBES` | `100` | Probed index lists for the initial small corpus |
| `EDGAR_NAME` / `EDGAR_EMAIL` | Empty | Required real SEC contact identity for downloads |

`LLM_PROVIDER=gemini` in the example is descriptive; there is currently no
implemented provider switch. Changing it alone does not select another LLM.

## Routine commands

```powershell
docker compose up -d --build
docker compose ps
docker compose logs --tail 100 api
docker compose logs -f api
```

After editing `.env`, recreate the API with `docker compose up -d --build api`.
A restart of the same container does not reload changed environment values.
Changing database initialization credentials does not change the password of a
database already stored in its persistent volume; apply an explicit database
credential rotation when needed.

```powershell
Invoke-RestMethod http://localhost:5000/health
docker compose stop api
docker compose down
```

The API listens on port 5000. PostgreSQL maps host 5433 to container 5432, to
avoid a local PostgreSQL service using host 5432. `down` keeps the database;
`down -v` deletes it and is not a normal maintenance step.

## Slow requests and Gemini failures

`/ask` generates five samples, so it uses more API requests than `/api/query`.
With three workers it runs two batches. Both Gunicorn workers may serve a
request at once, so total generation concurrency can exceed three across users.
Lower `SELF_CONSISTENCY_WORKERS` to `1` if your quota cannot tolerate bursts.
This changes scheduling without reducing samples or token usage.

The generation client retries selected transient failures with bounded delay;
the embedding client has a separate quota-retry path. Embedding timeout/retry
behavior is not controlled by `GEMINI_HTTP_TIMEOUT_MS`. The application has no
single end-to-end request deadline, and the Compose Gunicorn worker timeout is
180 seconds. A long embedding retry or provider stall can still exceed it.
See request errors in `docker compose logs --tail 100 api`.

| Symptom | What to check |
| --- | --- |
| HTTP 429 | Rate/quota limits on the key's Google project; a key change may share the same exhausted quota |
| HTTP 504 | Gemini deadline expired; check current model responsiveness and generation timeout |
| HTTP 503 | Other handled Gemini API failure; inspect the recorded provider status |
| HTTP 500 or unexpected HTML | Inspect the server traceback; this is not a normal confidence refusal |
| `WORKER TIMEOUT` | Total work exceeded Gunicorn's timeout; investigate provider/retry stages |
| Empty retrieval | Check documents/chunks, database URL, and successful ingestion |
| Model artifact missing | Restore the approved `.joblib` and JSON pair under `models/` |
| Database connection fails | Check Docker, `pg_isready`, host/container ports, and matching credentials |

`/health` is process liveness only. Gemini clients are lazy, so this check can
pass while provider credentials or quota are unusable. Never print the full
environment, API key, or `docker compose config` output into a public issue.

## Database care

The schema file initializes a new Docker database; `scripts.init_db` reapplies
`CREATE ... IF NOT EXISTS` statements but is not a general migration system.
There is no automatic migration runner. Plan explicit SQL migrations and backups
before changing existing data or embedding dimensions.

A logical backup, using the example database user/name, can be created without
PowerShell binary redirection:

```powershell
New-Item -ItemType Directory -Force data/backups
docker compose exec -T db pg_dump -U ccrag -d ccrag -Fc -f /tmp/ccrag-backup.dump
docker compose cp db:/tmp/ccrag-backup.dump data/backups/ccrag-backup.dump
```

Choose a new filename per backup to avoid overwriting one. Store important
backups outside this workstation as well. Test restores into a separate database
before replacing a working one; do not reset the development volume to solve an
API quota problem.

## Deployment boundary

Compose is the local development setup. It exposes the API and database ports
and does not provide authentication, TLS, request rate limiting, a migration
service, or managed backups. Before external deployment, add those controls,
limit database exposure, and use an appropriate secrets system. CI publishes
an image; it does not provision or deploy the runtime environment.
