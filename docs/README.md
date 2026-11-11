# Documentation

Start with [Getting started](getting-started.md), then read
[Architecture](architecture.md) and [Development](development.md).
All shell commands assume the repository root as the working directory.

| Document | Contents |
| --- | --- |
| [Getting started](getting-started.md) | Prerequisites, configuration, first ingestion, launch, verification |
| [Architecture](architecture.md) | Code map, data flow, API contracts, score definitions |
| [Development](development.md) | Local Flask development, tests, Git policy, CI/CD |
| [Operations](operations.md) | Environment variables, service commands, quota/timeouts, database care |
| [Calibration](calibration.md) | Development seed, labels, training, threshold selection, artifacts |
| [Evaluation](evaluation.md) | Final test set, baselines, experiment generation and reporting |
| [Evaluation review](evaluation-review.md) | Independent review and disagreement resolution |

`docs/local/` contains private historical planning/review notes and is ignored
by Git. `docs/generated/` is reserved for generated evaluation reports and is
also ignored. Neither directory is required to build or run the application.

When behavior changes, update the relevant guide and `.env.example` in the same
change. Treat source code, checked-in data, and saved model metadata as the
authority for executable behavior and measured results.
