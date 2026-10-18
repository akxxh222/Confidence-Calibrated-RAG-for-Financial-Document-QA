# Milestone 3 Independent Review

Every question needs review by a team member other than the person who wrote
its label. Do not mark `independent_verified` as `true` based only on the
stored answer.

The JSONL files are the canonical reviewed dataset. Do not rerun
`eval.build_test_set` after review begins; its `--force` option rebuilds the
initial author-reviewed seed and can erase later corrections or review fields.

For each JSONL record:

1. Read the question, category, company, filing type, and fiscal period.
2. Open each SEC `source_url` and locate the stored evidence excerpt.
3. For `answerable`, recalculate or directly confirm the ground-truth answer.
4. For `false_premise`, confirm that the evidence explicitly disproves the
   premise and that the correction is accurate.
5. For `unanswerable`, search all indexed filings for that company using the
   recorded terms and reasonable synonyms. Confirm that no filing supports the
   requested answer.
6. Set `independent_verified` to `true`, and fill in
   `independent_reviewer` and `independent_reviewed_on`.
7. If the first label was disputed, record the issue in `disagreement`, agree
   on the corrected record, and explain the outcome in `resolution`.

After editing, run:

```powershell
.\.venv\Scripts\python.exe -m eval.dataset eval\test_set
.\.venv\Scripts\python.exe -m eval.audit_dataset
```

Milestone 3 is complete only when the validator reports
`Independently verified: 75/75`, the corpus audit passes, and all disagreements
have a recorded resolution.
