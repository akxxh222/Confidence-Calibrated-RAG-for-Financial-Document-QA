"""Collect resumable real-pipeline runs for the calibration development set."""

import argparse
import json
from pathlib import Path

from app.generation.generate import MODEL, generate_answer
from app.generation.self_consistency import N_SAMPLES, score_samples
from app.retrieval.embed import EMBEDDING_MODEL
from app.retrieval.retriever import RetrievedChunk, retrieve


QUESTIONS_PATH = Path("eval/calibration_seed_questions.jsonl")
OUTPUT_PATH = Path("eval/calibration_seed_runs.jsonl")
PROGRESS_PATH = Path("eval/calibration_seed_progress.json")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_progress(progress: dict) -> None:
    temporary = PROGRESS_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(progress, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(PROGRESS_PATH)


def collect(limit: int | None = None) -> None:
    questions = load_jsonl(QUESTIONS_PATH)
    completed = load_jsonl(OUTPUT_PATH) if OUTPUT_PATH.exists() else []
    completed_ids = {row["question_id"] for row in completed}
    progress = (
        json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        if PROGRESS_PATH.exists()
        else {}
    )
    pending = [
        row for row in questions if row["question_id"] not in completed_ids
    ]
    if limit is not None:
        pending = pending[:limit]

    with OUTPUT_PATH.open("a", encoding="utf-8") as output:
        for index, question in enumerate(pending, start=1):
            question_id = question["question_id"]
            state = progress.get(question_id)
            if state is None:
                retrieval = retrieve(question["question_text"])
                state = {
                    "top1_score": retrieval.top1_score,
                    "spread": retrieval.spread,
                    "chunks": [chunk.__dict__ for chunk in retrieval.chunks],
                    "samples": [],
                }
                progress[question_id] = state
                save_progress(progress)

            chunks = [RetrievedChunk(**chunk) for chunk in state["chunks"]]
            while len(state["samples"]) < N_SAMPLES:
                sample = generate_answer(
                    question["question_text"], chunks, temperature=0.7
                )
                state["samples"].append(sample)
                save_progress(progress)
                print(
                    f"{question_id}: sample "
                    f"{len(state['samples'])}/{N_SAMPLES}"
                )

            consistency = score_samples(state["samples"])
            record = {
                **question,
                "top1_score": state["top1_score"],
                "spread": state["spread"],
                "self_consistency_score": (
                    consistency.self_consistency_score
                ),
                "generated_answer": consistency.samples[0],
                "samples": consistency.samples,
                "extracted_numbers": consistency.extracted_numbers,
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "section": chunk.section,
                        "similarity": chunk.similarity,
                    }
                    for chunk in chunks
                ],
                "generation_model": MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "sample_count": N_SAMPLES,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            progress.pop(question_id)
            save_progress(progress)
            print(
                f"Collected {index}/{len(pending)}: "
                f"{question['question_id']}"
            )

    print(f"Total collected: {len(completed_ids) + len(pending)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    collect(args.limit)


if __name__ == "__main__":
    main()
