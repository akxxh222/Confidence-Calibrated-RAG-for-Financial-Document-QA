CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id SERIAL PRIMARY KEY,
    company VARCHAR(50),
    doc_type VARCHAR(10),          -- '10-K' | '10-Q' | 'transcript'
    fiscal_period VARCHAR(20),
    source_url TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id SERIAL PRIMARY KEY,
    doc_id INT REFERENCES documents(doc_id),
    section VARCHAR(50),           -- 'MD&A' | 'financial_statements' | 'footnotes' | 'qa_segment'
    text TEXT NOT NULL,
    embedding VECTOR(768),         -- must match EMBEDDING_DIM in .env (gemini-embedding-2 @ 768)
    page_ref VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS eval_questions (
    question_id SERIAL PRIMARY KEY,
    question_text TEXT NOT NULL,
    category VARCHAR(20) NOT NULL,   -- 'answerable' | 'false_premise' | 'unanswerable'
    ground_truth_answer TEXT,
    ground_truth_source_chunk INT REFERENCES chunks(chunk_id)
);

CREATE TABLE IF NOT EXISTS eval_results (
    result_id SERIAL PRIMARY KEY,
    question_id INT REFERENCES eval_questions(question_id),
    system_variant VARCHAR(20) NOT NULL,  -- 'always_answer' | 'fixed_threshold' | 'calibrated'
    mode_chosen VARCHAR(10) NOT NULL,      -- 'answer' | 'hedge' | 'refuse'
    generated_answer TEXT,
    was_hallucination BOOLEAN,
    retrieval_confidence FLOAT,
    retrieval_spread FLOAT,
    self_consistency_score FLOAT,
    calibration_model_version VARCHAR(30)
);
