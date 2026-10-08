"""Applies db/schema.sql to the configured DATABASE_URL. Idempotent (IF NOT EXISTS)."""
from db.connection import get_connection

if __name__ == "__main__":
    with open("db/schema.sql") as f:
        sql = f.read()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("Schema applied.")
    finally:
        conn.close()
