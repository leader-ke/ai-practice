import psycopg
from pgvector.psycopg import register_vector


def connect(database_url):
    if not database_url:
        raise ValueError("DATABASE_URL is missing from environment configuration.")
    conn = psycopg.connect(database_url)
    register_vector(conn)
    return conn


def store_chunks(conn, chunks, vectors, config):
    """
    Replaces existing chunks for a configuration and stores
    their corresponding embeddings.
    """
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chunks WHERE config = %s",
            (config,),
        )

        for chunk, vector in zip(chunks, vectors, strict=True):
            cur.execute(
                """
                INSERT INTO chunks
                (document, paragraph_ids, text, config, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    chunk["document"],
                    chunk["paragraph_ids"],
                    chunk["text"],
                    config,
                    vector,
                ),
            )

    conn.commit()


def search(conn, query_vector, config, k):
    """
    Finds the k nearest chunks by cosine distance (pgvector's <=> operator).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT document, paragraph_ids, text, distance
            FROM (
                SELECT document, paragraph_ids, text,
                       embedding <=> %s AS distance
                FROM chunks
                WHERE config = %s
            ) sub
            ORDER BY distance
            LIMIT %s
            """,
            (query_vector, config, k),
        )
        return cur.fetchall()
