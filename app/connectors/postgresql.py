
import psycopg2
import os

DATABASE_URL = os.getenv("DATABASE_URL")

def get_postgresql_connection():
    """Create and return a PostgreSQL connection."""
    if not DATABASE_URL:
        return {"error": "DATABASE_URL not configured."}
    try:
        return psycopg2.connect(DATABASE_URL)
    except psycopg2.Error as e:
        return {"error": str(e)}

def list_tables():
    """List all tables in the database."""
    conn = get_postgresql_connection()
    if isinstance(conn, dict) and "error" in conn:
        return conn
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [table[0] for table in cur.fetchall()]
            return {"tables": tables}
    except psycopg2.Error as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

def scan_table(table_name: str):
    """Scan a table and return its content as a string."""
    conn = get_postgresql_connection()
    if isinstance(conn, dict) and "error" in conn:
        return conn
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name}")
            rows = cur.fetchall()
            # Convert rows to a string format for analysis
            content = "\n".join(["\t".join(map(str, row)) for row in rows])
            return content
    except psycopg2.Error as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()
