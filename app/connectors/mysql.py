
import mysql.connector
import os

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

def get_mysql_connection():
    """Create and return a MySQL connection."""
    if not all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE]):
        return {"error": "MySQL connection details not fully configured."}
    try:
        return mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
    except mysql.connector.Error as e:
        return {"error": str(e)}

def list_tables():
    """List all tables in the database."""
    conn = get_mysql_connection()
    if isinstance(conn, dict) and "error" in conn:
        return conn
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            tables = [table[0] for table in cur.fetchall()]
            return {"tables": tables}
    except mysql.connector.Error as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

def scan_table(table_name: str):
    """Scan a table and return its content as a string."""
    conn = get_mysql_connection()
    if isinstance(conn, dict) and "error" in conn:
        return conn
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name}")
            rows = cur.fetchall()
            # Convert rows to a string format for analysis
            content = "\n".join(["\t".join(map(str, row)) for row in rows])
            return content
    except mysql.connector.Error as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()
