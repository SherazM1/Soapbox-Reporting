import sqlite3
import json
from datetime import datetime

DB_PATH = "preview.db"

# ────────────────────────────────────────────────────────────────
# Table Setup
# ────────────────────────────────────────────────────────────────

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS clients (
            client_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_name TEXT NOT NULL UNIQUE,
            created TEXT NOT NULL
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS previews (
            preview_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            preview_name TEXT NOT NULL,
            report_date TEXT NOT NULL,
            notes TEXT,
            date_created TEXT NOT NULL,
            data_json TEXT NOT NULL,
            FOREIGN KEY (client_id) REFERENCES clients(client_id)
        );
    """)
    conn.commit()
    conn.close()

# ────────────────────────────────────────────────────────────────
# Client CRUD
# ────────────────────────────────────────────────────────────────

def add_client(client_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    created = datetime.now().isoformat()
    try:
        c.execute(
            "INSERT INTO clients (client_name, created) VALUES (?, ?)",
            (client_name, created)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # client_name already exists
    finally:
        conn.close()

def get_clients():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT client_id, client_name, created FROM clients ORDER BY client_name")
    rows = c.fetchall()
    conn.close()
    return [{"client_id": r[0], "client_name": r[1], "created": r[2]} for r in rows]

def delete_client(client_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM clients WHERE client_id = ?", (client_id,))
    conn.commit()
    conn.close()

# ────────────────────────────────────────────────────────────────
# Preview CRUD
# ────────────────────────────────────────────────────────────────

def add_preview(client_id, preview_name, report_date, notes, data_json):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    date_created = datetime.now().isoformat()
    c.execute(
        "INSERT INTO previews (client_id, preview_name, report_date, notes, date_created, data_json) VALUES (?, ?, ?, ?, ?, ?)",
        (client_id, preview_name, report_date, notes, date_created, json.dumps(data_json))
    )
    conn.commit()
    conn.close()

def get_previews_for_client(client_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT preview_id, preview_name, report_date, notes, date_created, data_json FROM previews WHERE client_id = ? ORDER BY report_date DESC",
        (client_id,)
    )
    rows = c.fetchall()
    conn.close()
    previews = []
    for r in rows:
        previews.append({
            "preview_id": r[0],
            "preview_name": r[1],
            "report_date": r[2],
            "notes": r[3],
            "date_created": r[4],
            "data_json": json.loads(r[5])
        })
    return previews

def delete_preview(preview_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM previews WHERE preview_id = ?", (preview_id,))
    conn.commit()
    conn.close()

# ────────────────────────────────────────────────────────────────
# Only run this if the script is called directly, not on import
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("Database tables created (if not already existing).")
