import sqlite3

def init_db(db_path="preview.db"):
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

if __name__ == "__main__":
    init_db()
    print("Database tables created (if not already existing).")
