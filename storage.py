import sqlite3
import os
from datetime import datetime

DB_PATH = os.getenv('SACK_DB_PATH', os.path.join(os.path.dirname(__file__), 'sack_counts.db'))


def init_db(db_path=None):
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS counts (
            date TEXT PRIMARY KEY,
            total INTEGER NOT NULL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            obj_id INTEGER,
            source TEXT,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()


def _conn(path=None):
    return sqlite3.connect(path or DB_PATH, timeout=30)


def get_today_total():
    today = datetime.utcnow().date().isoformat()
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT total FROM counts WHERE date = ?', (today,))
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def increment(amount=1):
    today = datetime.utcnow().date().isoformat()
    conn = _conn()
    cur = conn.cursor()
    # use upsert to add or increment
    cur.execute('INSERT INTO counts(date,total) VALUES(?,?) ON CONFLICT(date) DO UPDATE SET total = total + excluded.total', (today, amount))
    conn.commit()
    conn.close()
    return get_today_total()


def log_event(obj_id=None, source=None, details=None):
    ts = datetime.utcnow().isoformat()
    date = ts[:10]
    conn = _conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO events(timestamp,date,obj_id,source,details) VALUES(?,?,?,?,?)', (ts, date, obj_id, source, details))
    conn.commit()
    conn.close()
