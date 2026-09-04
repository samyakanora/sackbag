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
    # record each confirmed sack crossing with timestamp and sequential sack number
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sack_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            obj_id INTEGER,
            source TEXT,
            details TEXT,
            sack_number INTEGER
        )
    ''')

    # intervals for milestones (e.g., every 35 sacks)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS intervals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            milestone INTEGER NOT NULL,
            start_ts TEXT NOT NULL,
            end_ts TEXT NOT NULL,
            start_sack INTEGER NOT NULL,
            end_sack INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            rate_per_hour REAL NOT NULL
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


def record_sack(obj_id=None, source=None, details=None):
    """Record a confirmed sack crossing: insert into sack_events, increment daily total, and return the new global sack number and timestamp."""
    ts = datetime.utcnow().isoformat()
    date = ts[:10]
    conn = _conn()
    cur = conn.cursor()
    # determine next sack_number as total rows + 1
    cur.execute('SELECT COUNT(*) FROM sack_events')
    prev = cur.fetchone()[0] or 0
    sack_number = prev + 1
    cur.execute('INSERT INTO sack_events(timestamp,date,obj_id,source,details,sack_number) VALUES(?,?,?,?,?,?)', (ts, date, obj_id, source, details, sack_number))
    # also append to events for compatibility
    cur.execute('INSERT INTO events(timestamp,date,obj_id,source,details) VALUES(?,?,?,?,?)', (ts, date, obj_id, source, details))
    # update daily counts table
    cur.execute('INSERT INTO counts(date,total) VALUES(?,?) ON CONFLICT(date) DO UPDATE SET total = total + excluded.total', (date, 1))
    conn.commit()
    conn.close()

    return {
        'sack_number': sack_number,
        'timestamp': ts
    }


def get_sacks_in_period(start_ts, end_ts):
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM sack_events WHERE timestamp >= ? AND timestamp <= ?', (start_ts, end_ts))
    cnt = cur.fetchone()[0] or 0
    conn.close()
    return int(cnt)


def get_sacks_last_hour():
    end = datetime.utcnow()
    start = end.replace(minute=0, second=0, microsecond=0)
    # count since start of current hour
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM sack_events WHERE timestamp >= ? AND timestamp <= ?', (start.isoformat(), end.isoformat()))
    cnt = cur.fetchone()[0] or 0
    conn.close()
    return int(cnt)


def get_last_sack_time():
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT timestamp FROM sack_events ORDER BY id DESC LIMIT 1')
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def get_recent_sack_events(limit=10):
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT sack_number,timestamp,obj_id,source,details FROM sack_events ORDER BY id DESC LIMIT ?', (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_sack_by_number(sack_number):
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT sack_number,timestamp,obj_id,source,details FROM sack_events WHERE sack_number = ?', (sack_number,))
    row = cur.fetchone()
    conn.close()
    return row


def record_interval(milestone, start_sack, end_sack, start_ts, end_ts, duration_seconds, rate_per_hour):
    conn = _conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO intervals(milestone,start_ts,end_ts,start_sack,end_sack,duration_seconds,rate_per_hour) VALUES(?,?,?,?,?,?,?)', (milestone, start_ts, end_ts, start_sack, end_sack, duration_seconds, rate_per_hour))
    conn.commit()
    conn.close()


def get_last_interval():
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT milestone,start_ts,end_ts,start_sack,end_sack,duration_seconds,rate_per_hour FROM intervals ORDER BY id DESC LIMIT 1')
    row = cur.fetchone()
    conn.close()
    return row
