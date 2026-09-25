"""Shared key/value persistence -- SQLite locally, Postgres automatically
when DATABASE_URL is set (e.g. deployed on Railway, linked to its Postgres
service). Same cache_store(key, data, computed_at, error) shape either way,
so callers never need to know which backend is actually active.

psycopg2 is only imported when DATABASE_URL is actually set, so a local
run with no Postgres connection configured never needs it installed --
this keeps local development exactly as stdlib-only as before.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
_DATABASE_URL = os.environ.get("DATABASE_URL")

# Local SQLite file -- unused (and untouched) once DATABASE_URL is set.
_SQLITE_PATH = ROOT / "rally_state.db"

_CREATE_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS cache_store ("
    "key TEXT PRIMARY KEY, data TEXT, computed_at DOUBLE PRECISION, error TEXT)"
)
_CREATE_TABLE_SQL_SQLITE = _CREATE_TABLE_SQL.replace("DOUBLE PRECISION", "REAL")


def _sqlite_conn():
    import sqlite3

    conn = sqlite3.connect(_SQLITE_PATH)
    conn.execute(_CREATE_TABLE_SQL_SQLITE)
    return conn


def _pg_conn():
    import psycopg2

    conn = psycopg2.connect(_DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def db_load(key: str) -> tuple[object | None, float, str | None]:
    if _DATABASE_URL:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT data, computed_at, error FROM cache_store WHERE key = %s", (key,))
                row = cur.fetchone()
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            row = conn.execute(
                "SELECT data, computed_at, error FROM cache_store WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None, 0.0, None
    data_json, computed_at, error = row
    return (json.loads(data_json) if data_json is not None else None), (computed_at or 0.0), error


def db_save(key: str, data: object, computed_at: float, error: str | None = None) -> None:
    payload = json.dumps(data)
    if _DATABASE_URL:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO cache_store (key, data, computed_at, error) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET data=excluded.data, computed_at=excluded.computed_at, error=excluded.error",
                    (key, payload, computed_at, error),
                )
            conn.commit()
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            conn.execute(
                "INSERT INTO cache_store (key, data, computed_at, error) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET data=excluded.data, computed_at=excluded.computed_at, error=excluded.error",
                (key, payload, computed_at, error),
            )
            conn.commit()
        finally:
            conn.close()
