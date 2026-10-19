"""Lịch sử các lần tạo giọng (SQLite có sẵn trong Python)."""
import datetime as dt
import json
import sqlite3

from .config import HISTORY_DB, ensure_dirs

_SCHEMA = """CREATE TABLE IF NOT EXISTS gen (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, mode TEXT, voice_slug TEXT, voice_name TEXT,
  text TEXT, settings TEXT, file TEXT, duration REAL, sim REAL, take INTEGER, best INTEGER)"""


def _db():
    ensure_dirs()
    con = sqlite3.connect(HISTORY_DB)
    con.execute(_SCHEMA)
    return con


def add(mode, voice_slug, voice_name, text, settings, file, duration, sim, take, best) -> None:
    with _db() as con:
        con.execute("INSERT INTO gen (created, mode, voice_slug, voice_name, text, settings, file, duration, sim, take, best) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (dt.datetime.now().isoformat(timespec="seconds"), mode, voice_slug, voice_name, text,
                     json.dumps(settings, ensure_ascii=False), str(file), duration, sim, take, int(best)))


def rows(voice_slug: str | None = None, limit: int = 300) -> list[dict]:
    with _db() as con:
        con.row_factory = sqlite3.Row
        q = "SELECT * FROM gen" + (" WHERE voice_slug = ?" if voice_slug else "") + " ORDER BY id DESC LIMIT ?"
        args = (voice_slug, limit) if voice_slug else (limit,)
        return [dict(r) for r in con.execute(q, args)]


def get(row_id: int) -> dict | None:
    with _db() as con:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM gen WHERE id = ?", (row_id,)).fetchone()
        return dict(r) if r else None


def remove(row_id: int) -> None:
    with _db() as con:
        con.execute("DELETE FROM gen WHERE id = ?", (row_id,))
