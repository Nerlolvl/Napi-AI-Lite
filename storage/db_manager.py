from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from time import time
from typing import Any


class NapiBrain:
    """Unified SQLite database — memory, knowledge, and reflection diary.

    Uses a single persistent connection with WAL mode and memory-optimized
    pragmas for 2-core / 2 GB RAM operation.
    """

    def __init__(self, database_path: str, max_note_length: int = 1000) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._max_note_length = max_note_length
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
                return self._conn
            except sqlite3.Error:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-512")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=8388608")
        conn.execute("PRAGMA page_size=4096")
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL,
                    comment TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reflection_diary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'teacher',
                    category TEXT NOT NULL DEFAULT 'general',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES knowledge_docs(id)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
                    USING fts5(title, content, source);

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_notes_session
                    ON notes(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_reflection_rule
                    ON reflection_diary(category);
                """
            )
            conn.commit()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, time(), json.dumps(metadata or {})),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, str]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def add_note(self, session_id: str, note: str) -> None:
        clean_note = note.strip()[: self._max_note_length]
        if not clean_note:
            return
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO notes (session_id, note, created_at) VALUES (?, ?, ?)",
                (session_id, clean_note, time()),
            )
            conn.commit()

    def recent_notes(self, session_id: str, limit: int) -> list[str]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT note FROM notes WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [row["note"] for row in rows]

    def add_feedback(
        self,
        session_id: str,
        message_id: int,
        rating: int,
        comment: str,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO feedback (session_id, message_id, rating, comment, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, message_id, rating, comment[: self._max_note_length], time()),
            )
            conn.commit()
        if comment.strip():
            self.add_note(session_id, f"User feedback: rating={rating}; {comment.strip()}")

    def add_reflected_rule(self, rule: str, source: str = "teacher", category: str = "general") -> None:
        clean_rule = rule.strip()[: self._max_note_length]
        if not clean_rule:
            return
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO reflection_diary (rule, source, category, created_at) "
                "VALUES (?, ?, ?, ?)",
                (clean_rule, source, category, time()),
            )
            conn.commit()

    def get_reflected_rules(self, category: str | None = None, limit: int = 8) -> list[str]:
        with self._lock:
            conn = self._get_conn()
            if category:
                rows = conn.execute(
                    "SELECT rule FROM reflection_diary WHERE category = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT rule FROM reflection_diary ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [row["rule"] for row in rows]

    def upsert_document(self, source: str, title: str, content: str) -> int:
        chunks = _chunk_text(content)
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM knowledge_fts WHERE rowid IN "
                         "(SELECT k.id FROM knowledge_chunks k WHERE k.doc_id = "
                         "(SELECT d.id FROM knowledge_docs d WHERE d.source = ?))",
                         (source,))
            conn.execute("DELETE FROM knowledge_chunks WHERE doc_id = "
                         "(SELECT id FROM knowledge_docs WHERE source = ?)", (source,))
            conn.execute("DELETE FROM knowledge_docs WHERE source = ?", (source,))

            cursor = conn.execute(
                "INSERT INTO knowledge_docs (source, title, created_at) VALUES (?, ?, ?)",
                (source, title, time()),
            )
            doc_id = int(cursor.lastrowid)

            for index, chunk in enumerate(chunks):
                cursor2 = conn.execute(
                    "INSERT INTO knowledge_chunks (doc_id, chunk_index, title, content, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (doc_id, index, title, chunk, time()),
                )
                chunk_id = int(cursor2.lastrowid)
                conn.execute(
                    "INSERT INTO knowledge_fts (rowid, title, content, source) VALUES (?, ?, ?, ?)",
                    (chunk_id, title, chunk, source),
                )
            conn.commit()
        return len(chunks)

    def search_knowledge(self, query: str, limit: int = 6) -> list[dict[str, str]]:
        clean_query = _prepare_fts_query(query)
        if not clean_query:
            return []

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT c.title, c.content, d.source "
                "FROM knowledge_fts f "
                "JOIN knowledge_chunks c ON c.id = f.rowid "
                "JOIN knowledge_docs d ON d.id = c.doc_id "
                "WHERE knowledge_fts MATCH ? "
                "ORDER BY bm25(knowledge_fts) LIMIT ?",
                (clean_query, limit),
            ).fetchall()

        return [
            {"title": row["title"], "content": row["content"], "source": row["source"]}
            for row in rows
        ]

    def ingest_directory(self, directory: str) -> int:
        root = Path(directory)
        if not root.exists():
            return 0

        total = 0
        supported_suffixes = {".md", ".txt"}
        for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in supported_suffixes):
            text = path.read_text(encoding="utf-8")
            title = _first_heading(text) or path.stem.replace("_", " ").title()
            total += self.upsert_document(str(path), title, text)
        return total


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            return clean.lstrip("#").strip()
    return None


def _chunk_text(text: str, max_chars: int = 2800, overlap: int = 350) -> list[str]:
    clean = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        if end < len(clean):
            paragraph_break = clean.rfind("\n\n", start, end)
            if paragraph_break > start + 800:
                end = paragraph_break
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def _prepare_fts_query(query: str) -> str:
    terms = []
    for raw in query.replace('"', " ").replace("'", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in ("_", "-"))
        if len(token) >= 3:
            terms.append(token)
    return " OR ".join(terms[:24])