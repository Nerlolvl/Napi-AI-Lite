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

    def __init__(
        self,
        database_path: str,
        max_note_length: int = 1000,
        rule_ttl_days: float = 30.0,
        max_rules_total: int = 200,
    ) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._max_note_length = max_note_length
        self._rule_ttl_seconds = max(0.0, rule_ttl_days) * 86400
        self._max_rules_total = max(1, max_rules_total)
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
                    created_at REAL NOT NULL,
                    normalized TEXT NOT NULL DEFAULT '',
                    hits INTEGER NOT NULL DEFAULT 0,
                    last_used_at REAL,
                    score REAL NOT NULL DEFAULT 1.0
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
            self._migrate_rules_table(conn)
            conn.commit()

    def _migrate_rules_table(self, conn: sqlite3.Connection) -> None:
        """Bring legacy reflection_diary schemas up to the hygiene columns.

        Idempotent: adds missing columns, backfills `normalized`, removes
        duplicate rules left by the old append-only writer, and enforces
        uniqueness so future inserts can deduplicate.
        """
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(reflection_diary)")}
        if "normalized" not in columns:
            conn.execute("ALTER TABLE reflection_diary ADD COLUMN normalized TEXT NOT NULL DEFAULT ''")
        if "hits" not in columns:
            conn.execute("ALTER TABLE reflection_diary ADD COLUMN hits INTEGER NOT NULL DEFAULT 0")
        if "last_used_at" not in columns:
            conn.execute("ALTER TABLE reflection_diary ADD COLUMN last_used_at REAL")
        if "score" not in columns:
            conn.execute("ALTER TABLE reflection_diary ADD COLUMN score REAL NOT NULL DEFAULT 1.0")

        conn.execute("UPDATE reflection_diary SET normalized = lower(trim(rule)) WHERE normalized = ''")
        conn.execute(
            "DELETE FROM reflection_diary WHERE id NOT IN "
            "(SELECT MIN(id) FROM reflection_diary GROUP BY normalized)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_reflection_normalized "
            "ON reflection_diary(normalized)"
        )

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

    def add_reflected_rule(self, rule: str, source: str = "teacher", category: str = "general") -> bool:
        """Store a reflected rule, or reinforce it if an equivalent exists.

        Deduplication is case-insensitive (trimmed). Reinforcing bumps the
        score (capped) and refreshes the TTL instead of creating a duplicate.
        Returns True when a new rule was created.
        """
        clean_rule = rule.strip()[: self._max_note_length]
        if not clean_rule:
            return False

        now = time()
        normalized = clean_rule.lower()
        with self._lock:
            conn = self._get_conn()
            existing = conn.execute(
                "SELECT id FROM reflection_diary WHERE normalized = ?",
                (normalized,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO reflection_diary "
                    "(rule, source, category, created_at, normalized, score) "
                    "VALUES (?, ?, ?, ?, ?, 1.0)",
                    (clean_rule, source, category, now, normalized),
                )
                created = True
            else:
                conn.execute(
                    "UPDATE reflection_diary SET "
                    "score = MIN(score + 0.25, 3.0), "
                    "last_used_at = ? "
                    "WHERE id = ?",
                    (now, existing["id"]),
                )
                created = False
            self._prune_rules(conn, now)
            conn.commit()
        return created

    def get_reflected_rules(self, category: str | None = None, limit: int = 8) -> list[str]:
        """Return active rules ordered by score and recency, tracking usage.

        A rule is active while `last_used_at` (or `created_at` for never-used
        rules) is inside the TTL window. Selected rules get their hit counter
        and `last_used_at` bumped, which keeps frequently used rules alive.
        """
        if limit <= 0:
            return []

        now = time()
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT id, rule FROM reflection_diary "
                "WHERE (? IS NULL OR category = ?) "
                "AND COALESCE(last_used_at, created_at) >= ? "
                "ORDER BY score DESC, COALESCE(last_used_at, created_at) DESC, id DESC "
                "LIMIT ?",
                (category, category, now - self._rule_ttl_seconds, limit),
            ).fetchall()
            if rows:
                ids = [row["id"] for row in rows]
                marks = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE reflection_diary SET hits = hits + 1, last_used_at = ? "
                    f"WHERE id IN ({marks})",
                    (now, *ids),
                )
                conn.commit()
        return [row["rule"] for row in rows]

    def reinforce_rules(self, rules: list[str], bonus: float = 0.1) -> int:
        """Reinforce rules that contributed to a good answer.

        Bumps score for rules matching the given texts (case-insensitive).
        Unknown texts are ignored — rules may also come from session notes
        and not exist in the diary. Returns the number of updated rows.
        """
        normalized = [rule.strip().lower() for rule in rules if rule.strip()]
        if not normalized or bonus <= 0:
            return 0
        with self._lock:
            conn = self._get_conn()
            marks = ",".join("?" * len(normalized))
            cursor = conn.execute(
                f"UPDATE reflection_diary SET score = MIN(score + ?, 3.0) "
                f"WHERE normalized IN ({marks})",
                (bonus, *normalized),
            )
            conn.commit()
            return cursor.rowcount

    def prune_rules(self) -> int:
        """Remove TTL-expired and excess rules; return how many were removed."""
        now = time()
        with self._lock:
            conn = self._get_conn()
            before = conn.execute("SELECT COUNT(*) AS n FROM reflection_diary").fetchone()["n"]
            self._prune_rules(conn, now)
            conn.commit()
            after = conn.execute("SELECT COUNT(*) AS n FROM reflection_diary").fetchone()["n"]
        return before - after

    def rule_stats(self) -> list[dict[str, Any]]:
        """Return every rule with its hygiene metadata (for diagnostics)."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT id, rule, source, category, hits, score, created_at, last_used_at "
                "FROM reflection_diary ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def _prune_rules(self, conn: sqlite3.Connection, now: float) -> None:
        """Delete expired rules and enforce the hard cap. Call within the lock."""
        conn.execute(
            "DELETE FROM reflection_diary "
            "WHERE COALESCE(last_used_at, created_at) < ?",
            (now - self._rule_ttl_seconds,),
        )
        conn.execute(
            "DELETE FROM reflection_diary WHERE id NOT IN ("
            "SELECT id FROM reflection_diary "
            "ORDER BY score DESC, COALESCE(last_used_at, created_at) DESC, id DESC LIMIT ?"
            ")",
            (self._max_rules_total,),
        )

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