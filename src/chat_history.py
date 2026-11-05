from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path


@contextmanager
def _connection(db_path: str | Path):
    """Open a SQLite connection and always close it after the operation."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_chat_db(db_path: str | Path):
    """Create the SQLite table used to persist chat messages."""
    with _connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_session_id
            ON chat_messages(session_id, id)
            """
        )


def load_conversation(db_path: str | Path, session_id: str):
    """Load one conversation in chronological order."""
    with _connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT role, content, sources_json
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()

    messages = []
    for row in rows:
        try:
            sources = json.loads(row["sources_json"] or "[]")
        except json.JSONDecodeError:
            sources = []

        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "sources": sources,
            }
        )
    return messages


def save_message(db_path: str | Path, session_id: str, message: dict):
    """Append a user or assistant message to SQLite."""
    sources_json = json.dumps(message.get("sources", []), ensure_ascii=False)

    with _connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO chat_messages(session_id, role, content, sources_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                message["role"],
                message["content"],
                sources_json,
            ),
        )


def clear_conversation(db_path: str | Path, session_id: str):
    """Delete only the selected browser conversation."""
    with _connection(db_path) as connection:
        connection.execute(
            "DELETE FROM chat_messages WHERE session_id = ?",
            (session_id,),
        )
