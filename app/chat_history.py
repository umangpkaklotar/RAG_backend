import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "chat_history.db"


def _connect():
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect()) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conversations_user_updated
                ON conversations (user_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_conversation_created
                ON messages (conversation_id, created_at, id);
            """
        )


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _conversation(row):
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_conversation(user_id):
    conversation_id = str(uuid.uuid4())
    timestamp = _timestamp()
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO conversations (id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, user_id, "New Chat", timestamp, timestamp),
        )
        row = connection.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
    return _conversation(row)


def list_conversations(user_id):
    with closing(_connect()) as connection, connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
    return [_conversation(row) for row in rows]


def get_conversation(user_id, conversation_id):
    with closing(_connect()) as connection, connection:
        row = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ? AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()
    return _conversation(row) if row else None


def get_messages(user_id, conversation_id):
    with closing(_connect()) as connection, connection:
        owned = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if not owned:
            return None
        rows = connection.execute(
            """
            SELECT role, content, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at, id
            """,
            (conversation_id,),
        ).fetchall()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def delete_conversation(user_id, conversation_id):
    with closing(_connect()) as connection, connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
    return cursor.rowcount > 0


def save_exchange(user_id, conversation_id, question, answer, sources):
    timestamp = _timestamp()
    with closing(_connect()) as connection, connection:
        conversation = connection.execute(
            """
            SELECT title, EXISTS(
                SELECT 1 FROM messages
                WHERE conversation_id = conversations.id AND role = 'user'
            ) AS has_questions
            FROM conversations
            WHERE id = ? AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()
        if not conversation:
            return False
        if not conversation["has_questions"]:
            title = " ".join(question.split())[:80] or "New Chat"
        else:
            title = conversation["title"]
        connection.executemany(
            """
            INSERT INTO messages (conversation_id, role, content, sources, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (conversation_id, "user", question, "[]", timestamp),
                (conversation_id, "assistant", answer, json.dumps(sources), timestamp),
            ],
        )
        connection.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title, timestamp, conversation_id, user_id),
        )
    return True
