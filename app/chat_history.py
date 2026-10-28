import logging
import threading
import uuid
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from . import config

_client = None
_database = None
_initialization_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def _collections():
    if _database is None:
        raise RuntimeError("MongoDB chat history has not been initialized.")
    return _database["conversations"], _database["messages"]


def initialize_database():
    global _client, _database
    with _initialization_lock:
        if _database is not None:
            return
        uri = config.MONGODB_URL
        database_name = config.MONGODB_DB_NAME
        if not uri or not database_name:
            raise RuntimeError("Set MONGODB_URL and MONGODB_DB_NAME in backend/.env.")

        client = MongoClient(uri)
        try:
            client.admin.command("ping")
            database = client[database_name]
            database["conversations"].create_index(
                [("user_id", 1), ("updated_at", -1), ("_id", -1)],
                name="conversations_user_updated",
            )
            database["messages"].create_index(
                [("conversation_id", 1), ("created_at", 1), ("_id", 1)],
                name="messages_conversation_created",
            )
        except Exception as exc:
            client.close()
            if isinstance(exc, PyMongoError):
                raise RuntimeError("Could not initialize chat history in MongoDB.") from exc
            raise
        _client = client
        _database = database
        _logger.info("Chat history MongoDB initialized.")


def close_database():
    global _client, _database
    with _initialization_lock:
        if _client is not None:
            _client.close()
        _client = None
        _database = None


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _conversation(document):
    return {
        "id": document["_id"],
        "title": document["title"],
        "created_at": document["created_at"],
        "updated_at": document["updated_at"],
    }


def create_conversation(user_id):
    conversation_id = str(uuid.uuid4())
    timestamp = _timestamp()
    conversations, _ = _collections()
    document = {
        "_id": conversation_id,
        "user_id": user_id,
        "title": "New Chat",
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    conversations.insert_one(document)
    return _conversation(document)


def list_conversations(user_id):
    conversations, _ = _collections()
    rows = conversations.find({"user_id": user_id}).sort(
        [("updated_at", -1), ("_id", -1)]
    )
    return [_conversation(row) for row in rows]


def get_conversation(user_id, conversation_id):
    conversations, _ = _collections()
    row = conversations.find_one({"_id": conversation_id, "user_id": user_id})
    return _conversation(row) if row else None


def get_messages(user_id, conversation_id):
    conversations, messages = _collections()
    if not conversations.find_one(
        {"_id": conversation_id, "user_id": user_id}, {"_id": 1}
    ):
        return None
    rows = messages.find({"conversation_id": conversation_id}).sort(
        [("created_at", 1), ("_id", 1)]
    )
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": row["sources"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def delete_conversation(user_id, conversation_id):
    conversations, messages = _collections()
    result = conversations.delete_one({"_id": conversation_id, "user_id": user_id})
    if not result.deleted_count:
        return False
    messages.delete_many({"conversation_id": conversation_id})
    return True


def save_exchange(user_id, conversation_id, question, answer, sources):
    conversations, messages = _collections()
    conversation = conversations.find_one(
        {"_id": conversation_id, "user_id": user_id},
        {"title": 1},
    )
    if not conversation:
        return False

    has_questions = messages.find_one(
        {"conversation_id": conversation_id, "role": "user"}, {"_id": 1}
    )
    title = conversation["title"]
    if not has_questions:
        title = " ".join(question.split())[:80] or "New Chat"

    timestamp = _timestamp()
    messages.insert_many(
        [
            {
                "conversation_id": conversation_id,
                "role": "user",
                "content": question,
                "sources": [],
                "created_at": timestamp,
            },
            {
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "created_at": timestamp,
            },
        ]
    )
    conversations.update_one(
        {"_id": conversation_id, "user_id": user_id},
        {"$set": {"title": title, "updated_at": timestamp}},
    )
    return True
