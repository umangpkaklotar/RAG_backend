import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config
from .auth import get_authenticated_user_id
from . import chat_history
from .rag_service import ask_stream, index_files


@asynccontextmanager
async def lifespan(_app):
    chat_history.initialize_database()
    try:
        yield
    finally:
        chat_history.close_database()


app = FastAPI(title="General Document RAG Chatbot API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "General Document RAG Chatbot"}


@app.post("/api/conversations")
def create_conversation(user_id: str = Depends(get_authenticated_user_id)):
    return chat_history.create_conversation(user_id)


@app.get("/api/conversations")
def list_conversations(user_id: str = Depends(get_authenticated_user_id)):
    return {"conversations": chat_history.list_conversations(user_id)}


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str,
    user_id: str = Depends(get_authenticated_user_id),
):
    messages = chat_history.get_messages(user_id, conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": messages}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_authenticated_user_id),
):
    if not chat_history.delete_conversation(user_id, conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted"}


@app.post("/api/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    user_id: str = Depends(get_authenticated_user_id),
):
    paths = []
    try:
        for upload in files:
            filename = Path(upload.filename or "document").name
            suffix = Path(filename).suffix.lower()
            if suffix not in config.SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
            content = await upload.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"Empty file: {filename}")
            temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temporary.write(content)
            temporary.close()
            paths.append((temporary.name, filename))
        result = index_files(paths, user_id=user_id)
        return {"status": "indexed", "files": [name for _, name in paths], **result}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {exc}") from exc
    finally:
        for path, _ in paths:
            try:
                os.remove(path)
            except OSError:
                pass


@app.post("/api/ask")
def ask(request: AskRequest, user_id: str = Depends(get_authenticated_user_id)):
    conversation_id = request.conversation_id
    if conversation_id:
        if not chat_history.get_conversation(user_id, conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation_id = chat_history.create_conversation(user_id)["id"]

    def event_stream():
        answer_parts = []
        try:
            for event in ask_stream(request.question, user_id=user_id):
                if event["type"] == "token":
                    answer_parts.append(event["text"])
                elif event["type"] == "done":
                    answer = "".join(answer_parts).strip()
                    if not answer:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'The AI service returned an empty answer.'})}\n\n"
                        return
                    if not chat_history.save_exchange(
                        user_id,
                        conversation_id,
                        request.question.strip(),
                        answer,
                        event.get("sources", []),
                    ):
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Conversation not found.'})}\n\n"
                        return
                    event["conversation_id"] = conversation_id
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
