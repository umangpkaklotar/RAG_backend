import json
import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config
from .auth import get_authenticated_user_id
from .rag_service import ask_stream, index_files

app = FastAPI(title="General Document RAG Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "General Document RAG Chatbot"}


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
    def event_stream():
        try:
            for event in ask_stream(request.question, user_id=user_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
