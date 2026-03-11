import json
import importlib.util
import os
import shutil
import time
import uuid
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader

from backend.rag.pipeline import answer_query, reload_index
from backend.rag.rebuild import rebuild_index

BANK_DOCS_DIR = os.path.abspath("bank_docs")
INDEX_DIR = os.path.abspath(os.path.join("indexes", "rag_index"))
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
CHAT_ATTACHMENT_DIR = os.path.abspath(os.path.join("data", "chat_uploads"))
CHAT_ATTACHMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}
MAX_ATTACHMENT_CHARS = 6000

_attachment_lock = Lock()
_attachment_store: dict[str, dict] = {}
HAS_MULTIPART = importlib.util.find_spec("multipart") is not None

app = FastAPI(title="Bank RAG Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def _read_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


def _extract_attachment_content(path: str, ext: str, filename: str) -> str:
    if ext in {".txt", ".md", ".csv", ".json"}:
        return _read_text_file(path).strip()
    if ext == ".pdf":
        return _read_pdf_file(path)
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return (
            f"Image attachment named {filename}. The backend prototype does not extract image text, "
            "so use the filename only unless the user describes the image in their message."
        )
    return ""


def _get_attachment_contexts(session_id: str, attachment_ids: list[str]) -> list[dict]:
    contexts: list[dict] = []
    with _attachment_lock:
        for attachment_id in attachment_ids:
            item = _attachment_store.get(attachment_id)
            if not item or item["session_id"] != session_id:
                continue
            contexts.append({"name": item["filename"], "content": item["content"]})
    return contexts


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    result = answer_query(req.message)
    return {"session_id": req.session_id or "default", **result}


@app.get("/chat/stream")
def chat_stream(
    message: str,
    session_id: str = "default",
    attachment_ids: list[str] = Query(default=[]),
):
    def event_generator():
        attachment_contexts = _get_attachment_contexts(session_id, attachment_ids)
        result = answer_query(message, attachment_contexts=attachment_contexts)
        text = result.get("answer", "")

        for word in text.split():
            yield f"data: {json.dumps({'delta': word + ' '})}\n\n"
            time.sleep(0.02)

        yield f"data: {json.dumps({'done': True, 'sources': result.get('sources', [])})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if HAS_MULTIPART:
    @app.post("/chat/attachments")
    async def upload_chat_attachment(
        session_id: str = Form(...),
        file: UploadFile = File(...),
    ):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in CHAT_ATTACHMENT_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(CHAT_ATTACHMENT_EXTENSIONS)}",
            )

        os.makedirs(CHAT_ATTACHMENT_DIR, exist_ok=True)
        attachment_id = uuid.uuid4().hex
        safe_name = os.path.basename(file.filename or f"attachment{ext}")
        dest_path = os.path.join(CHAT_ATTACHMENT_DIR, f"{attachment_id}{ext}")

        try:
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(file.file, out)
        finally:
            await file.close()

        content = _extract_attachment_content(dest_path, ext, safe_name)
        preview = content[:280].replace("\n", " ").strip()

        with _attachment_lock:
            _attachment_store[attachment_id] = {
                "id": attachment_id,
                "session_id": session_id,
                "filename": safe_name,
                "content": content[:MAX_ATTACHMENT_CHARS],
                "file_path": dest_path,
                "mime_type": file.content_type or "application/octet-stream",
            }

        return {
            "id": attachment_id,
            "name": safe_name,
            "mime_type": file.content_type or "application/octet-stream",
            "preview": preview,
        }


    @app.post("/admin/upload")
    async def upload_doc(file: UploadFile = File(...)):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

        os.makedirs(BANK_DOCS_DIR, exist_ok=True)
        dest_path = os.path.join(BANK_DOCS_DIR, file.filename)

        try:
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(file.file, out)
        finally:
            await file.close()

        try:
            chunks_count = rebuild_index(BANK_DOCS_DIR, INDEX_DIR)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        reload_index()

        return {"ok": True, "filename": file.filename, "chunks_count": chunks_count}
else:
    @app.post("/chat/attachments")
    async def upload_chat_attachment_unavailable():
        raise HTTPException(
            status_code=503,
            detail='File uploads require the optional "python-multipart" package on the backend.',
        )


    @app.post("/admin/upload")
    async def upload_doc_unavailable():
        raise HTTPException(
            status_code=503,
            detail='Document uploads require the optional "python-multipart" package on the backend.',
        )
