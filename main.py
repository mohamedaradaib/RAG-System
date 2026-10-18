import logging
import secrets
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_file, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
from rag_utils import build_rag_prompt

load_dotenv()


def _friendly_openai_error(exc: Exception) -> str:
    message = str(exc).lower()
    if any(token in message for token in ["credit_balance_exhausted", "insufficient_quota", "429", "quota", "rate limit"]):
        return "The OpenAI API is unavailable because the account has no remaining credits. Add billing credits or use a different API key."
    if "api key" in message or "authentication" in message:
        return "The OpenAI API key is missing or invalid. Check your environment configuration."
    return f"The AI service failed: {exc}"


inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=os.getenv("INNGEST_IS_PRODUCTION", "false").lower() == "true",
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_document"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        file_path = ctx.event.data["file_path"]
        source_id = ctx.event.data.get("source_id", file_path)
        chunks = load_and_chunk_file(file_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        try:
            chunks = chunks_and_src.chunks
            source_id = chunks_and_src.source_id
            vecs = embed_texts([chunk.text for chunk in chunks])
            ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
            payloads = [
                {
                    "source": source_id,
                    "text": chunk.text,
                    "chunk_index": i,
                    **({"page": chunk.page} if chunk.page is not None else {}),
                }
                for i, chunk in enumerate(chunks)
            ]
            QdrantStorage().replace_source(ids, vecs, payloads, source_id)
            return RAGUpsertResult(ingested=len(chunks))
        except Exception as exc:
            return RAGUpsertResult(ingested=0, error=_friendly_openai_error(exc))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    try:
        Path(ctx.event.data["file_path"]).unlink(missing_ok=True)
    except OSError:
        logging.getLogger("uvicorn").warning("Could not remove uploaded file")
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        try:
            query_vec = embed_texts([question])[0]
            store = QdrantStorage()
            found = store.search(query_vec, top_k, query_text=question)
            return RAGSearchResult(
                contexts=found["contexts"],
                sources=found["sources"],
                citations=found["citations"],
            )
        except Exception as exc:
            return RAGSearchResult(contexts=[], sources=[], error=_friendly_openai_error(exc))

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    chat_history = ctx.event.data.get("chat_history", []) or []

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    if found.error:
        return {
            "answer": found.error,
            "sources": [],
            "num_contexts": 0,
            "error": found.error,
        }

    if not found.contexts:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": [],
            "num_contexts": 0,
        }

    context_block = "\n\n".join(
        f"[{citation.id}] Source: {citation.source}"
        + (
            f", page {citation.page}"
            if citation.page is not None
            else f", chunk {citation.chunk_index + 1}"
            if citation.chunk_index is not None
            else ""
        )
        + f"\n{citation.text}"
        for citation in found.citations
    )
    user_content = build_rag_prompt(question, context_block, chat_history)

    try:
        adapter = ai.openai.Adapter(
            auth_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

        res = await ctx.step.ai.infer(
            "llm-answer",
            adapter=adapter,
            body={
                "max_tokens": 1024,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "Answer only from the provided context and previous conversation when relevant. Cite supporting context using its [number]. Never invent a citation."},
                    {"role": "user", "content": user_content}
                ]
            }
        )

        answer = res["choices"][0]["message"]["content"].strip()
        return {
            "answer": answer,
            "sources": found.sources,
            "citations": [citation.model_dump() for citation in found.citations],
            "num_contexts": len(found.contexts),
        }
    except Exception as exc:
        return {
            "answer": _friendly_openai_error(exc),
            "sources": found.sources,
            "citations": [citation.model_dump() for citation in found.citations],
            "num_contexts": len(found.contexts),
            "error": _friendly_openai_error(exc),
        }

app = FastAPI()


@app.post("/upload")
async def upload_document(request: Request):
    api_token = os.getenv("APP_API_TOKEN")
    is_production = os.getenv("INNGEST_IS_PRODUCTION", "false").lower() == "true"
    if is_production and not api_token:
        raise HTTPException(status_code=503, detail="APP_API_TOKEN is not configured")
    if api_token and not secrets.compare_digest(
        request.headers.get("authorization", ""), f"Bearer {api_token}"
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")

    filename = Path(unquote(request.headers.get("x-file-name", "")).replace("\\", "/")).name
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid file name is required")

    contents = await request.body()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4().hex}-{filename}"
    file_path.write_bytes(contents)

    try:
        result = await inngest_client.send(
            inngest.Event(
                name="rag/ingest_document",
                data={"file_path": str(file_path.resolve()), "source_id": filename},
            )
        )
    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    return {"event_id": result[0]}


inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])