from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from .database import async_session_maker, embed_chunks, setup_database, store_chunks
from .generation import generate
from .ingestion import chunk_document
from .models import Document, IngestRequest, QueryRequest
from .retrieval import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_database()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/ingest")
async def ingest(request: IngestRequest) -> dict[str, int]:
    doc = Document(
        source=request.source,
        content=request.content,
        metadata=request.metadata,
    )
    chunks = chunk_document(doc)
    chunks_with_embeddings = await embed_chunks(chunks)
    count = await store_chunks(chunks_with_embeddings)
    return {"chunks_stored": count}


@app.post("/query")
async def query(request: QueryRequest):
    chunks = await retrieve(request.question, request.k)
    response = await generate(request.question, chunks)
    return response


@app.get("/chunks")
async def get_chunks() -> dict[str, int]:
    async with async_session_maker() as session:
        count_result = await session.execute(
            text("SELECT COUNT(*) FROM document_chunks")
        )
        total_chunks = count_result.scalar() or 0

        sources_result = await session.execute(
            text("SELECT COUNT(DISTINCT source) FROM document_chunks")
        )
        unique_sources = sources_result.scalar() or 0

        return {"total_chunks": total_chunks, "unique_sources": unique_sources}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}
