import asyncio
import os

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .models import Chunk

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
engine = create_async_engine(os.getenv("DATABASE_URL", ""))
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def setup_database() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    source VARCHAR(512),
                    chunk_index INT,
                    metadata JSONB,
                    embedding vector(1536)
                )
                """
            )
        )
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ivfflat ON document_chunks
                USING ivfflat (embedding vector_cosine_ops)
                """
            )
        )


async def embed_text(text: str) -> list[float]:
    response = await client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        input=text,
    )
    return response.data[0].embedding


async def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    tasks = [embed_text(chunk.content) for chunk in chunks]
    embeddings = await asyncio.gather(*tasks)
    return list(zip(chunks, embeddings))


async def store_chunks(chunks_with_embeddings: list[tuple[Chunk, list[float]]]) -> int:
    async with async_session_maker() as session:
        query = text(
            """
            INSERT INTO document_chunks (content, source, chunk_index, metadata, embedding)
            VALUES (:content, :source, :chunk_index, :metadata, :embedding)
            """
        )
        for chunk, embedding in chunks_with_embeddings:
            await session.execute(
                query,
                {
                    "content": chunk.content,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                },
            )
        await session.commit()
        return len(chunks_with_embeddings)
