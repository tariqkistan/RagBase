import os

from sqlalchemy import text

from .database import async_session_maker, embed_text
from .models import StoredChunk


async def retrieve(question: str, k: int) -> list[StoredChunk]:
    top_k = int(os.getenv("TOP_K", "4"))
    k = min(k, top_k)

    embedding = await embed_text(question)

    async with async_session_maker() as session:
        query = text(
            """
            SELECT id, content, source, chunk_index, metadata,
                   1 - (embedding <=> :embedding) as similarity
            FROM document_chunks
            ORDER BY embedding <=> :embedding
            LIMIT :k
            """
        )
        result = await session.execute(
            query, {"embedding": embedding, "k": k}
        )
        rows = result.fetchall()

        chunks = []
        for row in rows:
            chunks.append(
                StoredChunk(
                    id=row.id,
                    content=row.content,
                    source=row.source,
                    chunk_index=row.chunk_index,
                    metadata=row.metadata,
                    similarity=float(row.similarity),
                )
            )
        return chunks


async def retrieve_with_filter(
    question: str, source_filter: str, k: int
) -> list[StoredChunk]:
    top_k = int(os.getenv("TOP_K", "4"))
    k = min(k, top_k)

    embedding = await embed_text(question)

    async with async_session_maker() as session:
        query = text(
            """
            SELECT id, content, source, chunk_index, metadata,
                   1 - (embedding <=> :embedding) as similarity
            FROM document_chunks
            WHERE source = :source_filter
            ORDER BY embedding <=> :embedding
            LIMIT :k
            """
        )
        result = await session.execute(
            query, {"embedding": embedding, "source_filter": source_filter, "k": k}
        )
        rows = result.fetchall()

        chunks = []
        for row in rows:
            chunks.append(
                StoredChunk(
                    id=row.id,
                    content=row.content,
                    source=row.source,
                    chunk_index=row.chunk_index,
                    metadata=row.metadata,
                    similarity=float(row.similarity),
                )
            )
        return chunks
