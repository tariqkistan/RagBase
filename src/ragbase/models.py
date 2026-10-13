from typing import Any

from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredChunk(BaseModel):
    id: int
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any]
    similarity: float | None = None


class QueryRequest(BaseModel):
    question: str
    k: int = Field(default=4, ge=1, le=20)

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("k must be between 1 and 20")
        return v


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[StoredChunk]


class IngestRequest(BaseModel):
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalQuery(BaseModel):
    query_id: str
    question: str
    relevant_sources: list[str]
    relevant_content_keywords: list[str]
