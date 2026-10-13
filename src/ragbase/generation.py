import os

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from .database import client
from .models import QueryResponse, StoredChunk


async def generate(question: str, chunks: list[StoredChunk]) -> QueryResponse:
    context_parts = []
    sources = set()

    for i, chunk in enumerate(chunks):
        context_parts.append(f"[Source: {chunk.source}]\n{chunk.content}")
        sources.add(chunk.source)

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context.
- Use the context to answer the question accurately.
- Cite the sources you used in your answer.
- If the answer cannot be found in the context, say "I don't have enough information to answer this question."
- Do not use outside knowledge or make assumptions beyond the context."""

    messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        messages=messages,
        temperature=0,
    )

    answer = response.choices[0].message.content or "No response generated"

    return QueryResponse(
        question=question,
        answer=answer,
        sources=list(sources),
        retrieved_chunks=chunks,
    )
