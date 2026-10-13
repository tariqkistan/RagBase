import os
from pathlib import Path

import tiktoken
from pypdf import PdfReader

from .models import Chunk, Document


def load_text_file(filepath: str) -> Document:
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    return Document(source=Path(filepath).name, content=content)


def load_pdf(filepath: str) -> Document:
    reader = PdfReader(filepath)
    content = ""
    for page in reader.pages:
        content += page.extract_text() + "\n"
    return Document(source=Path(filepath).name, content=content)


def chunk_document(doc: Document) -> list[Chunk]:
    encoding = tiktoken.get_encoding("cl100k_base")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    tokens = encoding.encode(doc.content)
    chunks = []

    start = 0
    chunk_index = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(
            Chunk(
                content=chunk_text,
                source=doc.source,
                chunk_index=chunk_index,
                metadata=doc.metadata,
            )
        )
        chunk_index += 1
        start += chunk_size - chunk_overlap

    return chunks


def load_and_chunk(filepath: str) -> list[Chunk]:
    if filepath.lower().endswith(".pdf"):
        doc = load_pdf(filepath)
    else:
        doc = load_text_file(filepath)
    return chunk_document(doc)
