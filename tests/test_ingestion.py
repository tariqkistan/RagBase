import os
import tempfile
from pathlib import Path

import pytest

from ragbase.ingestion import chunk_document, load_and_chunk, load_pdf, load_text_file
from ragbase.models import Document


def test_load_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test content")
        filepath = f.name

    try:
        doc = load_text_file(filepath)
        assert doc.content == "Test content"
        assert doc.source == Path(filepath).name
        assert doc.metadata == {}
    finally:
        os.unlink(filepath)


def test_chunk_document_produces_multiple_chunks():
    doc = Document(
        source="test.txt",
        content="This is a test document with enough content to be split into multiple chunks. " * 20,
        metadata={},
    )
    os.environ["CHUNK_SIZE"] = "50"
    os.environ["CHUNK_OVERLAP"] = "10"

    chunks = chunk_document(doc)
    assert len(chunks) > 1


def test_chunk_document_index_increments():
    doc = Document(
        source="test.txt",
        content="This is a test document with enough content to be split into multiple chunks. " * 20,
        metadata={},
    )
    os.environ["CHUNK_SIZE"] = "50"
    os.environ["CHUNK_OVERLAP"] = "10"

    chunks = chunk_document(doc)
    indices = [chunk.chunk_index for chunk in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_overlap_creates_shared_content():
    doc = Document(
        source="test.txt",
        content="This is a test document with enough content to be split into multiple chunks. " * 20,
        metadata={},
    )
    os.environ["CHUNK_SIZE"] = "50"
    os.environ["CHUNK_OVERLAP"] = "25"

    chunks = chunk_document(doc)
    if len(chunks) >= 2:
        chunk0_words = set(chunks[0].content.split())
        chunk1_words = set(chunks[1].content.split())
        overlap = chunk0_words & chunk1_words
        assert len(overlap) > 0


def test_load_pdf():
    pytest.importorskip("pypdf")
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(f)
        filepath = f.name

    try:
        doc = load_pdf(filepath)
        assert doc.source == Path(filepath).name
        assert isinstance(doc.content, str)
    finally:
        os.unlink(filepath)


def test_load_and_chunk_text():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test content for chunking. " * 20)
        filepath = f.name

    try:
        os.environ["CHUNK_SIZE"] = "50"
        os.environ["CHUNK_OVERLAP"] = "10"
        chunks = load_and_chunk(filepath)
        assert len(chunks) > 0
        assert all(chunk.source == Path(filepath).name for chunk in chunks)
    finally:
        os.unlink(filepath)


def test_load_and_chunk_pdf():
    pytest.importorskip("pypdf")
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(f)
        filepath = f.name

    try:
        os.environ["CHUNK_SIZE"] = "50"
        os.environ["CHUNK_OVERLAP"] = "10"
        chunks = load_and_chunk(filepath)
        assert len(chunks) > 0
        assert all(chunk.source == Path(filepath).name for chunk in chunks)
    finally:
        os.unlink(filepath)
