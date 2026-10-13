# RAGbase

[![CI Status](https://github.com/tariqkistan/RagBase/actions/workflows/ci.yml/badge.svg)](https://github.com/tariqkistan/RagBase/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-quality Retrieval-Augmented Generation (RAG) pipeline built with Python, OpenAI embeddings, pgvector, and FastAPI. Includes a comprehensive retrieval evaluation suite measuring precision and recall.

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI architecture that enhances large language models by retrieving relevant context from a knowledge base before generating responses. This approach:

- Reduces hallucinations by grounding responses in factual documents
- Enables answering questions about specific, domain-specific information
- Provides source attribution for transparency and verification
- Allows updating knowledge without retraining the model

## Architecture

### Ingestion Flow

```
Document (PDF/Text)
    ↓
Load & Parse
    ↓
Chunk (tiktoken, 500 tokens, 50 overlap)
    ↓
Embed (OpenAI text-embedding-3-small, 1536-dim)
    ↓
Store (PostgreSQL + pgvector with IVFFlat index)
```

### Query Flow

```
User Question
    ↓
Embed (OpenAI text-embedding-3-small)
    ↓
Retrieve (pgvector cosine similarity, top-k)
    ↓
Generate (OpenAI GPT-4o with context)
    ↓
Response with sources
```

## Quickstart

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL with pgvector)
- OpenAI API key

### Setup

1. Clone the repository and navigate to the project directory

2. Start PostgreSQL with pgvector:
```bash
docker-compose up -d
```

3. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

### Ingest Documents

```bash
python scripts/ingest_documents.py
```

This loads all files from the `documents/` directory, chunks them, embeds with OpenAI, and stores in pgvector.

### Query the Pipeline

```bash
python scripts/query.py "What should I say when I first answer a call from a debt collector?"
```

Or use the interactive CLI with custom k:
```bash
python scripts/query.py "What are my rights under the National Credit Act?" --k 6
```

### Start the API Server

```bash
uvicorn ragbase.api:app --reload
```

API endpoints:
- `POST /ingest` - Ingest a document
- `POST /query` - Query the RAG pipeline
- `GET /chunks` - Get chunk statistics
- `GET /health` - Health check

## Evaluation

The project includes a retrieval evaluation suite measuring precision and recall.

### Running Evals

```bash
pytest tests/test_evals.py -m eval
```

### Eval Results

| Query ID | Question | Precision | Recall |
|----------|----------|-----------|--------|
| eval_001 | What should I say when I first answer a call from a debt collector? | ≥0.65 | ≥0.65 |
| eval_002 | How can I verify that a debt is legitimate? | ≥0.65 | ≥0.65 |
| eval_003 | What phrases are debt collectors forbidden from using? | ≥0.65 | ≥0.65 |
| eval_004 | What are the POPIA consent requirements for debt collection? | ≥0.65 | ≥0.65 |
| eval_005 | What happens at the end of a debt collection process? | ≥0.65 | ≥0.65 |

**Note:** Actual precision/recall values depend on the quality of embeddings and chunking. Run the eval suite after ingesting documents to see your specific results.

### Eval Methodology

Each eval query includes:
- `query_id`: Unique identifier
- `question`: The user question
- `relevant_sources`: Expected document sources that should be retrieved
- `relevant_content_keywords`: Keywords that should appear in retrieved chunks

Precision measures how many retrieved chunks are relevant, while recall measures how many relevant chunks were retrieved. Both metrics average source-level and keyword-level matching.

## Development

### Running Tests

```bash
# Unit tests (fast, mocked)
pytest tests/ -v -m "not eval"

# Full test suite including evals (slow, requires real API)
pytest tests/ -v
```

### Linting

```bash
ruff check src/ tests/ scripts/
ruff format src/ tests/ scripts/
```

### Type Checking

```bash
mypy src/ragbase/
```

## Design Decisions

### Why pgvector over specialized vector databases?

**PostgreSQL + pgvector** was chosen over specialized vector databases (Pinecone, Weaviate, Milvus) for several reasons:

1. **Simplicity**: Single database for both structured data and vectors
2. **Cost**: No additional hosting costs or API fees
3. **Performance**: IVFFlat index provides efficient approximate nearest neighbor search
4. **Integration**: Easy to join vector results with relational data
5. **Self-hosted**: Full control over data and no vendor lock-in
6. **SQL familiarity**: Leverages existing SQL skills and tooling

For production at scale, consider:
- Increasing IVFFlat list size for better recall
- Using HNSW index for faster approximate search
- Partitioning tables by source or date
- Adding connection pooling with PgBouncer

### Why text-embedding-3-small over larger models?

**OpenAI text-embedding-3-small** (1536 dimensions) was chosen because:

1. **Cost**: Significantly cheaper than ada-002 or larger models
2. **Speed**: Faster embedding generation
3. **Size**: Smaller embeddings reduce storage and memory
4. **Quality**: Excellent performance for most use cases
5. **Latest**: Current generation model with better multilingual support

For specialized domains, consider fine-tuning embeddings or using domain-specific models.

### Why chunking with overlap?

**Sliding window chunking with overlap** (500 tokens, 50 overlap) provides:

1. **Context preservation**: Important information less likely to be split
2. **Boundary flexibility**: Queries can match content across chunk boundaries
3. **Redundancy**: Multiple retrieval paths to the same information
4. **Balance**: Reasonable chunk size for context window limits

Alternative approaches considered:
- Fixed-size without overlap: Faster but loses boundary context
- Semantic chunking: Better coherence but requires additional processing
- Document-level embedding: Preserves full context but loses granularity

### Why FastAPI over Flask/Django?

**FastAPI** was chosen for the API layer because:

1. **Async native**: Built-in async support for I/O operations
2. **Type safety**: Automatic OpenAPI docs from type hints
3. **Performance**: Comparable to Go and Node.js frameworks
4. **Validation**: Pydantic integration for request/response validation
5. **Modern**: Designed for modern Python async patterns

### Why temperature=0 for generation?

**Zero temperature** ensures deterministic, factual responses:

1. **Consistency**: Same question produces same answer
2. **Factual**: Reduces hallucinations and creative embellishment
3. **Grounded**: Encourages reliance on provided context
4. **Debugging**: Easier to test and reproduce issues

For conversational applications, consider temperature 0.1-0.3 for slight variation while maintaining factual accuracy.

## Project Structure

```
ragbase/
├── documents/                  # Source documents (PDF, text)
│   ├── scorecard_rubrics.txt
│   ├── compliance_guide.txt
│   └── faq.txt
├── scripts/                    # Utility scripts
│   ├── ingest_documents.py    # Bulk document ingestion
│   └── query.py               # Interactive CLI querying
├── src/ragbase/               # Main package
│   ├── __init__.py
│   ├── api.py                 # FastAPI endpoints
│   ├── database.py           # Database setup & embeddings
│   ├── generation.py          # Answer generation
│   ├── ingestion.py          # Document loading & chunking
│   ├── models.py              # Pydantic models
│   └── retrieval.py          # Vector similarity search
├── tests/                     # Test suite
│   ├── eval_queries/          # Evaluation queries
│   │   └── retrieval_evals.json
│   ├── test_evals.py          # Retrieval eval tests
│   ├── test_generation.py    # Generation tests
│   ├── test_ingestion.py      # Ingestion tests
│   └── test_retrieval.py      # Retrieval tests
├── .github/workflows/         # CI configuration
│   └── ci.yml
├── docker-compose.yml         # PostgreSQL service
├── pyproject.toml            # Project configuration
├── ruff.toml                  # Linting configuration
├── mypy.ini                   # Type checking configuration
└── README.md                  # This file
```

## License

MIT
