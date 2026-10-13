import json
import os

import pytest
from dotenv import load_dotenv

from ragbase.database import setup_database
from ragbase.ingestion import load_and_chunk
from ragbase.models import EvalQuery
from ragbase.retrieval import retrieve

load_dotenv()


@pytest.mark.eval
@pytest.mark.asyncio
async def test_retrieval_precision_recall():
    await setup_database()

    with open("tests/eval_queries/retrieval_evals.json") as f:
        eval_queries = [EvalQuery(**q) for q in json.load(f)]

    documents_dir = "documents"
    for filepath in os.listdir(documents_dir):
        full_path = os.path.join(documents_dir, filepath)
        if os.path.isfile(full_path):
            chunks = load_and_chunk(full_path)
            from ragbase.database import embed_chunks, store_chunks

            chunks_with_embeddings = await embed_chunks(chunks)
            await store_chunks(chunks_with_embeddings)

    precision_scores = []
    recall_scores = []

    for eval_query in eval_queries:
        retrieved_chunks = await retrieve(eval_query.question, k=4)

        retrieved_sources = {chunk.source for chunk in retrieved_chunks}
        relevant_sources = set(eval_query.relevant_sources)

        retrieved_keywords_found = []
        for keyword in eval_query.relevant_content_keywords:
            keyword_lower = keyword.lower()
            for chunk in retrieved_chunks:
                if keyword_lower in chunk.content.lower():
                    retrieved_keywords_found.append(keyword)
                    break

        retrieved_keywords = set(retrieved_keywords_found)
        relevant_keywords = set(eval_query.relevant_content_keywords)

        if len(retrieved_sources) > 0:
            source_precision = len(retrieved_sources & relevant_sources) / len(
                retrieved_sources
            )
        else:
            source_precision = 0.0

        if len(relevant_sources) > 0:
            source_recall = len(retrieved_sources & relevant_sources) / len(
                relevant_sources
            )
        else:
            source_recall = 0.0

        if len(retrieved_keywords) > 0:
            keyword_precision = len(retrieved_keywords & relevant_keywords) / len(
                retrieved_keywords
            )
        else:
            keyword_precision = 0.0

        if len(relevant_keywords) > 0:
            keyword_recall = len(retrieved_keywords & relevant_keywords) / len(
                relevant_keywords
            )
        else:
            keyword_recall = 0.0

        precision = (source_precision + keyword_precision) / 2
        recall = (source_recall + keyword_recall) / 2

        precision_scores.append(precision)
        recall_scores.append(recall)

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    assert avg_precision >= 0.65, f"Average precision {avg_precision:.2f} below threshold 0.65"
    assert avg_recall >= 0.65, f"Average recall {avg_recall:.2f} below threshold 0.65"
