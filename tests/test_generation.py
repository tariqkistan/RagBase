import os

import pytest
from pytest_mock import MockerFixture

from ragbase.generation import generate
from ragbase.models import StoredChunk


@pytest.mark.asyncio
async def test_generate_returns_query_response(mocker: MockerFixture):
    mock_chunks = [
        StoredChunk(
            id=1,
            content="Test content from source1",
            source="source1.txt",
            chunk_index=0,
            metadata={},
            similarity=0.95,
        ),
        StoredChunk(
            id=2,
            content="Test content from source2",
            source="source2.txt",
            chunk_index=0,
            metadata={},
            similarity=0.90,
        ),
    ]

    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.content = "Test answer based on context."

    mock_create = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("ragbase.generation.client.chat.completions.create", mock_create)

    os.environ["OPENAI_CHAT_MODEL"] = "gpt-4o"

    response = await generate("Test question", mock_chunks)

    assert response.question == "Test question"
    assert response.answer == "Test answer based on context."
    assert set(response.sources) == {"source1.txt", "source2.txt"}
    assert len(response.retrieved_chunks) == 2


@pytest.mark.asyncio
async def test_generate_includes_context_in_prompt(mocker: MockerFixture):
    mock_chunks = [
        StoredChunk(
            id=1,
            content="Test content",
            source="test.txt",
            chunk_index=0,
            metadata={},
            similarity=0.95,
        )
    ]

    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.content = "Answer"

    mock_create = mocker.AsyncMock(return_value=mock_response)
    mocker.patch("ragbase.generation.client.chat.completions.create", mock_create)

    os.environ["OPENAI_CHAT_MODEL"] = "gpt-4o"

    await generate("Test question", mock_chunks)

    call_args = mock_create.call_args
    messages = call_args.kwargs["messages"]
    user_message = messages[1]["content"]

    assert "[Source: test.txt]" in user_message
    assert "Test content" in user_message
    assert "Test question" in user_message
