import os

import pytest
from pytest_mock import MockerFixture

from ragbase.retrieval import retrieve, retrieve_with_filter


@pytest.mark.asyncio
async def test_retrieve_returns_list(mocker: MockerFixture):
    mock_embedding = [0.1] * 1536
    mocker.patch("ragbase.retrieval.embed_text", return_value=mock_embedding)

    mock_rows = [
        type(
            "Row",
            (),
            {
                "id": 1,
                "content": "test content",
                "source": "test.txt",
                "chunk_index": 0,
                "metadata": {},
                "similarity": 0.95,
            },
        )()
    ]

    mock_result = mocker.MagicMock()
    mock_result.fetchall = mocker.MagicMock(return_value=mock_rows)
    mock_session = mocker.AsyncMock()
    mock_session.execute = mocker.AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch("ragbase.retrieval.async_session_maker", return_value=mock_session)

    os.environ["TOP_K"] = "4"
    chunks = await retrieve("test question", 2)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].content == "test content"
    assert chunks[0].source == "test.txt"
    assert chunks[0].similarity == 0.95


@pytest.mark.asyncio
async def test_retrieve_with_filter(mocker: MockerFixture):
    mock_embedding = [0.1] * 1536
    mocker.patch("ragbase.retrieval.embed_text", return_value=mock_embedding)

    mock_rows = [
        type(
            "Row",
            (),
            {
                "id": 1,
                "content": "test content",
                "source": "specific.txt",
                "chunk_index": 0,
                "metadata": {},
                "similarity": 0.95,
            },
        )()
    ]

    mock_result = mocker.MagicMock()
    mock_result.fetchall = mocker.MagicMock(return_value=mock_rows)
    mock_session = mocker.AsyncMock()
    mock_session.execute = mocker.AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch("ragbase.retrieval.async_session_maker", return_value=mock_session)

    os.environ["TOP_K"] = "4"
    chunks = await retrieve_with_filter("test question", "specific.txt", 2)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].source == "specific.txt"
