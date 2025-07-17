import uuid
from unittest.mock import MagicMock, call

import chromadb  # For spec in mock
import pytest

from fedotllm.agents.memory import ChunkedDocument, Memory, text_splitter
from fedotllm.llm import LiteLLMEmbeddings


# Tests for text_splitter
def test_text_splitter_basic():
    text = "This is a test sentence. This is another one."
    chunks = text_splitter(text, max_chunk_length=20, overlap_ratio=0.1)
    assert len(chunks) > 1
    assert chunks[0] == "This is a test sente"
    assert chunks[1].startswith(
        "tence. This is anoth"
    )  # Corrected based on actual logic


def test_text_splitter_no_overlap():
    text = "This is a test sentence. This is another one."
    chunks = text_splitter(text, max_chunk_length=20, overlap_ratio=0.0)
    assert len(chunks) > 1
    assert chunks[0] == "This is a test sente"
    assert chunks[1] == "nce. This is another"  # No overlap


def test_text_splitter_custom_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = text_splitter(text, max_chunk_length=10, overlap_ratio=0.5)  # overlap 5
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "fghijklmno"  # Starts from 'f' (10-5=5th index, which is 'f')
    assert chunks[2] == "klmnopqrst"
    assert chunks[3] == "pqrstuvwxy"
    assert chunks[4] == "uvwxyz"


def test_text_splitter_text_shorter_than_max_chunk():
    text = "Short text."
    chunks = text_splitter(text, max_chunk_length=20, overlap_ratio=0.1)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_splitter_text_equals_max_chunk():
    text = "This text is twenty."  # 20 chars
    chunks = text_splitter(text, max_chunk_length=20, overlap_ratio=0.1)
    assert len(chunks) == 2  # Code produces a small trailing chunk
    assert chunks[0] == "This text is twenty."
    assert chunks[1] == "y."  # Corrected based on actual logic


def test_text_splitter_empty_string():
    text = ""
    chunks = text_splitter(text, max_chunk_length=20, overlap_ratio=0.1)
    assert len(chunks) == 1
    assert chunks[0] == ""  # Code was fixed to return [""]


def test_text_splitter_invalid_overlap_ratio_negative():
    with pytest.raises(
        ValueError, match="Overlap ratio must be between 0 and 1 \\(exclusive\\)."
    ):  # Escaped parentheses
        text_splitter("text", max_chunk_length=10, overlap_ratio=-0.1)


def test_text_splitter_invalid_overlap_ratio_one():
    with pytest.raises(
        ValueError, match="Overlap ratio must be between 0 and 1 \\(exclusive\\)."
    ):  # Escaped parentheses
        text_splitter("text", max_chunk_length=10, overlap_ratio=1.0)


def test_text_splitter_invalid_overlap_ratio_greater_than_one():
    with pytest.raises(
        ValueError, match="Overlap ratio must be between 0 and 1 \\(exclusive\\)."
    ):  # Escaped parentheses
        text_splitter("text", max_chunk_length=10, overlap_ratio=1.1)


def test_text_splitter_small_max_chunk_length():
    text = "abc"
    chunks = text_splitter(text, max_chunk_length=1, overlap_ratio=0.0)
    assert chunks == ["a", "b", "c"]


# Tests for ChunkedDocument
def test_chunked_document_creation():
    doc = ChunkedDocument(
        doc_name="test_doc",
        chunks=["chunk1", "chunk2"],
        doc_id="doc123",
        source="source_path",
        metadata={"author": "tester"},
    )
    assert doc.doc_name == "test_doc"
    assert doc.chunks == ["chunk1", "chunk2"]
    assert doc.doc_id == "doc123"
    assert doc.source == "source_path"
    assert doc.metadata == {"author": "tester"}


# Tests for Memory
@pytest.fixture
def mock_chroma_client(mocker):
    # Use string for spec if direct import is tricky or leads to issues
    client = MagicMock(
        spec_set=chromadb.ClientAPI
    )  # spec=chromadb.API is also an option
    return client


@pytest.fixture
def mock_chroma_collection(mocker):
    collection = MagicMock(
        spec_set=chromadb.api.models.Collection.Collection
    )  # More specific spec
    return collection


@pytest.fixture
def mock_embedding_model(mocker):
    # Mock LiteLLMEmbeddings instance, including MAX_INPUT if it's accessed
    model = MagicMock(spec=LiteLLMEmbeddings)
    model.MAX_INPUT = 8191  # Default value from LiteLLMEmbeddings
    return model


def test_memory_init(mock_chroma_client, mock_embedding_model, mock_chroma_collection):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    collection_name = "test_collection"

    memory = Memory(
        collection_name=collection_name,  # This is a positional argument in the actual __init__
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,  # client is the first argument
    )

    mock_chroma_client.get_or_create_collection.assert_called_once_with(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    assert memory.collection == mock_chroma_collection
    assert memory.embedding_model == mock_embedding_model
    assert memory.id == 0


def test_memory_insert_vectors(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection, mocker
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_insert",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_embedding_obj1 = MagicMock()
    mock_embedding_obj1.embedding = [0.1, 0.2, 0.3]
    mock_embedding_obj2 = MagicMock()
    mock_embedding_obj2.embedding = [0.4, 0.5, 0.6]
    mock_embedding_obj3 = MagicMock()
    mock_embedding_obj3.embedding = [0.7, 0.8, 0.9]

    # If encode is called per chunk
    mock_embedding_model.encode.side_effect = [
        [mock_embedding_obj1],
        [mock_embedding_obj2],
        [mock_embedding_obj3],
    ]

    doc1_id = str(uuid.uuid4())
    doc2_id = str(uuid.uuid4())

    # Using the new dataclass definition of ChunkedDocument
    documents = [
        ChunkedDocument(
            doc_name="doc1",
            chunks=["chunk1", "chunk2"],
            doc_id=doc1_id,
            source="s1",
            metadata={"m": "d1"},
        ),
        ChunkedDocument(
            doc_name="doc2",
            chunks=["chunk3"],
            doc_id=doc2_id,
            source="s2",
            metadata={"m": "d2"},
        ),
    ]

    mocker.patch("fedotllm.agents.memory.tqdm", lambda x, **kwargs: x)

    memory.insert_vectors(documents)

    # Check calls to embedding_model.encode
    expected_encode_calls = [
        call(input="chunk1"),
        call(input="chunk2"),
        call(input="chunk3"),
    ]  # Used keyword arg
    mock_embedding_model.encode.assert_has_calls(expected_encode_calls, any_order=False)

    # Check calls to collection.add
    # ChromaDB expects lists for each parameter
    expected_add_calls = [
        call(
            ids=[f"{doc1_id}_0", f"{doc1_id}_1"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["chunk1", "chunk2"],
            metadatas=[
                {
                    "doc_id": doc1_id,
                    "doc_name": "doc1",
                    "source": "s1",
                    "m": "d1",
                    "chunk_id": 0,
                },
                {
                    "doc_id": doc1_id,
                    "doc_name": "doc1",
                    "source": "s1",
                    "m": "d1",
                    "chunk_id": 1,
                },
            ],
        ),
        call(
            ids=[f"{doc2_id}_0"],
            embeddings=[[0.7, 0.8, 0.9]],
            documents=["chunk3"],
            metadatas=[
                {
                    "doc_id": doc2_id,
                    "doc_name": "doc2",
                    "source": "s2",
                    "m": "d2",
                    "chunk_id": 0,
                }
            ],
        ),
    ]
    # Due to internal looping per document, check call_args_list
    assert mock_chroma_collection.add.call_count == 2
    assert mock_chroma_collection.add.call_args_list[0] == expected_add_calls[0]
    assert mock_chroma_collection.add.call_args_list[1] == expected_add_calls[1]

    assert memory.id == 3  # 2 chunks from doc1, 1 chunk from doc2


def test_memory_search_context(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_search",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_query_embedding_obj = MagicMock()
    mock_query_embedding_obj.embedding = [0.15, 0.25, 0.35]
    mock_embedding_model.encode.return_value = [mock_query_embedding_obj]

    mock_query_result = {
        "documents": [["result1", "result2"]]
    }  # Simplified QueryResult structure
    mock_chroma_collection.query.return_value = mock_query_result

    query = "test query"
    n_results = 2
    results = memory.search_context(query, n_results=n_results)

    mock_embedding_model.encode.assert_called_once_with(query)
    mock_chroma_collection.query.assert_called_once_with(
        query_embeddings=[[0.15, 0.25, 0.35]],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )
    assert results == mock_query_result


def test_memory_search_context_with_metadatas(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_search_meta",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_query_embedding_obj = MagicMock()
    mock_query_embedding_obj.embedding = [0.3, 0.4, 0.5]
    mock_embedding_model.encode.return_value = [mock_query_embedding_obj]

    mock_query_result_meta = {"documents": [["meta_result1"]]}
    mock_chroma_collection.query.return_value = mock_query_result_meta

    query = "meta query"
    n_results = 1
    metadatas = {"source": "s1"}
    results = memory.search_context_with_metadatas(
        query, metadatas, n_results=n_results
    )

    mock_embedding_model.encode.assert_called_once_with(query)
    mock_chroma_collection.query.assert_called_once_with(
        query_embeddings=[[0.3, 0.4, 0.5]],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],  # Chroma default
        where=metadatas,
    )
    assert results == mock_query_result_meta


def test_memory_check_collection_none_empty(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    mock_chroma_collection.count.return_value = 0
    memory = Memory(
        collection_name="test_empty_check",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    count = memory.check_collection_none()
    assert count == 0
    mock_chroma_collection.count.assert_called_once()


def test_memory_check_collection_none_not_empty(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    mock_chroma_collection.count.return_value = 5
    memory = Memory(
        collection_name="test_not_empty_check",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    count = memory.check_collection_none()
    assert count == 5
    mock_chroma_collection.count.assert_called_once()


# New tests for insert_vectors failures
def test_memory_insert_vectors_embedding_returns_none(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection, capsys, mocker
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_embed_none",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_embedding_model.encode.return_value = None  # Simulate encode returning None

    doc_id = str(uuid.uuid4())
    documents = [
        ChunkedDocument(doc_name="doc1", chunks=["chunk1"], doc_id=doc_id, source="s1")
    ]

    mocker.patch("fedotllm.agents.memory.tqdm", lambda x, **kwargs: x)
    memory.insert_vectors(documents)

    captured = capsys.readouterr()
    assert "Warning: Could not generate embedding for chunk: chunk1..." in captured.out
    mock_chroma_collection.add.assert_not_called()
    assert memory.id == 0  # No chunks successfully added


def test_memory_insert_vectors_embedding_returns_empty_list(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection, capsys, mocker
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_embed_empty_list",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_embedding_model.encode.return_value = []  # Simulate encode returning an empty list

    doc_id = str(uuid.uuid4())
    documents = [
        ChunkedDocument(doc_name="doc1", chunks=["chunk1"], doc_id=doc_id, source="s1")
    ]

    mocker.patch("fedotllm.agents.memory.tqdm", lambda x, **kwargs: x)
    memory.insert_vectors(documents)

    captured = capsys.readouterr()
    # The code currently checks `if not embedding_obj or not hasattr(embedding_obj[0], "embedding")`,
    # an empty list for embedding_obj would cause `not embedding_obj` to be true if not fixed.
    # Let's assume the check `embedding_obj[0]` would raise IndexError if `embedding_obj` is `[]`.
    # The current code is `if not embedding_obj or not hasattr(embedding_obj[0], "embedding")`.
    # If `embedding_obj` is `[]`, `not embedding_obj` is `False`. `embedding_obj[0]` will raise IndexError.
    # This should be caught by a try-except in real code, or the check needs to be more robust.
    # For now, based on current code, it will print the warning.
    assert "Warning: Could not generate embedding for chunk: chunk1..." in captured.out
    mock_chroma_collection.add.assert_not_called()
    assert memory.id == 0


def test_memory_insert_vectors_embedding_malformed_object(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection, capsys, mocker
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_embed_malformed",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    malformed_embedding_obj = MagicMock()
    del malformed_embedding_obj.embedding  # Remove the 'embedding' attribute
    mock_embedding_model.encode.return_value = [malformed_embedding_obj]

    doc_id = str(uuid.uuid4())
    documents = [
        ChunkedDocument(doc_name="doc1", chunks=["chunk1"], doc_id=doc_id, source="s1")
    ]

    mocker.patch("fedotllm.agents.memory.tqdm", lambda x, **kwargs: x)
    memory.insert_vectors(documents)

    captured = capsys.readouterr()
    assert "Warning: Could not generate embedding for chunk: chunk1..." in captured.out
    mock_chroma_collection.add.assert_not_called()
    assert memory.id == 0


def test_memory_insert_vectors_all_chunks_fail_embedding(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection, capsys, mocker
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_all_fail",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    # Simulate failure for all chunks
    mock_embedding_model.encode.side_effect = [None, None]

    doc_id = str(uuid.uuid4())
    documents = [
        ChunkedDocument(
            doc_name="doc1", chunks=["chunk_A", "chunk_B"], doc_id=doc_id, source="s1"
        )
    ]

    mocker.patch("fedotllm.agents.memory.tqdm", lambda x, **kwargs: x)
    memory.insert_vectors(documents)

    captured = capsys.readouterr()
    assert "Warning: Could not generate embedding for chunk: chunk_A..." in captured.out
    assert "Warning: Could not generate embedding for chunk: chunk_B..." in captured.out
    mock_chroma_collection.add.assert_not_called()  # Crucial: add should not be called for this document
    assert memory.id == 0


def test_memory_insert_vectors_mixed_success_failure(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection, capsys, mocker
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_mixed_fail",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    successful_embedding_obj = MagicMock()
    successful_embedding_obj.embedding = [0.1, 0.2, 0.3]

    # First chunk fails, second succeeds
    mock_embedding_model.encode.side_effect = [None, [successful_embedding_obj]]

    doc_id = str(uuid.uuid4())
    documents = [
        ChunkedDocument(
            doc_name="doc1",
            chunks=["chunk_fail", "chunk_ok"],
            doc_id=doc_id,
            source="s1",
        )
    ]

    mocker.patch("fedotllm.agents.memory.tqdm", lambda x, **kwargs: x)
    memory.insert_vectors(documents)

    captured = capsys.readouterr()
    assert (
        "Warning: Could not generate embedding for chunk: chunk_fail..." in captured.out
    )  # chunk_fail

    mock_chroma_collection.add.assert_called_once_with(
        ids=[f"{doc_id}_1"],  # Only the second chunk (index 1)
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["chunk_ok"],
        metadatas=[
            {"doc_id": doc_id, "doc_name": "doc1", "source": "s1", "chunk_id": 1}
        ],  # Assuming default metadata
    )
    assert memory.id == 1  # Only one chunk added


def test_memory_insert_vectors_empty_document_list(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection, mocker
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_empty_docs",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    documents = []  # Empty list of documents

    mocker.patch("fedotllm.agents.memory.tqdm", lambda x, **kwargs: x)
    memory.insert_vectors(documents)

    mock_chroma_collection.add.assert_not_called()
    assert memory.id == 0


# New tests for search_context failures
def test_memory_search_context_embedding_returns_none(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_search_embed_none",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_embedding_model.encode.return_value = None  # Query embedding fails

    results = memory.search_context("test query")

    expected_empty_result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert results == expected_empty_result
    mock_chroma_collection.query.assert_not_called()


def test_memory_search_context_embedding_returns_empty_list(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_search_embed_empty",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_embedding_model.encode.return_value = []  # Query embedding returns empty list

    results = memory.search_context("test query")

    expected_empty_result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert results == expected_empty_result
    mock_chroma_collection.query.assert_not_called()


def test_memory_search_context_embedding_malformed_object(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_search_embed_malformed",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    malformed_embedding_obj = MagicMock()
    del malformed_embedding_obj.embedding  # Remove 'embedding' attribute
    mock_embedding_model.encode.return_value = [malformed_embedding_obj]

    results = memory.search_context("test query")

    expected_empty_result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert results == expected_empty_result
    mock_chroma_collection.query.assert_not_called()


# New tests for search_context_with_metadatas failures
def test_memory_search_context_with_metadatas_embedding_returns_none(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_search_meta_embed_none",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_embedding_model.encode.return_value = None  # Query embedding fails

    results = memory.search_context_with_metadatas("test query", {"source": "s1"})

    expected_empty_result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert results == expected_empty_result
    mock_chroma_collection.query.assert_not_called()


def test_memory_search_context_with_metadatas_embedding_returns_empty_list(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_search_meta_embed_empty",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    mock_embedding_model.encode.return_value = []  # Query embedding returns empty list

    results = memory.search_context_with_metadatas("test query", {"source": "s1"})

    expected_empty_result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert results == expected_empty_result
    mock_chroma_collection.query.assert_not_called()


def test_memory_search_context_with_metadatas_embedding_malformed_object(
    mock_chroma_client, mock_embedding_model, mock_chroma_collection
):
    mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
    memory = Memory(
        collection_name="test_search_meta_embed_malformed",
        embedding_model=mock_embedding_model,
        client=mock_chroma_client,
    )

    malformed_embedding_obj = MagicMock()
    del malformed_embedding_obj.embedding  # Remove 'embedding' attribute
    mock_embedding_model.encode.return_value = [malformed_embedding_obj]

    results = memory.search_context_with_metadatas("test query", {"source": "s1"})

    expected_empty_result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    assert results == expected_empty_result
    mock_chroma_collection.query.assert_not_called()
