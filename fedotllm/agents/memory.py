from chromadb.api.client import Client
from chromadb.api.types import QueryResult
from fedotllm.llm import OpenaiEmbeddings
from tqdm import tqdm
from typing import List
from dataclasses import dataclass, field


def text_splitter(
    text: str,
    max_chunk_length: int = OpenaiEmbeddings.MAX_INPUT,
    overlap_ratio: float = 0.1,
):
    if not text:
        return [""]
    if not (0 <= overlap_ratio < 1):
        raise ValueError("Overlap ratio must be between 0 and 1 (exclusive).")

    overlap_length = int(max_chunk_length * overlap_ratio)
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_length, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chunk_length - overlap_length

    return chunks


@dataclass
class ChunkedDocument:
    doc_name: str
    chunks: List[str]
    doc_id: str
    source: str
    metadata: dict = field(default_factory=dict)


class Memory:
    def __init__(
        self, client: Client, collection_name: str, embedding_model: OpenaiEmbeddings
    ):
        self.chroma_client = client
        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )

        self.embedding_model = embedding_model
        self.id = 0

    def insert_vectors(self, documents: List[ChunkedDocument]):
        for document in tqdm(documents, desc="Embedding"):
            chunk_embeddings = []
            chunk_texts = []
            chunk_ids = []
            chunk_metadatas = []

            for i, chunk in enumerate(document.chunks):
                # Assuming encode returns a list of embedding objects, and we take the first
                embedding_obj = self.embedding_model.encode(input=chunk)
                if not embedding_obj or not hasattr(embedding_obj[0], "embedding"):
                    # Skip this chunk if embedding fails or has unexpected structure
                    print(
                        f"Warning: Could not generate embedding for chunk: {chunk[:50]}..."
                    )
                    continue

                chunk_embeddings.append(embedding_obj[0].embedding)
                chunk_texts.append(chunk)
                chunk_ids.append(f"{document.doc_id}_{i}")

                # Combine document metadata with chunk-specific metadata
                combined_metadata = {
                    **document.metadata,
                    "doc_id": document.doc_id,
                    "doc_name": document.doc_name,  # Use doc_name
                    "source": document.source,
                    "chunk_id": i,
                }
                chunk_metadatas.append(combined_metadata)
                self.id += 1  # Increment global ID for each chunk

            if chunk_texts:  # Only add if there are valid chunks to add
                self.collection.add(
                    ids=chunk_ids,
                    embeddings=chunk_embeddings,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas,
                )
        print("---------------------------------")
        print(f"Finished inserting vectors for <{self.collection_name}>!")
        print("---------------------------------")

    def search_context(self, query: str, n_results=5) -> QueryResult:
        query_embedding_obj = self.embedding_model.encode(query)
        if not query_embedding_obj or not hasattr(query_embedding_obj[0], "embedding"):
            # Handle cases where query embedding fails
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }  # Return empty QueryResult-like dict

        query_embeddings_list = [query_embedding_obj[0].embedding]

        results = self.collection.query(
            query_embeddings=query_embeddings_list,  # Must be List[List[float]]
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        return results

    def search_context_with_metadatas(
        self, query: str, where: dict, n_results=5
    ) -> dict:
        query_embedding_obj = self.embedding_model.encode(query)
        if not query_embedding_obj or not hasattr(query_embedding_obj[0], "embedding"):
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_embeddings_list = [query_embedding_obj[0].embedding]

        results = self.collection.query(
            query_embeddings=query_embeddings_list,  # Must be List[List[float]]
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
            where=where,
        )
        return results

    def check_collection_none(self):
        document_count = self.collection.count()
        if document_count == 0:
            print(f"The collection <{self.collection_name}> is empty.")
        else:
            print(
                f"The collection <{self.collection_name}>  has {document_count} documents."
            )

        return document_count
