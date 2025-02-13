from chromadb.api.client import Client
from chromadb.api.types import QueryResult
from fedotllm.llm.inference import OpenaiEmbeddings
from tqdm import tqdm
from typing import NamedTuple, List


def text_splitter(text: str, max_chunk_length: int = OpenaiEmbeddings.MAX_INPUT, overlap_ratio: float = 0.1):
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


class ChunkedDocument(NamedTuple):
    source: str
    title: str
    chunks: str


class Memory:
    def __init__(self, client: Client, collection_name: str, embedding_model: OpenaiEmbeddings):
        self.chroma_client = client
        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"})

        self.embedding_model = embedding_model
        self.id = 0

    def insert_vectors(self, documents: List[ChunkedDocument]):
        for document in tqdm(documents, desc='Embedding'):
            for chunk in document.chunks:
                text_embedding = self.embedding_model.encode(input=chunk)[
                    0].embedding

                metadata = {
                    'source': document.source,
                    'title': document.title
                }

                self.collection.add(
                    documents=chunk,
                    ids=f'{self.id}',
                    embeddings=text_embedding,
                    metadatas=metadata
                )
                self.id += 1
        print('---------------------------------')
        print(f'Finished inserting vectors for <{self.collection_name}>!')
        print('---------------------------------')

    def search_context(self, query: str, n_results=5) -> QueryResult:
        query_embeddings = self.embedding_model.encode(query)[0].embedding
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=['documents', 'distances', 'metadatas'])
        return results

    def search_context_with_metadatas(self, query: str, where: dict, n_results=5) -> dict:
        query_embeddings = self.embedding_model.encode(query)[0].embedding
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=['documents', 'distances', 'metadatas'],
            where=where)
        return results

    def check_collection_none(self):
        document_count = self.collection.count()
        if document_count == 0:
            print(f"The collection <{self.collection_name}> is empty.")
        else:
            print(
                f"The collection <{self.collection_name}>  has {document_count} documents.")

        return document_count
