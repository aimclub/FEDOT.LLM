from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Callable
from chromadb.config import Settings

import chromadb
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from typing_extensions import List, Dict
from typing_extensions import NamedTuple

from fedot_llm.ai.agents.load import load_fedot_docs

# logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"


class Resource(NamedTuple):
    collection_name: str
    loader: Callable[[], List[Document]]


resources: List[Resource] = [
    Resource("FedotDocs", load_fedot_docs)
]


@dataclass
class Collection:
    collection_name: str
    client: InitVar[ClientAPI]
    embedding_function: Embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5",
                                                     inference_mode="local")
    vectorstore: VectorStore = field(init=False)

    def __post_init__(self, client: ClientAPI):
        self.vectorstore = Chroma(
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            create_collection_if_not_exists=False,
            client=client
        )

    @classmethod
    def create(cls, collection_name: str, client: ClientAPI, embedding_function: Embeddings):
        Chroma(
            embedding_function=embedding_function,
            collection_name=collection_name,
            create_collection_if_not_exists=True,
            client=client
        )
        return cls(collection_name=collection_name, client=client)

    @staticmethod
    def is_exists(collection_name: str, client: ClientAPI):
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def get_retriever(self):
        return self.vectorstore.as_retriever()

    def add_documents(self, docs: List[Document]):
        self.vectorstore.add_documents(docs)

    def reset(self):
        self.vectorstore.delete()


@dataclass
class LongTermMemory:
    persistent_path: Path = Path(CHROMA_PATH)
    client: ClientAPI = field(init=False)
    embedding_function: Embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5",
                                                     inference_mode="local")
    _cached_collections: Dict[str, Collection] = field(default_factory=dict)

    def __post_init__(self):
        self.client = chromadb.PersistentClient(path=str(self.persistent_path),
                                                settings=Settings(anonymized_telemetry=False))
        self.load_resources()

    def load_resources(self):
        for resource in resources:
            if not self.is_collection_exists(resource.collection_name):
                self.create_collection(resource.collection_name)
                self.add_documents(resource.collection_name, resource.loader())
            # else:
            #     logger.info(
            #         f"Collection: '{resource.collection_name}' already exists in Chroma. Skipping document load.")

    def is_collection_exists(self, collection_name: str):
        return Collection.is_exists(collection_name, self.client)

    def create_collection(self, collection_name: str):
        collection = Collection.create(collection_name=collection_name,
                                       client=self.client,
                                       embedding_function=self.embedding_function)
        self._cached_collections[collection_name] = collection
        return collection

    def add_documents(self, collection_name: str, docs: List[Document]):
        self.get_collection(collection_name).add_documents(docs)

    def reset_collection(self, collection_name: str):
        if collection_name in self._cached_collections:
            self._cached_collections[collection_name].reset()
            del self._cached_collections[collection_name]
        else:
            raise ValueError(f"Collection {collection_name} not found")

    def get_collection(self, collection_name: str):
        if Collection.is_exists(collection_name, self.client):
            if collection_name not in self._cached_collections:
                self._cached_collections[collection_name] = Collection(collection_name=collection_name,
                                                                       client=self.client,
                                                                       embedding_function=self.embedding_function)
            return self._cached_collections[collection_name]
        else:
            del self._cached_collections[collection_name]
            raise ValueError(f"Collection {collection_name} not found")
