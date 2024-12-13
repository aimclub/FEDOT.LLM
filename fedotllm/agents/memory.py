from pathlib import Path
from typing import Callable, List, Dict, Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict, field_validator, ValidationInfo, model_validator

from fedotllm.agents.load import load_fedot_docs

CHROMA_PATH = "chroma"


class MemoryResource(BaseModel):
    collection_name: str
    loader: Callable[[], List[Document]]


resources: List[MemoryResource] = [
    MemoryResource(collection_name="FedotDocs", loader=load_fedot_docs)
]


class Collection(BaseModel):
    collection_name: str
    client: ClientAPI
    embedding_function: Embeddings = Field(
        default_factory=lambda: NomicEmbeddings(
            model="nomic-embed-text-v1.5", inference_mode="local"
        )
    )
    _vectorstore: VectorStore = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, collection_name: str, client: ClientAPI, embedding_function: Embeddings):
        Chroma(
            embedding_function=embedding_function,
            collection_name=collection_name,
            create_collection_if_not_exists=True,
            client=client
        )
        return cls(
            collection_name=collection_name,
            client=client,
            embedding_function=embedding_function
        )

    @staticmethod
    def is_exists(collection_name: str, client: ClientAPI) -> bool:
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def __init__(self, **data):
        super().__init__(**data)
        self._vectorstore = Chroma(
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            create_collection_if_not_exists=False,
            client=self.client
        )

    def get_retriever(self):
        return self._vectorstore.as_retriever()

    def add_documents(self, docs: List[Document]):
        self._vectorstore.add_documents(docs)

    def reset(self):
        self._vectorstore.delete()


class LongTermMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    persistent_path: Path = Path(CHROMA_PATH)
    embedding_function: Embeddings = Field(
        default_factory=lambda: NomicEmbeddings(
            model="nomic-embed-text-v1.5", inference_mode="local"
        )
    )
    _cached_collections: Dict[str, Collection] = PrivateAttr(
        default_factory=dict)
    client: ClientAPI = Field(
        default_factory=lambda: chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
    )

    @field_validator("client", mode="before")
    @classmethod
    def init_client(cls, v: Optional[ClientAPI], info: ValidationInfo):
        if v is None:
            v = chromadb.PersistentClient(
                path=str(info.data["persistent_path"]),
                settings=Settings(anonymized_telemetry=False)
            )
        return v

    @model_validator(mode="after")
    def load_resources(self):
        for resource in resources:
            if not self.is_collection_exists(resource.collection_name):
                self.create_collection(resource.collection_name)
                self.add_documents(resource.collection_name, resource.loader())
        return self

    def is_collection_exists(self, collection_name: str) -> bool:
        return Collection.is_exists(collection_name, self.client)

    def create_collection(self, collection_name: str) -> Collection:
        collection = Collection.create(
            collection_name=collection_name,
            client=self.client,
            embedding_function=self.embedding_function
        )
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

    def get_collection(self, collection_name: str) -> Collection:
        if Collection.is_exists(collection_name, self.client):
            if collection_name not in self._cached_collections:
                self._cached_collections[collection_name] = Collection(
                    collection_name=collection_name,
                    client=self.client,
                    embedding_function=self.embedding_function
                )
            return self._cached_collections[collection_name]
        else:
            if collection_name in self._cached_collections:
                del self._cached_collections[collection_name]
            raise ValueError(f"Collection {collection_name} not found")
