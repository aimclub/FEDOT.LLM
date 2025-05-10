from fedotllm.llm import OpenaiEmbeddings
import chromadb
from fedotllm.agents.memory import Memory, ChunkedDocument, text_splitter
from fedotllm.agents.scraper import recursive_url_loader
from bs4 import BeautifulSoup
from chromadb.api.types import QueryResult
from html2text import html2text
from chromadb.config import Settings


class RetrieveTool:
    def __init__(
        self,
        embeddings: OpenaiEmbeddings,
        base_url: str = "https://fedot.readthedocs.io/en/latest/",
        collection_name: str = "docs",
    ):
        self.embeddings = embeddings
        self.base_url = base_url
        self.client = chromadb.PersistentClient(
            path="db", settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name

        self.db = Memory(self.client, self.collection_name, self.embeddings)

    def count(self):
        return self.db.check_collection_none()

    def create_db_docs(self):
        documents = recursive_url_loader(url=self.base_url)

        def extract_sections_with_ids(raw_html: str):
            soup = BeautifulSoup(raw_html, "html.parser")
            sections = soup.find_all("section")
            sections_with_id = [
                str(section) for section in sections if section.get("id") is not None
            ]
            if len(sections_with_id) > 0:
                return "".join(sections_with_id)
            return None

        filtered_docs = [
            doc
            for doc in documents
            if not doc["metadata"]["source"].endswith(
                ("index.html", "/", "search.html")
            )
            and doc["metadata"]["content_type"].startswith("text/html")
        ]

        chunked_docs = []
        for doc in filtered_docs:
            if extracted := extract_sections_with_ids(doc["content"]):
                ch_doc = ChunkedDocument(
                    source=doc["metadata"]["source"],
                    title=doc["metadata"]["title"],
                    chunks=text_splitter(html2text(extracted)),
                )
                chunked_docs.append(ch_doc)

        self.db.insert_vectors(documents=chunked_docs)

    def query_docs(self, query: str) -> QueryResult:
        results = self.db.search_context(query)
        return results
