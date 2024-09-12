from typing import List

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownTextSplitter
from typing_extensions import Sequence


def load_fedot_docs() -> List[Document]:
    docs = split_md_docs(
        convert_html_to_md(
            clean_fedot_docs(
                fetch_fedot_docs()
            )
        )
    )
    return list(docs)


def fetch_fedot_docs():
    loader = RecursiveUrlLoader(url='https://fedot.readthedocs.io/en/latest/', max_depth=10)
    return loader.load()


def __extract_sections_with_ids(html_doc: str):
    soup = BeautifulSoup(html_doc, 'html.parser')
    sections = soup.find_all('section')
    sections_with_id = [str(section) for section in sections if section.get('id') is not None]
    if len(sections_with_id) > 0:
        return ''.join(sections_with_id)
    return None


def clean_fedot_docs(docs: Sequence[Document]) -> Sequence[Document]:
    filtered_docs = [doc for doc in docs
                     if not doc.metadata['source'].endswith(('index.html', '/', 'search.html'))
                     and doc.metadata['content_type'].startswith('text/html')]
    content_docs = []
    for doc in filtered_docs:
        if extracted := __extract_sections_with_ids(doc.page_content):
            doc.page_content = extracted
            content_docs.append(doc)

    return content_docs


def convert_html_to_md(docs: Sequence[Document]) -> Sequence[Document]:
    md_docs = MarkdownifyTransformer().transform_documents(docs)
    return md_docs


def split_md_docs(docs: Sequence[Document]) -> Sequence[Document]:
    md_splitter = MarkdownTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    docs_splited = md_splitter.split_documents(docs)
    docs_splited = filter_complex_metadata(docs_splited)
    return docs_splited
