import warnings
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from pydantic import HttpUrl
from tqdm import tqdm

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def extract_sub_links(raw_html: str, url: HttpUrl, base_url: HttpUrl):
    parsed_url = urlparse(url)
    parsed_base_url = urlparse(base_url)
    absolute_paths = set()

    soup = BeautifulSoup(raw_html, "html.parser")
    for link_tag in soup.find_all("a", href=True):
        link = link_tag["href"]
        try:
            parsed_link = urlparse(link)
            if parsed_link.scheme == "http" or parsed_link.scheme == "https":
                absolute_path = link
            elif link.startswith("//"):
                absolute_path = f"{parsed_url.scheme}:{link}"
            else:
                absolute_path = urljoin(url, parsed_link.path)
                if parsed_link.query:
                    absolute_path += f"?{parsed_link.query}"

            if urlparse(absolute_path).netloc == parsed_base_url.netloc:
                if absolute_path.startswith(base_url):
                    absolute_paths.add(absolute_path)
        except Exception as _:
            continue

    return absolute_paths


def extract_metadata(raw_html: str, url: str, response: requests.Response) -> dict:
    content_type = getattr(response, "headers").get("Content-Type", "")
    metadata = {"source": url, "content_type": content_type}
    soup = BeautifulSoup(raw_html, "html.parser")
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    return metadata


def recursive_url_loader(url: str, max_depth: int = 10, timeout: int = 10):
    base_url = url
    visited = set()
    documents = []

    def recursive_scrape(url: str, depth: int):
        if depth < 0 or url in visited:
            return

        visited.add(url)
        try:
            response = requests.get(url, timeout=timeout)

            if 400 <= response.status_code <= 599:
                raise ValueError(f"Received HTTP status {response.status_code}")
        except Exception as _:
            return

        document = {
            "content": response.text,
            "metadata": extract_metadata(
                raw_html=response.text, url=url, response=response
            ),
        }
        sub_links = extract_sub_links(
            raw_html=response.text, url=url, base_url=base_url
        )
        return document, sub_links

    depth_bar = tqdm(total=max_depth + 1, desc="Scraping")

    depth = max_depth
    waiting = {base_url}
    while True:
        links = waiting.copy()
        for link in links:
            depth_bar.set_postfix_str(f"{link}")
            if scraped := recursive_scrape(link, depth):
                document, sub_links = scraped
                documents.append(document)
                waiting.update(sub_links)
        waiting.difference_update(visited)
        depth = depth - 1
        depth_bar.update(1)
        if depth < 0 or len(waiting) == 0:
            depth_bar.n = max_depth + 1
            return documents
