from pathlib import Path

from openai import OpenAI
from llama_index.readers.file.html import HTMLTagReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from custom_types import RAGChunk

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_file(path: str):
    file_path = Path(path)
    if file_path.suffix.lower() in {".html", ".htm"}:
        docs = HTMLTagReader(tag="body").load_data(file=file_path)
    else:
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
    chunks = []
    for doc in docs:
        text = getattr(doc, "text", None)
        if not text:
            continue
        metadata = getattr(doc, "metadata", None) or {}
        page = metadata.get("page_label") or metadata.get("page_number")
        chunks.extend(
            RAGChunk(text=chunk, page=page)
            for chunk in splitter.split_text(text)
        )
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]