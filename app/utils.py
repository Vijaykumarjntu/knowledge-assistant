import os
import re
import uuid
from pathlib import Path
from datetime import datetime

from .config import DATA_DIR, DOCUMENT_DIR, STORAGE_DIR


def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    os.makedirs(STORAGE_DIR, exist_ok=True)


def generate_document_id() -> str:
    return str(uuid.uuid4())


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current.strip())
            overlap_text = current[-overlap:] if overlap and len(current) > overlap else ""
            current = overlap_text + " " + sentence if overlap_text else sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence

    if current:
        chunks.append(current.strip())

    return chunks


def serialize_embedding(vector) -> bytes:
    return vector.astype("float32").tobytes()


def deserialize_embedding(data: bytes):
    import numpy as np

    return np.frombuffer(data, dtype="float32")


def build_document_path(file_name: str) -> str:
    return str(Path(DOCUMENT_DIR) / file_name)


def format_sources(chunks):
    return [f"{chunk.document.name} - page {chunk.page_number or 'unknown'}" for chunk in chunks]


def get_timestamp() -> datetime:
    return datetime.utcnow()
