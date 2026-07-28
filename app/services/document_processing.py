import logging
import os
import numpy as np
from PyPDF2 import PdfReader
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from .embeddings import EmbeddingService, VectorStore
from .. import crud
from ..models import Document, Chunk
from ..utils import normalize_text, chunk_text, serialize_embedding
from ..config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def extract_pdf_text(self, file_path: str):
        reader = PdfReader(file_path)
        pages = []
        for page_index, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            pages.append({"page_number": page_index, "text": normalize_text(raw)})
        return pages

    def build_chunks(self, pages):
        chunks = []
        vector_id = 0

        for page in pages:
            page_chunks = chunk_text(page["text"], max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for chunk in page_chunks:
                chunks.append({
                    "text": chunk,
                    "page_number": page["page_number"],
                    "vector_id": vector_id,
                })
                vector_id += 1

        return chunks

    def process_document(self, db: Session, document: Document):
        document.status = "processing"
        db.commit()
        pages = self.extract_pdf_text(document.file_path)
        all_chunks = []
        vector_id_base = self._current_vector_count(db)

        for page in pages:
            page_chunks = chunk_text(page["text"], max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for local_index, chunk_text_str in enumerate(page_chunks):
                all_chunks.append({
                    "text": chunk_text_str,
                    "page_number": page["page_number"],
                    "vector_id": vector_id_base + len(all_chunks),
                })

        if not all_chunks:
            document.total_pages = len(pages)
            document.total_chunks = 0
            document.status = "processed"
            db.commit()
            return []

        embeddings = self.embedding_service.embed_texts([chunk["text"] for chunk in all_chunks])
        chunks = []
        ids = []
        vectors = []

        for chunk_def, vector in zip(all_chunks, embeddings):
            chunk = Chunk(
                document_id=document.id,
                text=chunk_def["text"],
                page_number=chunk_def["page_number"],
                vector_id=chunk_def["vector_id"],
                embedding=serialize_embedding(vector),
            )
            db.add(chunk)
            chunks.append(chunk)
            ids.append(chunk_def["vector_id"])
            vectors.append(vector)

        db.commit()
        self.vector_store.add(np.array(ids, dtype=np.int64), np.vstack(vectors))
        self.vector_store.save_index()

        document.total_pages = len(pages)
        document.total_chunks = len(chunks)
        document.status = "processed"
        db.commit()
        return chunks

    def reprocess_document(self, db: Session, document: Document):
        chunk_ids = [chunk.vector_id for chunk in document.chunks]
        crud.delete_chunks_by_document(db, document.id)
        self.vector_store.remove(chunk_ids)
        self.vector_store.save_index()
        return self.process_document(db, document)

    def _current_vector_count(self, db: Session):
        current_max = db.query(func.max(Chunk.vector_id)).scalar()
        return int(current_max + 1) if current_max is not None else 0
