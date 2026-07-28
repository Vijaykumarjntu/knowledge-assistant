import logging
from typing import List

import torch
from sqlalchemy.orm import Session
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .embeddings import EmbeddingService, VectorStore
from ..models import Chunk, Document, Interaction
from ..config import HF_MODEL_NAME, SEARCH_TOP_K

logger = logging.getLogger(__name__)


# import logging
import os

from huggingface_hub import InferenceClient

from ..config import HF_MODEL_NAME

# logger = logging.getLogger(__name__)


# class LLMService:
#     def __init__(self):
#         self.model_name = HF_MODEL_NAME
#         self.available = False
#         self.model = None
#         self.tokenizer = None
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
#             self.model.to(self.device)
#             self.available = True
#             logger.info("Loaded Hugging Face model %s for generation", self.model_name)
#         except Exception as exc:
#             logger.warning("Failed to load Hugging Face generation model: %s", exc)
#             self.available = False

#     def generate(self, prompt: str, max_tokens: int = 250):
#         if not self.available:
#             return (
#                 "LLM generation is not available. "
#                 "The answer is based on retrieved document context only."
#             )

#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#         )
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#         )
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()



class LLMService:
    def __init__(self):
        self.model_name = HF_MODEL_NAME  # e.g. "Qwen/Qwen2.5-1.5B-Instruct"
        self.api_key = os.getenv("HF_API_TOKEN")
        self.available = bool(self.api_key)
        self.client = None

        if self.available:
            try:
                self.client = InferenceClient(api_key=self.api_key)
                logger.info("Using Hugging Face Inference API for model %s", self.model_name)
            except Exception as exc:
                logger.warning("Failed to init HF InferenceClient: %s", exc)
                self.available = False
        else:
            logger.warning("HF_API_TOKEN not set; LLM generation unavailable.")

    def generate(self, prompt: str, max_tokens: int = 400):
        if not self.available:
            return (
                "LLM generation is not available. "
                "The answer is based on retrieved document context only."
            )

        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("HF Inference API call failed: %s", exc)
            return "LLM generation failed due to an API error."


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()

    def _search_chunks(self, query: str, db: Session, top_k: int = SEARCH_TOP_K):
        query_vector = self.embedding_service.embed_text(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        chunks = []
        for vector_id, score in results:
            chunk = db.query(Chunk).filter(Chunk.vector_id == vector_id).first()
            if chunk:
                chunks.append({
                    "chunk": chunk,
                    "score": score,
                    "document": db.query(Document).filter(Document.id == chunk.document_id).first(),
                })
        return chunks

    def answer_question(self, query: str, db: Session, session_id: str = None, top_k: int = SEARCH_TOP_K):
        chunks_meta = self._search_chunks(query, db, top_k=top_k)
        if not chunks_meta:
            return {
                "answer": "I could not find enough information in the uploaded documents to answer that question.",
                "sources": [],
                "retrieved_context": [],
                "confidence": 0.0,
            }

        prompt_context = []
        sources = []
        for item in chunks_meta:
            chunk = item["chunk"]
            doc = item["document"]
            prompt_context.append(f"[{doc.name} - page {chunk.page_number}] {chunk.text}")
            sources.append(f"{doc.name} - page {chunk.page_number}")

        prompt = (
            "Use only the following document excerpts to answer the question. "
            "If the answer cannot be determined from the provided content, say so clearly.\n\n"
            f"Context:\n{chr(10).join(prompt_context)}\n\n"
            f"Question: {query}\n"
            "Answer concisely and include citations for the documents used."
        )

        completion = self.llm_service.generate(prompt)
        return {
            "answer": completion,
            "sources": list(dict.fromkeys(sources)),
            "retrieved_context": [
                {
                    "document_id": item["document"].document_id,
                    "document_name": item["document"].name,
                    "page_number": item["chunk"].page_number,
                    "text": item["chunk"].text,
                    "score": float(item["score"]),
                }
                for item in chunks_meta
            ],
            "confidence": None,
        }

    def compare_documents(self, document_ids: List[str], question: str, db: Session, top_k: int = SEARCH_TOP_K):
        docs = db.query(Document).filter(Document.document_id.in_(document_ids)).all()
        if not docs:
            return {
                "answer": "No documents matched the comparison request.",
                "sources": [],
                "retrieved_context": [],
            }

        query_vector = self.embedding_service.embed_text(question)
        results = self.vector_store.search(query_vector, top_k=top_k)
        context = []
        sources = []
        for vector_id, score in results:
            chunk = db.query(Chunk).filter(Chunk.vector_id == vector_id, Chunk.document_id.in_([doc.id for doc in docs])).first()
            if chunk:
                doc = db.query(Document).filter(Document.id == chunk.document_id).first()
                context.append(f"[{doc.name} - page {chunk.page_number}] {chunk.text}")
                sources.append(f"{doc.name} - page {chunk.page_number}")

        if not context:
            return {
                "answer": "No matching content was found for the selected documents.",
                "sources": [],
                "retrieved_context": [],
            }

        prompt = (
            "Compare the document excerpts below and answer the question. "
            "Provide similarities, differences, and any key findings.\n\n"
            f"Context:\n{chr(10).join(context)}\n\n"
            f"Question: {question}\n"
            "Answer with reference to the source documents."
        )
        answer = self.llm_service.generate(prompt)
        return {
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),
            "retrieved_context": [
                {
                    "document_id": db.query(Document).filter(Document.id == chunk.document_id).first().document_id,
                    "document_name": db.query(Document).filter(Document.id == chunk.document_id).first().name,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "score": float(score),
                }
                for vector_id, score in results
                for chunk in [db.query(Chunk).filter(Chunk.vector_id == vector_id).first()]
                if chunk and chunk.document_id in [doc.id for doc in docs]
            ],
        }

    def summarize_document(self, document: Document, db: Session, summary_type: str = "executive"):
        contents = []
        for chunk in document.chunks:
            contents.append(f"[Page {chunk.page_number}] {chunk.text}")
        text = "\n".join(contents[:10])
        summary_prompt = (
            f"Create a {summary_type} summary for the following document excerpts. "
            "Highlight the most important points in a clear and concise manner.\n\n"
            f"{text}\n\nSummary:"
        )
        summary = self.llm_service.generate(summary_prompt)
        return summary
