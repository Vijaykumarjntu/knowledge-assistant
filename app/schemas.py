from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    document_id: str
    name: str
    upload_timestamp: datetime
    total_pages: int
    total_chunks: int
    status: str
    category: Optional[str] = None

    class Config:
        orm_mode = True


class DocumentCreateResponse(BaseModel):
    document_id: str
    message: str


class SearchRequest(BaseModel):
    query: str
    mode: Optional[str] = "semantic"
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    document_id: str
    document_name: str
    page_number: Optional[int]
    text: str
    score: float


class QARequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5


class QAResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_context: List[SearchResult]
    confidence: Optional[float] = None


class CompareRequest(BaseModel):
    document_ids: List[str]
    question: str
    top_k: Optional[int] = 5


class SummarizeRequest(BaseModel):
    document_id: str
    summary_type: Optional[str] = "executive"


class SummaryResponse(BaseModel):
    document_id: str
    summary_type: str
    summary: str


class AnalyticsResponse(BaseModel):
    documents_count: int
    total_chunks: int
    total_embeddings: int
    total_interactions: int
    most_active_documents: List[str]


class ClassifyResponse(BaseModel):
    document_id: str
    category: str
    confidence: Optional[float] = None
