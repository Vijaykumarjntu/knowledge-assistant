import os
import shutil
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .config import DOCUMENT_DIR
from .database import SessionLocal, init_db
from .crud import (
    add_interaction,
    create_document,
    delete_chunks_by_document,
    delete_document,
    get_document_by_id,
    list_documents,
)

from .models import Chunk, Interaction
from .schemas import (
    AnalyticsResponse,
    ClassifyResponse,
    CompareRequest,
    DocumentCreateResponse,
    DocumentMetadata,
    QARequest,
    QAResponse,
    SearchRequest,
    SearchResult,
    SummarizeRequest,
    SummaryResponse,
)
from .services.classifier import ClassifierService
from .services.conversation import ConversationStore
from .services.document_processing import DocumentProcessor
from .services.qa import RAGService
from .utils import ensure_directories, generate_document_id, build_document_path

app = FastAPI(title="Research Knowledge Assistant")

ensure_directories()
init_db()
processor = DocumentProcessor()
rag_service = RAGService()
classifier_service = ClassifierService()
conversation_store = ConversationStore()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/documents/upload", response_model=List[DocumentMetadata])
def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    created = []
    print("this route working")
    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
        doc_id = generate_document_id()
        target_path = build_document_path(f"{doc_id}_{upload.filename}")
        with open(target_path, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        document = create_document(db, doc_id, upload.filename, target_path)
        processor.process_document(db, document)
        if classifier_service.is_ready():
            text_for_classification = " ".join(chunk.text for chunk in document.chunks)
            category, confidence = classifier_service.predict(text_for_classification)
            document.category = category
            db.commit()
        created.append(document)
    return created


@app.get("/documents", response_model=List[DocumentMetadata])
def documents(db: Session = Depends(get_db)):
    return list_documents(db)


@app.get("/documents/{document_id}", response_model=DocumentMetadata)
def get_document(document_id: str, db: Session = Depends(get_db)):
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@app.delete("/documents/{document_id}")
def remove_document(document_id: str, db: Session = Depends(get_db)):
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    vector_ids = [chunk.vector_id for chunk in document.chunks]
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    delete_document(db, document)
    processor.vector_store.remove(vector_ids)
    processor.vector_store.save_index()
    return JSONResponse({"message": "Document deleted."})


@app.post("/documents/{document_id}/reprocess", response_model=DocumentMetadata)
def reprocess_document(document_id: str, db: Session = Depends(get_db)):
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    delete_chunks_by_document(db, document.id)
    processor.reprocess_document(db, document)
    return document


@app.post("/search", response_model=List[SearchResult])
def search(request: SearchRequest, db: Session = Depends(get_db)):
    result_chunks = []
    print("search route working")
    if request.mode == "keyword":
        query_lower = request.query.lower()
        candidates = db.query(Chunk).filter(Chunk.text.ilike(f"%{request.query}%"))
        for chunk in candidates.limit(request.top_k).all():
            result_chunks.append((chunk, 1.0))
    else:
        query_vector = processor.embedding_service.embed_text(request.query)
        results = processor.vector_store.search(query_vector, top_k=request.top_k)
        for vector_id, score in results:
            chunk = db.query(Chunk).filter(Chunk.vector_id == vector_id).first()
            if chunk:
                result_chunks.append((chunk, score))

    if request.mode == "hybrid":
        keyword_hits = {chunk.id: score for chunk, score in result_chunks}
        candidates = db.query(Chunk).filter(Chunk.text.ilike(f"%{request.query}%"))
        for chunk in candidates.limit(request.top_k).all():
            keyword_hits[chunk.id] = max(keyword_hits.get(chunk.id, 0.0), 0.2)
        result_chunks = [(db.get(Chunk, chunk_id), score) for chunk_id, score in sorted(keyword_hits.items(), key=lambda x: x[1], reverse=True)[: request.top_k]]

    output = []
    for chunk, score in result_chunks:
        if not chunk:
            continue
        output.append(
            SearchResult(
                document_id=chunk.document.document_id,
                document_name=chunk.document.name,
                page_number=chunk.page_number,
                text=chunk.text,
                score=score,
            )
        )
    return output


@app.post("/qa", response_model=QAResponse)
def ask_question(request: QARequest, db: Session = Depends(get_db)):
    response = rag_service.answer_question(request.query, db, session_id=request.session_id, top_k=request.top_k)
    if request.session_id and response["retrieved_context"]:
        conversation_store.update_context(request.session_id, response["retrieved_context"][0]["document_id"])
    add_interaction(db, request.session_id or "anonymous", request.query, "qa", referenced_docs=",".join(response["sources"]))
    return QAResponse(**response)


@app.post("/documents/{document_id}/classify", response_model=ClassifyResponse)
def classify_document(document_id: str, db: Session = Depends(get_db)):
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not classifier_service.is_ready():
        raise HTTPException(status_code=503, detail="Classifier is not trained.")
    full_text = " ".join(chunk.text for chunk in document.chunks)
    category, confidence = classifier_service.predict(full_text)
    document.category = category
    db.commit()
    return ClassifyResponse(document_id=document_id, category=category, confidence=confidence)


@app.post("/compare")
def compare_documents(request: CompareRequest, db: Session = Depends(get_db)):
    response = rag_service.compare_documents(request.document_ids, request.question, db, top_k=request.top_k)
    add_interaction(db, None, request.question, "compare", referenced_docs=",".join(request.document_ids))
    return response


@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummarizeRequest, db: Session = Depends(get_db)):
    document = get_document_by_id(db, request.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    summary = rag_service.summarize_document(document, db, summary_type=request.summary_type)
    return SummaryResponse(document_id=request.document_id, summary_type=request.summary_type, summary=summary)


@app.get("/analytics", response_model=AnalyticsResponse)
def analytics(db: Session = Depends(get_db)):
    docs = list_documents(db)
    interactions = db.query(Interaction).count()
    total_chunks = sum(doc.total_chunks for doc in docs)
    most_active = [doc.name for doc in docs[:5]]
    return AnalyticsResponse(
        documents_count=len(docs),
        total_chunks=total_chunks,
        total_embeddings=total_chunks,
        total_interactions=interactions,
        most_active_documents=most_active,
    )


@app.post("/classifier/train")
def train_classifier():
    import scripts.train_classifier as trainer

    dataset = trainer.build_training_dataset()
    classifier_service.train(dataset["texts"], dataset["labels"], epochs=8)
    return {"message": "Classifier trained and persisted."}
