# Research Knowledge Assistant Backend

A production-oriented backend application for an AI-powered research assistant.

Features:
- PDF upload and document management
- Semantic retrieval with vector search
- Retrieval-Augmented Generation (RAG) QA with citation support
- Document comparison and summarization
- TensorFlow-based document classification
- Conversation memory and analytics

## Run locally

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the API server:

```bash
uvicorn app.main:app --reload
```

3. Use the OpenAPI docs at `http://localhost:8000/docs`.

## API Endpoints

- `POST /documents/upload` — upload one or more PDFs
- `GET /documents` — list uploaded documents
- `DELETE /documents/{document_id}` — remove a document
- `POST /documents/{document_id}/reprocess` — reprocess a document
- `POST /search` — semantic / keyword search across uploaded content
- `POST /qa` — ask a question with RAG-based grounded answers
- `POST /compare` — compare multiple documents
- `POST /summarize` — generate document summaries
- `GET /analytics` — get system analytics
- `POST /classifier/train` — train the document classifier

## Model Training

Train or retrain the document classifier:

```bash
python scripts/train_classifier.py
```

Important: training the classifier is only required if you want to use the `POST /classifier/train` or `POST /documents/{document_id}/classify` functionality. You do NOT need to run classifier training for search, QA, compare, or summarize endpoints.

The classifier is also used for automatic document category labeling on upload only when a trained model is already available; if no classifier exists, uploads still work normally.

Required steps for QA/search to work:
- Upload PDFs via `POST /documents/upload` (this extracts text, creates chunks, computes embeddings, and adds vectors to the index).
- Ensure the FAISS index and embeddings are available (uploads/processes create them automatically). Once documents are processed the `POST /search`, `POST /qa`, `POST /compare`, and `POST /summarize` endpoints will return results based on the uploaded content.

## Notes

- The system uses `sentence-transformers` for embeddings and a FAISS-backed vector store.
- The RAG service uses a free open-source Hugging Face sequence-to-sequence model for generation.
- Uploaded PDFs are stored in `storage/documents`.
