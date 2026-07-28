import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
DOCUMENT_DIR = STORAGE_DIR / "documents"
FAISS_INDEX_PATH = STORAGE_DIR / "faiss.index"
CLASSIFIER_DIR = STORAGE_DIR / "classifier"
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")
# HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/flan-t5-small")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))
