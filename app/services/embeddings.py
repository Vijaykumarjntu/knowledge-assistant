import logging
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import FAISS_INDEX_PATH, SEARCH_TOP_K

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def embed_texts(self, texts):
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if len(vectors.shape) == 1:
            vectors = np.expand_dims(vectors, axis=0)
        return vectors


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.use_faiss = False
        self._load_index()

    def _load_index(self):
        try:
            import faiss

            if Path(FAISS_INDEX_PATH).exists():
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                logger.info("Loaded FAISS index from %s", FAISS_INDEX_PATH)
            else:
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIDMap(quantizer)
                logger.info("Initialized new FAISS index")
            self.use_faiss = True
        except Exception as exc:
            logger.warning("FAISS unavailable, falling back to brute-force search: %s", exc)
            self.index = {}
            self.use_faiss = False

    def save_index(self):
        if self.use_faiss:
            import faiss

            faiss.write_index(self.index, str(FAISS_INDEX_PATH))
            logger.info("Saved FAISS index to %s", FAISS_INDEX_PATH)

    def add(self, ids, vectors):
        if self.use_faiss:
            self.index.add_with_ids(vectors.astype("float32"), ids)
        else:
            for chunk_id, vector in zip(ids, vectors):
                self.index[int(chunk_id)] = vector

    def remove(self, ids):
        if self.use_faiss:
            self.index.remove_ids(np.array(ids, dtype=np.int64))
        else:
            for chunk_id in ids:
                self.index.pop(int(chunk_id), None)

    def search(self, query_vector, top_k: int = SEARCH_TOP_K):
        if self.use_faiss:
            query_vector = np.array([query_vector], dtype="float32")
            scores, ids = self.index.search(query_vector, top_k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

        else:
            ids = list(self.index.keys())
            if not ids:
                return []
            vectors = np.vstack([self.index[i] for i in ids])
            similarities = cosine_similarity([query_vector], vectors)[0]
            ranked = sorted(zip(ids, similarities), key=lambda x: x[1], reverse=True)
            return [(int(chunk_id), float(score)) for chunk_id, score in ranked[:top_k]]
