"""Embedding back-ends. TF-IDF runs offline; the neural adapter is optional."""
import numpy as np


class TfidfEmbedder:
    """Offline default: TF-IDF vectors (L2-normalised), so cosine similarity = dot product."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.vec.fit_transform(texts).toarray().astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.vec.transform([text]).toarray().astype(np.float32)[0]


class SentenceTransformerEmbedder:
    """Optional neural embeddings (pip install sentence-transformers). Not used by the tests."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
