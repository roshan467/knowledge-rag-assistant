"""Minimal vector store: cosine-similarity search over normalised vectors, with save and load."""
import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.meta: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        self.vectors, self.meta = vectors, metadata

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[tuple[float, dict]]:
        if len(self.meta) == 0:
            return []
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        scores = self.vectors @ (query_vec / norm)
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.meta[i]) for i in order]

    def save(self, folder: str) -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        np.save(Path(folder) / "vectors.npy", self.vectors)
        (Path(folder) / "meta.json").write_text(json.dumps(self.meta))

    @classmethod
    def load(cls, folder: str) -> "VectorStore":
        store = cls()
        store.vectors = np.load(Path(folder) / "vectors.npy")
        store.meta = json.loads((Path(folder) / "meta.json").read_text())
        return store
