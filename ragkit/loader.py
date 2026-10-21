"""Load documents and split them into overlapping word chunks with source metadata."""
from pathlib import Path


def load_documents(folder: str) -> list[dict]:
    docs = []
    for path in sorted(Path(folder).glob("*.md")) + sorted(Path(folder).glob("*.txt")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    if not docs:
        raise FileNotFoundError(f"No .md or .txt documents found in {folder}")
    return docs


def chunk_text(text: str, chunk_words: int = 100, overlap: int = 20) -> list[str]:
    if chunk_words <= overlap:
        raise ValueError("chunk_words must be larger than overlap")
    words = text.split()
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_documents(docs: list[dict], chunk_words: int = 100, overlap: int = 20) -> list[dict]:
    out = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"], chunk_words, overlap)):
            out.append({"source": d["source"], "chunk_id": i, "text": c})
    return out
