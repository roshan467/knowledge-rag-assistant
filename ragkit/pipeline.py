"""RAG workflow: retrieve -> guardrail -> generate -> validate. Each step is recorded in a trace."""
from dataclasses import dataclass, field

from .embeddings import TfidfEmbedder
from .llm import ExtractiveLLM
from .loader import chunk_documents, load_documents
from .vectorstore import VectorStore

NO_ANSWER = "I don't know based on the provided documents."


@dataclass
class Answer:
    question: str
    text: str
    grounded: bool
    sources: list = field(default_factory=list)  # [{"source", "chunk_id", "score"}]
    trace: list = field(default_factory=list)


class RAGPipeline:
    def __init__(self, embedder=None, llm=None, chunk_words=100, overlap=20, top_k=3, min_score=0.16):
        self.embedder = embedder or TfidfEmbedder()
        self.llm = llm or ExtractiveLLM()
        self.chunk_words, self.overlap, self.top_k, self.min_score = chunk_words, overlap, top_k, min_score
        self.store = VectorStore()
        self.chunks: list[dict] = []

    def ingest(self, folder: str) -> int:
        self.chunks = chunk_documents(load_documents(folder), self.chunk_words, self.overlap)
        vectors = self.embedder.fit_transform([c["text"] for c in self.chunks])
        self.store.add(vectors, self.chunks)
        return len(self.chunks)

    def ask(self, question: str) -> Answer:
        trace = ["retrieve"]
        hits = self.store.search(self.embedder.embed_query(question), self.top_k)
        sources = [{"source": m["source"], "chunk_id": m["chunk_id"], "score": round(s, 3)} for s, m in hits]

        trace.append("guardrail")
        if not hits or hits[0][0] < self.min_score:   # nothing relevant enough: refuse instead of guessing
            return Answer(question, NO_ANSWER, False, sources, trace + ["refused"])

        trace.append("generate")
        text = self.llm.generate(question, [m for _, m in hits])

        trace.append("validate")
        grounded = bool(text) and not text.startswith("I don't know")
        return Answer(question, text, grounded, sources, trace)
