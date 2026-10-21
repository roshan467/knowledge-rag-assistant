"""REST API: uvicorn api:app --reload   (docs at /docs).  Set USE_CLAUDE=1 and ANTHROPIC_API_KEY to use Claude."""
import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ragkit.pipeline import RAGPipeline

app = FastAPI(title="Knowledge RAG Assistant", version="1.0.0")
llm = None
if os.getenv("USE_CLAUDE") == "1":
    from ragkit.llm import AnthropicLLM
    llm = AnthropicLLM()
pipeline = RAGPipeline(llm=llm)
pipeline.ingest(os.getenv("DOCS_DIR", "data/docs"))


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(pipeline.chunks)}


@app.post("/ask")
def ask(q: Question):
    a = pipeline.ask(q.question)
    return {"answer": a.text, "grounded": a.grounded, "sources": a.sources, "trace": a.trace}
