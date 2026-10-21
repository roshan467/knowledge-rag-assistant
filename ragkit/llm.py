"""Answer generators. ExtractiveLLM is deterministic and offline (used in tests); AnthropicLLM calls Claude."""
import os
import re

PROMPT = """You are an enterprise knowledge assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: I don't know based on the provided documents.
Cite the source file name in square brackets after each fact.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in contexts)
    return PROMPT.format(context=ctx, question=question)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


class ExtractiveLLM:
    """Picks the context sentences that best overlap with the question (no model call, fully reproducible)."""

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[dict]) -> str:
        q = _tokens(question)
        scored = []
        for c in contexts:
            for s in _sentences(c["text"]):
                scored.append((len(q & _tokens(s)), s, c["source"]))
        scored.sort(key=lambda x: -x[0])
        best = [x for x in scored[: self.max_sentences] if x[0] > 0]
        if not best:
            return "I don't know based on the provided documents."
        return " ".join(f"{s} [{src}]" for _, s, src in best)


class AnthropicLLM:
    """Optional: needs `pip install anthropic` and ANTHROPIC_API_KEY. Not exercised by the offline tests."""

    def __init__(self, model: str | None = None, max_tokens: int = 400):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def generate(self, question: str, contexts: list[dict]) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": build_prompt(question, contexts)}])
        return msg.content[0].text.strip()
