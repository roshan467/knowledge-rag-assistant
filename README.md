# Enterprise Knowledge RAG Assistant (with evaluation harness)

A Retrieval-Augmented Generation (RAG) assistant that answers employee questions from company policy documents,
cites its sources, and refuses to guess when the documents do not contain the answer. It includes a test harness
that measures answer quality with defined test cases, which is the part most demos skip.

## Workflow
`retrieve` (embed the question, cosine-similarity search) -> `guardrail` (refuse if best match is below a threshold)
-> `generate` (answer from the retrieved context only) -> `validate` (answer is grounded and cited). Every answer
returns its sources, similarity scores and the step trace.

## Components
| File | Purpose |
|---|---|
| `ragkit/loader.py` | Load `.md`/`.txt` files, split into overlapping word chunks with source metadata |
| `ragkit/embeddings.py` | TF-IDF embeddings (offline default); optional `SentenceTransformerEmbedder` |
| `ragkit/vectorstore.py` | Cosine-similarity vector store with save/load |
| `ragkit/llm.py` | `ExtractiveLLM` (offline, deterministic) and `AnthropicLLM` (Claude, optional) with a grounded prompt |
| `ragkit/pipeline.py` | The RAG workflow above |
| `ragkit/evaluate.py` | Runs test cases; reports retrieval hit rate, keyword coverage, refusal accuracy, failures |
| `api.py` | FastAPI service: `GET /health`, `POST /ask` |
| `data/docs/` | Five synthetic company policy documents (leave, expenses, security and access, cloud, incidents) |

## Run
```bash
pip install -r requirements.txt
python cli.py ask "How many days of annual leave do employees get?"
python cli.py eval                      # writes EVAL_REPORT.md
python -m unittest discover -s tests -v # 7 tests
uvicorn api:app --reload                # http://localhost:8000/docs
docker build -t knowledge-rag . && docker run -p 8000:8000 knowledge-rag
```
Use Claude for answers: `pip install anthropic`, set `ANTHROPIC_API_KEY`, then `USE_CLAUDE=1 uvicorn api:app`.

## Evaluation and improvement log
13 test cases: 10 answerable questions (expected source file and key facts) and 3 out-of-scope questions
(the assistant must refuse).

| Setting | Pass rate | Retrieval hit rate | Refusal accuracy |
|---|---|---|---|
| Baseline (similarity threshold 0.12) | 84.6% (11/13) | 90% | 66.7% |
| Tuned (threshold 0.16, grid search over chunk size, overlap, threshold, top-k) | 92.3% (12/13) | 90% | 100% |

**What the failures showed.** At the baseline threshold the assistant answered an out-of-scope question from loosely
related text, so the guardrail threshold was raised. The one remaining failure ("What is the minimum password length?")
is a vocabulary mismatch: the policy says "at least 14 characters long" and TF-IDF only matches shared words. Switching to
`SentenceTransformerEmbedder` (semantic embeddings) is the next improvement to test.

## Limits (be honest in interviews)
Documents are synthetic. TF-IDF is a lexical baseline, not a neural vector database; swap in a neural embedder or a
vector database (Chroma, FAISS, pgvector) behind the same `fit_transform` / `embed_query` interface. `api.py`, the Claude
generator and the Docker image need to be run on your machine; the offline pipeline and tests run without them.
