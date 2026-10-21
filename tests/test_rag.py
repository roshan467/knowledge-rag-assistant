import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ragkit.evaluate import run_eval
from ragkit.loader import chunk_text
from ragkit.pipeline import NO_ANSWER, RAGPipeline
from ragkit.vectorstore import VectorStore

DOCS = str(ROOT / "data" / "docs")
CASES = str(ROOT / "tests" / "eval_cases.json")


def make_pipeline(**kw):
    p = RAGPipeline(**kw)
    p.ingest(DOCS)
    return p


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = make_pipeline()

    def test_chunking_overlaps_and_covers_text(self):
        words = [f"w{i}" for i in range(250)]
        chunks = chunk_text(" ".join(words), chunk_words=100, overlap=20)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].split()[-20:], chunks[1].split()[:20])   # 20-word overlap
        self.assertEqual(chunks[-1].split()[-1], "w249")                    # nothing dropped

    def test_chunk_params_validated(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", chunk_words=10, overlap=10)

    def test_vector_store_ranks_and_roundtrips(self):
        store = VectorStore()
        vecs = np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)
        store.add(vecs, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
        top = store.search(np.array([1, 0], dtype=np.float32), k=2)
        self.assertEqual([m["id"] for _, m in top], ["a", "c"])
        with tempfile.TemporaryDirectory() as d:
            store.save(d)
            again = VectorStore.load(d)
            self.assertEqual(again.search(np.array([0, 1], dtype=np.float32), k=1)[0][1]["id"], "b")

    def test_retrieves_correct_source_with_citation(self):
        a = self.pipe.ask("How quickly must a lost laptop be reported?")
        self.assertTrue(a.grounded)
        self.assertEqual(a.sources[0]["source"], "security_access_policy.md")
        self.assertIn("1 hour", a.text)
        self.assertIn("[security_access_policy.md]", a.text)

    def test_refuses_out_of_scope_question(self):
        a = self.pipe.ask("Who won the cricket world cup?")
        self.assertFalse(a.grounded)
        self.assertEqual(a.text, NO_ANSWER)
        self.assertIn("refused", a.trace)

    def test_trace_records_workflow_steps(self):
        a = self.pipe.ask("How many days of annual leave do employees get?")
        self.assertEqual(a.trace, ["retrieve", "guardrail", "generate", "validate"])

    def test_evaluation_meets_quality_bar(self):
        report = run_eval(self.pipe, CASES)
        s = report["summary"]
        self.assertGreaterEqual(s["retrieval_hit_rate"], 0.9)
        self.assertEqual(s["refusal_accuracy"], 1.0)
        self.assertGreaterEqual(s["pass_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
