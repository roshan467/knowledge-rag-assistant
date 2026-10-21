"""Command line: python cli.py ask "How many days of annual leave?"   |   python cli.py eval"""
import json
import sys

from ragkit.evaluate import run_eval, write_markdown
from ragkit.pipeline import RAGPipeline


def main():
    pipe = RAGPipeline()
    n = pipe.ingest("data/docs")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ask" and len(sys.argv) > 2:
        a = pipe.ask(" ".join(sys.argv[2:]))
        print(a.text, "\nsources:", a.sources, "\ntrace:", a.trace)
    elif cmd == "eval":
        report = run_eval(pipe, "tests/eval_cases.json")
        write_markdown(report, "EVAL_REPORT.md")
        print(f"{n} chunks indexed")
        print(json.dumps(report["summary"], indent=2))
        print("report written to EVAL_REPORT.md")
    else:
        print('usage: python cli.py ask "<question>" | python cli.py eval')


if __name__ == "__main__":
    main()
