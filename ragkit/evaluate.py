"""Functional test harness for the assistant: runs defined test cases, scores them, lists failures for analysis."""
import json
from pathlib import Path


def run_eval(pipeline, cases_path: str) -> dict:
    cases = json.loads(Path(cases_path).read_text())
    rows = []
    for c in cases:
        ans = pipeline.ask(c["question"])
        row = {"id": c["id"], "question": c["question"], "answerable": c["answerable"], "answer": ans.text}
        if c["answerable"]:
            hit = c["expected_source"] in [s["source"] for s in ans.sources]
            kws = c["keywords"]
            cov = sum(k.lower() in ans.text.lower() for k in kws) / len(kws)
            row.update(retrieval_hit=hit, keyword_coverage=round(cov, 2), passed=hit and cov >= 0.5)
            row["failure"] = "" if row["passed"] else ("wrong source retrieved" if not hit else "answer missing key facts")
        else:
            ok = not ans.grounded
            row.update(correct_refusal=ok, passed=ok, failure="" if ok else "answered an out-of-scope question")
        rows.append(row)

    ans_rows = [r for r in rows if r["answerable"]]
    refuse_rows = [r for r in rows if not r["answerable"]]
    summary = {
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in ans_rows) / len(ans_rows), 3) if ans_rows else None,
        "avg_keyword_coverage": round(sum(r["keyword_coverage"] for r in ans_rows) / len(ans_rows), 3) if ans_rows else None,
        "refusal_accuracy": round(sum(r["correct_refusal"] for r in refuse_rows) / len(refuse_rows), 3) if refuse_rows else None,
    }
    return {"summary": summary, "results": rows, "failures": [r for r in rows if not r["passed"]]}


def write_markdown(report: dict, path: str) -> None:
    s = report["summary"]
    lines = ["# Evaluation report", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s.items()]
    lines += ["", "## Failures and improvement notes"]
    if not report["failures"]:
        lines.append("None.")
    for f in report["failures"]:
        lines.append(f"- **{f['id']}** ({f['failure']}): {f['question']}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
