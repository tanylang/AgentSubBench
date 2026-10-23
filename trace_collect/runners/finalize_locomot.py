"""LocoMo trace finalization: official evaluation criteria + pass write-back.

Evaluation calls the official scoring code directly (tasks/locomo_tasks/task_eval/evaluation.py,
vendored from the official snap-research/LoCoMo repository; the only modification is lazy-loading of bert_score —
that metric requires torch and is not triggered on the default F1 path). Zero porting drift in the criteria.

- eval_question_answering(qas, eval_key='prediction', metric='f1')
  · category 3: gold takes the part before the semicolon (official semantics)
  · category 1: split on commas into multiple sub-answers, average of max-F1
  · category 5: refusal phrases scored as 0/1
- Dual criteria: official token-F1 (primary criterion, pass decision) + official BERTScore (semantic similarity,
  mitigating format false-negatives like "ten years ago vs 10 years ago"; requires bert-score+torch,
  gracefully degrades to F1-only when missing)
- pass = mean F1 >= PASS_THRESHOLD(0.5) recorded as "success", otherwise "fail"
- Retrieval recall = |evidence ∩ read_ids| / |evidence| (skipped when evidence is empty)

Usage:
    python runners/finalize_locomot.py <trace_dir> [--limit N] [--metric f1|bert_score|both]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"),
          str(ROOT / "runners"), str(ROOT / "tasks" / "locomo_tasks" / "task_eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation import eval_question_answering  # noqa: E402 (official scorer, vendored version)
from adapter.locomot import load_locomo_samples  # noqa: E402

PASS_THRESHOLD = 0.5


def _bert_scores(qas) -> list[float] | None:
    """Official BERTScore per-question scoring; returns None when bert-score/torch are missing."""
    import os
    # HF model downloads go through the mirror (deberta-xlarge-mnli is ~1.4GB on first download)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from evaluation import bert_score
        return [bert_score(q["prediction"], q["answer"]) for q in qas]
    except ImportError:
        return None


def evaluate_locomo_trace(trace_path: Path, metric: str = "both") -> dict:
    """Read trace → build official qas rows → official eval_question_answering → write pass back."""
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    meta = data.get("locomo") or {}
    sample_id = meta.get("sample_id") or trace_path.stem
    answers = meta.get("answers") or []
    if not answers:
        return {"trace": trace_path.name, "error": "trace has no locomo.answers"}
    samples = load_locomo_samples(sample_ids=[sample_id])
    if not samples:
        return {"trace": trace_path.name, "error": f"data file has no {sample_id}"}
    sample = samples[0]
    gold_by_q = {q.question: q for q in sample.qas}
    read_ids = set(meta.get("read_ids") or [])

    qas = []
    per_q: list[dict] = []
    recall_hits = recall_total = 0
    for a in answers:
        gold = gold_by_q.get(a.get("question", ""))
        if gold is None:
            continue
        qas.append({
            "answer": str(gold.answer),          # official bert_score has no str guard; numeric answers would crash
            "category": int(gold.category),
            "prediction": str(a.get("pred", "")),
            "evidence": gold.evidence,
        })
        recall = None
        if gold.evidence:
            hits = len(set(gold.evidence) & read_ids)
            recall = hits / len(gold.evidence)
            recall_hits += hits
            recall_total += len(gold.evidence)
        per_q.append({"idx": a.get("idx"), "category": gold.category,
                      "pred": str(a.get("pred", ""))[:300],
                      "gold": str(gold.answer)[:300]})

    if not qas:
        return {"trace": trace_path.name, "error": "no questions could be aligned (gold missing)"}
    # Official return is (all_scores, F1, RL, lens)
    f1_scores = eval_question_answering(qas, eval_key="prediction", metric="f1")[0]
    avg_f1 = sum(f1_scores) / len(f1_scores)
    success = avg_f1 >= PASS_THRESHOLD
    # BERTScore dual criterion (None when dependencies are missing, graceful degradation)
    bert_scores = None
    if metric in ("both", "bert_score"):
        bert_scores = _bert_scores(qas)

    cat_scores: dict[str, list[float]] = {}
    for q, s in zip(per_q, f1_scores):
        cat_scores.setdefault(str(q["category"]), []).append(s)
        q["f1"] = round(s, 4)
    if bert_scores is not None:
        for q, s in zip(per_q, bert_scores):
            q["bert_f1"] = round(s, 4)
    for q in per_q:
        q["recall"] = None
    # recall attached to the corresponding question (per_q order matches qas; non-official metric, computed separately)
    for a, q in zip(answers, per_q):
        gold = gold_by_q.get(a.get("question", ""))
        if gold is not None and gold.evidence:
            hits = len(set(gold.evidence) & read_ids)
            q["recall"] = round(hits / len(gold.evidence), 4)

    data["pass"] = "success" if success else "fail"
    details = {
        "success": success,
        "f1": round(avg_f1, 4),
        "threshold": PASS_THRESHOLD,
        "num_questions": len(per_q),
        "category_counts": {k: len(v) for k, v in cat_scores.items()},
        "category_avg_f1": {k: round(sum(v) / len(v), 4)
                            for k, v in cat_scores.items()},
        "overall_recall": round(recall_hits / recall_total, 4) if recall_total else None,
        "per_question": per_q,
    }
    if bert_scores is not None:
        details["bert_f1"] = round(sum(bert_scores) / len(bert_scores), 4)
    else:
        details["bert_f1"] = None
    data["pass_details"] = details
    trace_path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")
    return {"trace": trace_path.name, "success": success, "f1": round(avg_f1, 4)}


def finalize_locomot_dir(trace_dir: str, limit: int = 0, metric: str = "both") -> None:
    """Directory loop: glob *.json → evaluate → stats."""
    files = sorted(Path(trace_dir).glob("*.json"))
    if limit:
        files = files[:limit]
    stats = {"evaluated": 0, "success": 0, "fail": 0, "errors": 0}
    for i, fp in enumerate(files, 1):
        r = evaluate_locomo_trace(fp, metric)
        if "error" in r:
            stats["errors"] += 1
            print(f"    [{i}/{len(files)}] {r['trace']}: ERROR {r['error']}")
        else:
            stats["evaluated"] += 1
            stats["success" if r["success"] else "fail"] += 1
            print(f"    [{i}/{len(files)}] {r['trace']}: "
                  f"success={r['success']} f1={r['f1']}")
    print(f"[finalize] evaluated {stats['evaluated']} traces, "
          f"success={stats['success']}, fail={stats['fail']}, "
          f"errors={stats['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LocoMo trace finalization (official-criteria evaluation + pass write-back)")
    parser.add_argument("trace_dir")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metric", default="both",
                        choices=["f1", "bert_score", "both"])
    args = parser.parse_args()
    finalize_locomot_dir(args.trace_dir, args.limit, args.metric)


if __name__ == "__main__":
    main()
