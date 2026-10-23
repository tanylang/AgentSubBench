"""
Improved LoCoMo trace validator (on top of the official token-F1 criteria in finalize_locomot.py).

Known biases of the existing criteria:
  1. token-F1 is a bag-of-words comparison: verbose answers are penalized by precision — agent answers often carry explanations
     ("Melanie painted ... last year, i.e., in 2022 ..." vs gold "2022" scores only ~0.1),
     producing many format false-negatives that are "right but judged wrong by F1"
  2. Number-format sensitivity: "ten" vs "10", "10th" vs "10", and thousands separators all fail to match each other
  3. Word-order insensitivity: "not in 2022" scores the same as "in 2022" (the negation simply evaporates)
  4. The pass threshold of 0.5 sits directly on the most biased metric

Improved design (per-question cross-check of three metrics + two-level LLM judge):
  A. Deterministic metrics (zero cost, reproducible offline):
     - f1: official token-F1 (criteria unchanged, for comparability)
     - num_f1: number-normalized F1 (number words / ordinals / thousands separators → digits, then computed with the official f1)
     - contains: whether the gold word sequence appears as a contiguous fragment in pred (extractive check,
       fixes verbose false-negatives; includes a proximity guard for negations — "not in 2022" does not count as a hit)
     - rouge_l: LCS criteria (word-order-sensitive supplement)
  B. LLM judge (DeepSeek, temperature 0, batched JSON output):
     - strong deterministic signals decide directly (zero calls); only ambiguous questions go to the judge
     - verdict ∈ correct(1.0) / partial(0.5) / incorrect(0.0), with a one-line reason
     - adversarial (category 5) questions: always take the deterministic shortcut (the official refusal-phrase 0/1 substring
       decision); the judge layer is effectively unreachable for cat5 — the same criteria as the three-level decision on the
       offloading side (langgraph_env.py); the judge's adversarial rules only take effect if a future flow change
       removes the cat5 shortcut
  C. pass = per-question mean score >= PASS_THRESHOLD(0.5) (threshold semantics unchanged, score semantics changed)
     - pass_details keeps the official f1/threshold fields (comparable with the old criteria)
     - outputs the disagreement list between the judge and official F1 (rescued format false-negatives + bag-of-words false positives), for manual spot-checking

Usage:
    python runners/validate_locomot.py <trace_dir> [--no-llm] [--limit N] [--batch N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"),
           str(ROOT / "runners"), str(ROOT / "tasks" / "locomo_tasks" / "task_eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation import (SimpleTokenizer, eval_question_answering,  # noqa: E402 official scoring
                        f1_score, normalize_answer)
from nltk.stem import PorterStemmer  # noqa: E402 (same stemmer as the official one)
from adapter.locomot import load_locomo_samples  # noqa: E402
from adapter.model_config import (MAIN_MODEL_NAME, MAIN_BASE_URL,  # noqa: E402
                                  JUDGE_MODEL_NAME, JUDGE_BASE_URL,
                                  JUDGE_TEMPERATURE, JUDGE_SEED,
                                  JUDGE_MAX_TOKENS, JUDGE_THINKING_MODE)

_ps = PorterStemmer()
_tokenizer = SimpleTokenizer()
PASS_THRESHOLD = 0.5

# ── Number normalization (num_f1 metric) ────────────────────────────────────────
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def _norm_numbers(text: str) -> str:
    """Normalize number words / ordinals / thousands separators to Arabic digits (only affects num_f1, leaves official f1 untouched)."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text)                      # thousands separator 1,000 → 1000
    def _tens_ones(m):
        return str(_TENS[m.group(1).lower()] + _ONES[m.group(2).lower()])
    text = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
                  r"(one|two|three|four|five|six|seven|eight|nine)\b",
                  _tens_ones, text, flags=re.I)
    for w, v in {**_ONES, **_TENS}.items():
        text = re.sub(rf"\b{w}\b", str(v), text, flags=re.I)
    def _ord(m):
        return str(_ORDINALS.get(m.group(1).lower(), 0))
    text = re.sub(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
                  _ord, text, flags=re.I)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", text)        # 10th → 10
    return text


def _num_f1(pred: str, gold: str) -> float:
    return f1_score(_norm_numbers(pred), _norm_numbers(gold))


# ── Containment matching (extractive check + negation guard) ────────────────────
_NEG_WORDS = {"not", "no", "never", "wasnt", "didnt", "doesnt", "isnt",
              "arent", "couldnt", "wouldnt", "shouldnt", "without", "except"}


def _stem_tokens(text: str) -> list[str]:
    toks = _tokenizer.tokenize(normalize_answer(text), uncased=True)
    return [_ps.stem(t) for t in toks]


def contains_score(pred: str, gold: str) -> float | None:
    """Whether the gold word sequence appears as a contiguous fragment in pred.

    Returns None for "hit but possibly negated" (cannot be decided deterministically; leave it to the judge).
    """
    g = _stem_tokens(gold)
    p = _stem_tokens(pred)
    if not g or not p:
        return 0.0
    for i in range(len(p) - len(g) + 1):
        if p[i:i + len(g)] == g:
            # Negation guard: a negation word within the 2 words before the hit window → suspected negation
            window = set(p[max(0, i - 2):i])
            if window & _NEG_WORDS:
                return None
            return 1.0
    return 0.0


def rouge_l_f1(pred: str, gold: str) -> float:
    """LCS-based F1 (word-order sensitive; a lightweight replacement for the official rougel_score, no rouge dependency)."""
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    dp = [[0] * (len(g) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(g) + 1):
            dp[i][j] = (dp[i - 1][j - 1] + 1 if p[i - 1] == g[j - 1]
                        else max(dp[i - 1][j], dp[i][j - 1]))
    lcs = dp[len(p)][len(g)]
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


# ── LLM judge (second level of the two-level decision) ──────────────────────────
_REFUSAL_PATTERNS = ["no information available", "not mentioned",
                     "i don't know", "don't know", "cannot recall",
                     "can't recall", "can't remember", "cannot remember",
                     "not sure", "no record", "unable to find",
                     "doesn't say", "does not say", "not stated",
                     "never mentioned", "i'm not sure"]

JUDGE_SYSTEM = """You are a strict but fair grader for the LoCoMo long-term conversational memory benchmark. You judge whether a model's answer to a question is semantically correct given the gold answer.

Rules:
- Score "correct" if the model's answer contains the correct information, even if it adds extra explanation, hedging, or a longer narrative. Exact wording, number formatting ("ten" vs "10"), or answer order do not matter.
- Score "incorrect" if the core fact is wrong, contradicts the gold answer, or the model answers something else.
- Score "partial" only when the answer is right but incomplete (e.g., gold has several sub-answers and the model gives only some).
- Beware of negation: if the model says "not X" while gold is "X", that is incorrect, even though it contains the words of the gold answer.
- For questions that ask about information NOT present in the conversation (adversarial), the correct behavior is to say the information is unavailable; score "correct" for such refusals and "incorrect" for making up an answer.
- Dates/names/numbers must be factually equivalent to the gold (approximate year-only vs exact date is "partial" at best; a wrong number is "incorrect").

Output strict JSON only: {"judgments": [{"q": <question index>, "verdict": "correct"|"partial"|"incorrect", "reason": "<one short sentence>"}]}"""

_JUDGE_EXAMPLES = [
    ("Question: When did Melanie paint a sunrise?\nGold: 2022\nAnswer: Melanie said she painted the lake sunrise last year, i.e., in 2022, since she mentioned it on 8 May, 2023.",
     '{"q": 0, "verdict": "correct", "reason": "Verbose but states 2022 correctly."}'),
    ("Question: Where did they go?\nGold: Paris\nAnswer: They did not go to Paris; they went to Berlin instead.",
     '{"q": 1, "verdict": "incorrect", "reason": "Negates the gold answer."}'),
    ("Question: What is the name of Sam's dog?\nGold: Max\nAnswer: Sam has a cat named Max and a dog named Rex.",
     '{"q": 2, "verdict": "incorrect", "reason": "Max is the cat, dog is Rex."}'),
    ("Question: When was the wedding?\nGold: 7 May 2023\nAnswer: In May 2023.",
     '{"q": 3, "verdict": "partial", "reason": "Right month/year but missing the day."}'),
    ("Question: What did they buy in May?\nGold: no information available\nAnswer: The conversation does not mention what they bought in May.",
     '{"q": 4, "verdict": "correct", "reason": "Correct refusal: no information available."}'),
]

_VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def _load_api_key() -> str:
    try:
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        return cfg.get("deepseek_api_key", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY", "")


def _judge_batch(items: list[dict], *, model: str, base_url: str,
                 retries: int = 3) -> dict[int, dict]:
    """Judge a batch of questions in one call, returning {q_idx: {verdict, reason}}; degrades to {} after failed retries."""
    import litellm
    user_lines = ["Judge the following answers. For each, output the verdict JSON."
                  "\n\nExamples:\n"]
    for qa, out in _JUDGE_EXAMPLES:
        user_lines.append(f"{qa}\nExpected: {out}\n")
    user_lines.append("Now judge these (same JSON format, one entry per question):\n")
    for it in items:
        cat_note = (" (adversarial: asks about info possibly absent from the "
                    "conversation)" if it["category"] == 5 else "")
        user_lines.append(f'--- q={it["idx"]} category={it["category"]}{cat_note}\n'
                          f'Question: {it["question"]}\nGold: {it["gold"]}\n'
                          f'Answer: {it["pred"][:800]}\n')
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": "\n".join(user_lines)}]
    for attempt in range(retries):
        # Temperature/seed are fixed (reproducible), thinking mode on by default (injected via config);
        # drop the thinking switch on the final attempt (fallback when some gateways' json mode is incompatible with thinking)
        extra = JUDGE_THINKING_MODE if attempt < retries - 1 else None
        try:
            resp = litellm.completion(
                model=f"openai/{model}", api_base=base_url, messages=messages,
                temperature=JUDGE_TEMPERATURE, seed=JUDGE_SEED,
                max_completion_tokens=JUDGE_MAX_TOKENS,
                response_format={"type": "json_object"},
                extra_body=extra)
            content = resp.choices[0].message.content or ""
            content = re.sub(r"^```(json)?|```$", "", content.strip()).strip()
            data = json.loads(content)
            out = {}
            for j in data.get("judgments", []):
                v = j.get("verdict", "")
                if v in _VERDICT_SCORE:
                    out[int(j["q"])] = {"verdict": v, "reason": str(j.get("reason", ""))[:200]}
            return out
        except Exception as e:
            print(f"    [judge] attempt {attempt + 1}/{retries} failed: {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
            time.sleep(10 * (attempt + 1))
    return {}


def _shortcut_score(category: int, f1: float, contains: float | None,
                    pred: str) -> float | None:
    """Deterministic shortcut: only high-confidence signals decide directly; everything else (including zero-overlap rewrites) goes to the judge.

    Difference from the old two-stage version: "f1==0 and no containment" is no longer automatically scored 0 — that would wrongly kill
    rewrite answers with zero word overlap (gold "hard work's paying off" vs pred "store was
    growing"); such questions must be judged semantically by the judge.
    """
    if not pred.strip():
        return 0.0                      # empty answer
    if category == 5:
        return float(f1)                # official refusal phrases 0/1 (deterministic rule)
    if contains == 1 and f1 >= 0.4:
        return 1.0                      # strong hit (with negation guard)
    return None


# ── Single trace validation ─────────────────────────────────────────
def evaluate_trace(trace_path: Path, *, use_llm: bool = True,
                   batch: int = 50) -> dict:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    meta = data.get("locomo") or {}
    sample_id = meta.get("sample_id") or trace_path.stem
    answers = meta.get("answers") or []
    if not answers:
        return {"trace": trace_path.name, "error": "trace has no locomo.answers"}
    samples = load_locomo_samples(sample_ids=[sample_id])
    if not samples:
        return {"trace": trace_path.name, "error": f"data file has no {sample_id}"}
    gold_by_q = {q.question: q for q in samples[0].qas}
    read_ids = set(meta.get("read_ids") or [])

    # ── Official f1 (zero criteria drift: same construction as finalize_locomot.py) ──
    qas, rows = [], []
    recall_hits = recall_total = 0
    for a in answers:
        gold = gold_by_q.get(a.get("question", ""))
        if gold is None:
            continue
        pred = str(a.get("pred", ""))
        qas.append({"answer": str(gold.answer), "category": int(gold.category),
                    "prediction": pred, "evidence": gold.evidence})
        rows.append({"idx": a.get("idx"), "question": str(a.get("question", "")),
                     "category": int(gold.category), "pred": pred[:600],
                     "gold": str(gold.answer)})
        if gold.evidence:
            recall_hits += len(set(gold.evidence) & read_ids)
            recall_total += len(gold.evidence)
    if not qas:
        return {"trace": trace_path.name, "error": "no questions could be aligned (gold missing)"}
    f1_scores = eval_question_answering(qas, eval_key="prediction", metric="f1")[0]

    # ── Three deterministic metrics ──
    for row, f1 in zip(rows, f1_scores):
        row["f1"] = round(f1, 4)
        row["num_f1"] = round(_num_f1(row["pred"], row["gold"]), 4)
        row["contains"] = contains_score(row["pred"], row["gold"])
        row["rouge_l"] = round(rouge_l_f1(row["pred"], row["gold"]), 4)
        row["det"] = _shortcut_score(row["category"], f1, row["contains"],
                                     row["pred"])

    # ── LLM judge (only for ambiguous questions) ──
    judged = 0
    if use_llm:
        todo = [r for r in rows if r["det"] is None]
        model = JUDGE_MODEL_NAME
        base_url = JUDGE_BASE_URL
        if todo:
            print(f"    [judge] {len(todo)}/{len(rows)} questions sent to the judge "
                  f"({model})", flush=True)
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            verdicts = _judge_batch(chunk, model=model, base_url=base_url)
            for r in chunk:
                v = verdicts.get(r["idx"])
                if v is not None:
                    r["llm"] = _VERDICT_SCORE[v["verdict"]]
                    r["verdict"] = v["verdict"]
                    r["reason"] = v["reason"]
                    judged += 1

    # ── Per-question scores + pass decision ──
    for r in rows:
        if "llm" in r:
            r["score"] = r["llm"]
            r["source"] = "llm"
        else:
            # Fall back to deterministic metrics when there is no judge result (--no-llm or judge call failure)
            r["score"] = (r["det"] if r["det"] is not None
                          else max(r["f1"], r["num_f1"], r["contains"] or 0.0))
            r["source"] = "rule" if r["det"] is not None else "metric-fallback"
        r["contains"] = r["contains"] if r["contains"] is not None else None
        # Per-question pass: strict criteria — only fully correct (score == 1.0) counts as pass, partial counts as fail
        r["pass"] = "success" if r["score"] == 1.0 else "fail"
    avg = sum(r["score"] for r in rows) / len(rows)
    success = avg >= PASS_THRESHOLD
    avg_f1 = sum(r["f1"] for r in rows) / len(rows)

    cat_scores: dict[str, list[float]] = {}
    for r in rows:
        cat_scores.setdefault(str(r["category"]), []).append(r["score"])
    # Disagreement list: judge vs official f1 (0.5 threshold criteria)
    rescues = [r for r in rows if "llm" in r and r["f1"] < 0.5 and r["llm"] == 1.0]
    demotes = [r for r in rows if "llm" in r and r["f1"] >= 0.5 and r["llm"] == 0.0]

    # No longer write a top-level pass — per-question annotation lives in pass_details.per_question[i].pass (strict criteria)
    data.pop("pass", None)
    data["pass_details"] = {
        "success": success,
        "score_avg": round(avg, 4),
        "threshold": PASS_THRESHOLD,
        "f1": round(avg_f1, 4),          # official-criteria mean (comparable with old results)
        "num_f1_avg": round(sum(r["num_f1"] for r in rows) / len(rows), 4),
        "contains_rate": round(sum(1 for r in rows if r["contains"] == 1) / len(rows), 4),
        "llm_judged": judged,
        "num_questions": len(rows),
        "category_counts": {k: len(v) for k, v in cat_scores.items()},
        "category_avg_score": {k: round(sum(v) / len(v), 4)
                               for k, v in cat_scores.items()},
        "overall_recall": round(recall_hits / recall_total, 4) if recall_total else None,
        "rescued_by_llm": len(rescues),   # wrongly judged by official f1, rescued by the judge
        "demoted_by_llm": len(demotes),   # official f1 inflated, corrected by the judge
        "per_question": rows,
    }
    trace_path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")
    return {"trace": trace_path.name, "success": success,
            "score": round(avg, 4), "f1": round(avg_f1, 4),
            "judged": judged, "rescued": len(rescues), "demoted": len(demotes),
            "rescue_examples": rescues[:5], "demote_examples": demotes[:5]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved LoCoMo trace validator")
    parser.add_argument("trace_dir", help="trace directory")
    parser.add_argument("--no-llm", action="store_true",
                        help="use only deterministic metrics (no LLM judge calls)")
    parser.add_argument("--limit", type=int, default=0, help="validate only the first N traces")
    parser.add_argument("--batch", type=int, default=50, help="judge batch size (questions per call)")
    args = parser.parse_args()

    if not args.no_llm:
        os.environ.setdefault("OPENAI_API_KEY", _load_api_key())
        os.environ.setdefault("LITELLM_REQUEST_TIMEOUT", "600")

    trace_dir = Path(args.trace_dir)
    files = sorted(trace_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]

    results = []
    for fp in files:
        print(f"[{fp.stem}]", flush=True)
        r = evaluate_trace(fp, use_llm=not args.no_llm, batch=args.batch)
        results.append(r)
        print(f"    pass={r.get('success')} score={r.get('score')} "
              f"official_f1={r.get('f1')} judged={r.get('judged')} "
              f"rescued={r.get('rescued')} demoted={r.get('demoted')}", flush=True)
        if r.get("rescue_examples"):
            ex = r["rescue_examples"][0]
            print(f"    rescued example: Q{ex['idx']} gold={ex['gold'][:40]!r} "
                  f"f1={ex['f1']} -> correct ({ex['reason'][:60]})", flush=True)
        if r.get("demote_examples"):
            ex = r["demote_examples"][0]
            print(f"    demoted example: Q{ex['idx']} gold={ex['gold'][:40]!r} "
                  f"f1={ex['f1']} -> incorrect ({ex['reason'][:60]})", flush=True)

    ok = [r for r in results if "success" in r]
    succ = sum(1 for r in ok if r["success"])
    old_pass = 0
    for fp, r in zip(files, results):
        if "f1" in r:
            old_pass += 1 if r["f1"] >= PASS_THRESHOLD else 0
    print(f"\n  Done: {len(ok)} traces (new criteria success={succ}, "
          f"old official F1 criteria success={old_pass})")
    if ok:
        print(f"  avg score: {sum(r['score'] for r in ok) / len(ok):.4f} | "
              f"avg official F1: {sum(r['f1'] for r in ok) / len(ok):.4f} | "
              f"total rescued: {sum(r['rescued'] for r in ok)} | "
              f"total demoted: {sum(r['demoted'] for r in ok)}")
    # Put the summary report outside the trace directory (to avoid polluting the collected *.json directory)
    report = {"dir": str(trace_dir),
              "results": [{k: v for k, v in r.items()
                           if k not in ("rescue_examples", "demote_examples")}
                          for r in results]}
    report_path = trace_dir.parent / f"validation_report_{trace_dir.name}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      default=str), encoding="utf-8")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
