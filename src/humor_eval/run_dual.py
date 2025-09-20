"""Utility to run evaluation in both simple and reasoned modes sequentially.

It will invoke the CLI twice (in-process) to avoid reloading the dataset twice,
while reusing the same loaded model for efficiency.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import List

from .data import load_entries
from tqdm import tqdm
from .models import load_model, chat_infer, parse_model_response, extract_answer, summarize_device_allocation, MODEL_ID
from .dataset_types import DatasetEntryResult


def run_mode(entries, processor, model, answer_mode: str, max_new_tokens: int, show_progress: bool, split: str) -> dict:
    ranking: List[DatasetEntryResult] = []
    matching: List[DatasetEntryResult] = []
    iterator = entries if not show_progress else tqdm(entries, desc=answer_mode)
    for entry in iterator:
        resp = chat_infer(processor, model, entry["images"], entry["problem"], max_new_tokens=max_new_tokens, answer_mode=answer_mode)
        reasoning, _ = parse_model_response(resp)
        extracted_answer = extract_answer(resp)
        is_correct = extracted_answer == entry["answer"]
        rec = DatasetEntryResult(
            contest_number=entry["contest_number"],
            problem=entry["problem"],
            correct_answer=entry["answer"],
            model_answer=resp,
            reasoning=reasoning,
            extracted_answer=extracted_answer,
            task=entry["task"],
            is_correct=is_correct,
        )
        (ranking if entry["task"] == "ranking" else matching).append(rec)

    def summarize(lst: List[DatasetEntryResult], task_name: str):
        total = len(lst)
        correct = sum(1 for r in lst if r["is_correct"])
        return {
            "total_entries": total,
            "correct_answers": correct,
            "accuracy": (correct / total) if total else 0.0,
            "answer_mode": answer_mode,
            "task": task_name,
            "split": split,
        }

    return {
        "ranking": {"summary": summarize(ranking, "ranking"), "results": ranking},
        "matching": {"summary": summarize(matching, "matching"), "results": matching},
    }


def run_dual(split: str = "test", max_new_tokens: int = 4096, output_dir: str = ".", show_progress: bool = True) -> tuple[str, str]:
    entries = load_entries(split)
    processor, model = load_model()
    try:
        print("Model device allocation:", summarize_device_allocation(model))
    except Exception as e:
        print(f"(Could not summarize devices: {e})")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_simple = Path(output_dir) / f"results_simple_{ts}.json"
    out_reasoned = Path(output_dir) / f"results_reasoned_{ts}.json"

    simple_data = run_mode(entries, processor, model, "simple", max_new_tokens, show_progress, split)
    reasoned_data = run_mode(entries, processor, model, "reasoned", max_new_tokens, show_progress, split)

    meta = {"split": split, "max_new_tokens": max_new_tokens}
    out_simple.write_text(json.dumps({"meta": meta, **simple_data}, indent=2))
    out_reasoned.write_text(json.dumps({"meta": meta, **reasoned_data}, indent=2))

    print(f"Saved simple mode results to {out_simple}")
    print(f"Saved reasoned mode results to {out_reasoned}")
    return str(out_simple), str(out_reasoned)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run dual-mode (simple + reasoned) evaluation")
    ap.add_argument("--split", default="test", help="Dataset split: test | test_hard | test_very_hard")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    args = ap.parse_args()
    run_dual(split=args.split, max_new_tokens=args.max_new_tokens, output_dir=args.output_dir, show_progress=not args.no_progress)
