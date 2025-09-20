"""Run simple (no reasoning prompt) evaluation over a split.

Outputs a single JSON file with summary + per-item records.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import List

from .data import load_entries
from tqdm import tqdm
from .models import load_model, chat_infer, parse_model_response, extract_answer, summarize_device_allocation
from .dataset_types import DatasetEntryResult

def run_simple(split: str = "test", max_new_tokens: int = 512, output_dir: str = ".", show_progress: bool = True) -> str:
    entries = load_entries(split)
    processor, model = load_model()
    try:
        print("Model device allocation:", summarize_device_allocation(model))
    except Exception as e:  # pragma: no cover
        print(f"(Could not summarize devices: {e})")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(output_dir) / f"results_simple_only_{split}_{ts}.json"

    results_ranking: List[DatasetEntryResult] = []
    results_matching: List[DatasetEntryResult] = []
    iterator = entries if not show_progress else tqdm(entries, desc=f"simple:{split}")
    for entry in iterator:
        resp = chat_infer(processor, model, entry["images"], entry["problem"], max_new_tokens=max_new_tokens, answer_mode="simple")
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
        if entry["task"] == "ranking":
            results_ranking.append(rec)
        else:
            results_matching.append(rec)

    def summarize(lst: List[DatasetEntryResult]):
        total = len(lst)
        correct = sum(1 for r in lst if r["is_correct"])
        acc = correct / total if total else 0.0
        return {
            "total_entries": total,
            "correct_answers": correct,
            "accuracy": acc,
            "answer_mode": "simple",
            "split": split,
            "task": lst[0]["task"] if total else None,
        }

    payload = {
        "ranking": {
            "summary": summarize(results_ranking),
            "results": results_ranking,
        },
        "matching": {
            "summary": summarize(results_matching),
            "results": results_matching,
        },
        "meta": {
            "split": split,
            "max_new_tokens": max_new_tokens,
            "generated_at": ts,
        }
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved simple evaluation to {out_path}")
    return str(out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run simple-mode evaluation only")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar")
    args = ap.parse_args()
    run_simple(split=args.split, max_new_tokens=args.max_new_tokens, output_dir=args.output_dir, show_progress=not args.no_progress)
