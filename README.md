# humor-eval

Evaluation helpers for humor caption multiple-choice tasks (matching & ranking)
using multi-modal LLMs.

## What's New

* Dataset upgraded to `newyccku/caption_dataset_rl_v5` supporting splits:
	* `test`
	* `test_hard`
	* `test_very_hard`
* Model switched to `Xkev/Llama-3.2V-11B-cot` (chain-of-thought vision model).
* Dual-mode evaluation ("simple" vs "reasoned") with reasoning extracted from
	`<think>` tags and final answers from `<answer>` tags when present.

## Installation

Create / activate a Python 3.13 environment, then install requirements:

```
pip install -r requirements.txt
```

Install the package (editable optional):

```
pip install -e .
```

## Quick Single Example

```
python -m humor_eval.cli --split test --index 0 --max_new_tokens 256
```

## Dual Evaluation (simple + reasoned)

Runs both modes sequentially, saving two JSON files with summaries & per-item
results.

```
python -m humor_eval.run_dual --split test --max_new_tokens 512
```

You can choose harder splits:

```
python -m humor_eval.run_dual --split test_hard
python -m humor_eval.run_dual --split test_very_hard
```

Each run produces files like:

```
results_simple_YYYYMMDD_HHMMSS.json
results_reasoned_YYYYMMDD_HHMMSS.json
```

## Output JSON Structure

Simple schema (per file):
```
{
	"summary": {"total_entries": int, "correct_answers": int, "accuracy": float, "answer_mode": "simple"|"reasoned"},
	"results": [
		{
			"contest_number": int,
			"problem": str,
			"correct_answer": "A"-"E",
			"model_answer": str,          # raw full model output
			"reasoning": str,             # extracted <think> content (may be empty)
			"extracted_answer": "A"-"E"|"Unknown",
			"task": "matching"|"ranking",
			"is_correct": bool
		}, ...
	]
}
```

## Notes

* The model prompts differ between modes:
	* simple: requests only the final letter.
	* reasoned: asks for `<think>` reasoning then `<answer>` tag.
* Answer extraction priority: `<answer>` tag > last standalone letter A-E;
	fallback `Unknown`.
* For large images / memory limits you can reduce `--max_new_tokens`.

## Development

Run tests (parsing helpers):
```
pytest -q
```

## License

See `LICENSE`.
