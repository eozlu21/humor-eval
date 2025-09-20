"""Model utilities and parsing helpers for humor-eval.

Updated to use Llama 3.2V chain-of-thought model and provide reasoning vs simple
answer prompting. Includes answer extraction utilities validated by tests.
"""
from __future__ import annotations

from typing import Tuple
import re
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL.Image import Image

MODEL_ID = "Xkev/Llama-3.2V-11B-cot"
CHOICES = ["A", "B", "C", "D", "E"]


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return processor, model


def _build_prompt(problem: str, answer_mode: str) -> str:
    base = (
        "You are an assistant solving a multiple-choice humor caption problem. "
        "Choices are A, B, C, D, E. Respond with ONLY the letter when possible."
    )
    if answer_mode == "reasoned":
        return (
            base
            + " First think step by step inside <think> </think> tags, then output the final choice wrapped in <answer> tags.\nQuestion: "
            + problem
        )
    else:
        return base + " Question: " + problem + "\nAnswer:"


def chat_infer(
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    image: Image,
    text: str,
    max_new_tokens: int = 64,
    answer_mode: str = "simple",
) -> str:
    prompt = _build_prompt(text, answer_mode)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip()


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_CONCLUSION_TAG_RE = re.compile(r"<conclusion>(.*?)</conclusion>", re.IGNORECASE | re.DOTALL)
_REASONING_TAG_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_BOX_LETTER_RE = re.compile(r"<\|begin_of_box\|>([A-E])<\|end_of_box\|>")
_LETTER_RE = re.compile(r"\b([A-E])\b")


def parse_model_response(resp: str) -> Tuple[str, str]:
    """Return (reasoning_segment, answer_segment_raw).

    Reasoning priority:
      1. <answer> or <conclusion> do NOT count as reasoning; we capture <think> or <reasoning>.
      2. Prefer <think> if present, else <reasoning>, else ''.
    Answer segment priority (raw segment returned for transparency):
      1. <answer>...</answer>
      2. <conclusion>...</conclusion>
      3. Single standalone letter (A-E) if entire trimmed output is that letter.
      4. Full response as last resort.
    """
    think_match = _THINK_TAG_RE.search(resp)
    reasoning_match = think_match or _REASONING_TAG_RE.search(resp)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    for regex in (_ANSWER_TAG_RE, _CONCLUSION_TAG_RE):
        m = regex.search(resp)
        if m:
            return reasoning, m.group(0)
    stripped = resp.strip()
    if stripped in CHOICES:
        return reasoning, stripped
    return reasoning, resp


def extract_answer(resp: str) -> str:
    # Priority 1: <answer> tag
    answer_match = _ANSWER_TAG_RE.search(resp)
    if answer_match:
        inner = answer_match.group(1)
        letter = _LETTER_RE.search(inner)
        if letter:
            return letter.group(1)
    # Priority 2: <conclusion> tag
    concl = _CONCLUSION_TAG_RE.search(resp)
    if concl:
        letter = _LETTER_RE.search(concl.group(1))
        if letter:
            return letter.group(1)
    # Priority 3: answer box token pattern
    box_matches = list(_BOX_LETTER_RE.finditer(resp))
    candidate_from_box = box_matches[-1].group(1) if box_matches else None
    # Priority 4: last standalone letter overall
    letters = _LETTER_RE.findall(resp)
    if letters:
        return letters[-1]
    if candidate_from_box:
        return candidate_from_box
    return "Unknown"


def summarize_device_allocation(model) -> str:
    try:
        alloc = []
        for name, p in model.named_parameters():
            if p.device not in [a[0] for a in alloc]:
                alloc.append((p.device, 0))
        return ", ".join(str(d) for d, _ in alloc) or "unknown"
    except Exception as e:  # pragma: no cover
        return f"error: {e}"

