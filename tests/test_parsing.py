import pytest
from humor_eval.models import parse_model_response, extract_answer


def test_parse_simple():
    reasoning, answer_seg = parse_model_response("B")
    assert reasoning == ""
    assert answer_seg == "B"
    assert extract_answer("B") == "B"


def test_parse_reasoned_with_tags():
    resp = "<think>some reasoning here</think><answer> C </answer>"
    reasoning, answer_seg = parse_model_response(resp)
    assert "reasoning" in reasoning
    assert "<answer>" in answer_seg  # raw segment keeps answer tag text
    assert extract_answer(resp) == "C"


def test_priority_order():
    resp = "Random text <|begin_of_box|>D<|end_of_box|> also <answer> B </answer>"
    # answer tag should override earlier box letter
    assert extract_answer(resp) == "B"


def test_last_letter_fallback():
    resp = "Some analysis chooses A then revises to E"
    assert extract_answer(resp) == "E"


def test_unknown():
    resp = "No valid choice here: Z or Q"
    assert extract_answer(resp) == "Unknown"
