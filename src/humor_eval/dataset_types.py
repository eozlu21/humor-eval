from typing import Literal, TypedDict
from PIL.Image import Image

class DatasetEntry(TypedDict):
    images: Image
    contest_number: int
    problem: str
    answer: Literal["A", "B", "C", "D", "E"]
    task: Literal["matching", "ranking"]

class DatasetEntryResult(TypedDict):
    contest_number: int
    problem: str
    correct_answer: Literal["A", "B", "C", "D", "E"]
    model_answer: str
    reasoning: str
    extracted_answer: Literal["A", "B", "C", "D", "E", "Unknown"]
    task: Literal["matching", "ranking"]
    is_correct: bool
