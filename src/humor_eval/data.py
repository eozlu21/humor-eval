from typing import Iterable, List
from datasets import load_dataset
from .dataset_types import DatasetEntry

DATASET_NAME = "newyccku/caption_dataset_rl_v5"
DEFAULT_SPLIT = "test"


def load_entries(split: str = DEFAULT_SPLIT) -> List[DatasetEntry]:
    ds = load_dataset(DATASET_NAME, split=split)
    ranking = ds.filter(lambda x: x["task"] == "ranking")
    matching = ds.filter(lambda x: x["task"] == "matching")

    def to_entry(x) -> DatasetEntry:
        return DatasetEntry(
            images=x["images"],
            contest_number=x["contest_number"],
            problem=x["problem"],
            answer=x["answer"],
            task=x["task"],
        )

    return [to_entry(x) for x in ranking] + [to_entry(x) for x in matching]
