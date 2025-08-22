import argparse
from data import load_entries
from models import load_model, chat_infer


def main():
    p = argparse.ArgumentParser(description="humor-eval CLI")
    p.add_argument("--split", default="test")
    p.add_argument("--index", type=int, default=0, help="dataset entry index")
    p.add_argument("--max_new_tokens", type=int, default=512)
    args = p.parse_args()

    entries = load_entries(args.split)
    entry = entries[args.index]

    processor, model = load_model()
    resp = chat_infer(processor, model, entry["images"], entry["problem"], max_new_tokens=args.max_new_tokens)  # type: ignore[index]

    print("Question:", entry["problem"])  # type: ignore[index]
    print("Response:", resp)


if __name__ == "__main__":
    main()
