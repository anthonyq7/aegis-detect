import json
import os

from aegis.normalize import strip_docstrings

OUT_PATH = "data/processed"
os.makedirs(OUT_PATH, exist_ok=True)

TEST = "test2.jsonl"
VAL = "validate2.jsonl"
TRAIN = "train2.jsonl"

TEST_OUT = "test.jsonl"
VAL_OUT = "val.jsonl"
TRAIN_OUT = "train.jsonl"

def clean() -> None:

    update_samples(in_path=f"{OUT_PATH}/{TEST}", out_path=f"{OUT_PATH}/{TEST_OUT}")
    update_samples(in_path=f"{OUT_PATH}/{VAL}", out_path=f"{OUT_PATH}/{VAL_OUT}")
    update_samples(in_path=f"{OUT_PATH}/{TRAIN}", out_path=f"{OUT_PATH}/{TRAIN_OUT}")


def code_to_ast_sequence(code: str) -> str | None:
    """Standardize formatting and remove comments + docstrings. None if unparsable."""
    return strip_docstrings(code)

def update_samples(in_path: str, out_path: str) -> None:
    kept = dropped = 0
    with open(in_path, "r") as infile, open(out_path, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            cleaned = code_to_ast_sequence(data.get("code", ""))
            if not cleaned:
                dropped += 1  # unparsable (e.g. markdown fences); never pass raw code through
                continue
            outfile.write(json.dumps({"code": cleaned, "label": data.get("label")}) + "\n")
            kept += 1
    print(f"{out_path}: kept {kept}, dropped {dropped} unparsable")

if __name__ == "__main__":
    clean()

