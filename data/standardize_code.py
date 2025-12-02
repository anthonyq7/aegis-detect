import ast, os, json

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


def code_to_ast_sequence(code: str) -> str:
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except:
        return code

def update_samples(in_path: str, out_path: str) -> None:
    with open(in_path, "r") as infile, open(out_path, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            cleaned = code_to_ast_sequence(data.get("code", ""))
            outfile.write(json.dumps({"code": cleaned, "label": data.get("label")}) + "\n")

if __name__ == "__main__":
    clean()

