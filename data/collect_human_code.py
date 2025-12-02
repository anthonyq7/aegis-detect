"""Script for collecting and cleaning data. Saves function signatures to build prompts for OpenAI calls"""

import ast
import json
import os
from typing import List, Optional

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
HUGGING_FACE_TOKEN=os.getenv("HUGGING_FACE_TOKEN", "")
login(token=HUGGING_FACE_TOKEN)

PATH = "data/raw"
RAW_PATH = "data/raw/raw_human.jsonl"
CLEAN_PATH = "data/raw/cleaned.jsonl"
PROMPTS_PATH = "data/raw/clean_prompts.jsonl"
RAW_N_TARGET = 200000
N_TARGET = 50000
SEED=22
os.makedirs(PATH, exist_ok=True)

def collect_human_code() -> None:
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        streaming=True,
        data_dir="data/python",
        split="train"
    )
    collected = 0

    try:
        with open(RAW_PATH, "w") as file:
            for i, sample in enumerate(ds):

                if collected >= RAW_N_TARGET:
                    break

                if collected % 100 == 0:
                    file.flush()
                    print(f"Retrieved {i} samples so far...")
                    print(f"Collected {collected} samples...")

                filename = sample.get("path", "")
                code = sample.get("content", "")
                num_lines = len(code.splitlines())

                if not code:
                    continue

                if "test" in filename or "spec" in filename:
                    continue

                if 20 <= num_lines <= 100:
                    sample_metadata = {"code": code, "label": 0}
                    file.write(json.dumps(sample_metadata) + "\n")
                    collected += 1

    except Exception as e:
        raise Exception(f"Failed to collect human samples: {e}")

def is_standalone_func(node) -> bool:

    if node.args.args:
        first_arg = node.args.args[0].arg
        if first_arg in ["self", "cls"]:
            return False

    #reject magic methods
    if node.name.startswith("__") and node.name.endswith("__"):
        return False

    return True

def is_english(code: str) -> bool:
    try:
        code.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

#removes comments + standardizes format aggressively
def clean(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)

        has_function = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return None

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not is_standalone_func(node):
                    return None
                has_function = True

        if not has_function:
            return None

        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return None

#gets function signatures
def extract_info(code: str) -> List[str] | None:
    try:
        signatures = []
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):

                node.decorator_list = []
                node.body = [ast.Expr(value=ast.Constant(value=...))]

                stub = ast.unparse(node)
                signature_line = stub.split('\n')[0]
                signatures.append(signature_line)

        return signatures
    except Exception:
        pass

    return None

#cleans and processes raw data + saves
def process(input_file: str = RAW_PATH, clean_output: str = CLEAN_PATH, prompts_out: str = PROMPTS_PATH) -> None:
    clean_samples = 0
    with open(input_file, "r") as file_in, open(clean_output, "w") as cleaned_human, open(prompts_out, "w") as cleaned_prompts:

        for data in file_in:

            if clean_samples == N_TARGET:
                break

            if clean_samples % 100 == 0:
                print(f"Processed {clean_samples} so far...")

            raw_sample = json.loads(data)
            sample = raw_sample.get("code", "")

            if not sample:
                    continue

            cleaned = clean(sample)
            signatures = extract_info(sample)

            if cleaned and signatures:
                metadata = {"signatures": signatures, "code": cleaned, "label": 0}
                cleaned_prompts.write(json.dumps(metadata) + "\n")

                dataset_metadata = {"code": cleaned, "label": 0}
                cleaned_human.write(json.dumps(dataset_metadata) + "\n")

                clean_samples += 1


def main() -> None:
    collect_human_code()
    print("Finished collecting human samples")
    process()
    print("Finished cleaning and processing human samples")

    return

if __name__ == "__main__":
    main()


