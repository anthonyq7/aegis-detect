
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from aegis.normalize import strip_docstrings

HUMAN_CODE = "data/raw/human_dataset.jsonl"
AI_CODE = "data/raw/ai_dataset.jsonl"
#generation logs: user_content is the prompt, which ties an AI sample to the human sample it was built from
AI_CLEAN_LOG = "data/raw/ai_clean_code.jsonl"
AI_MESSY_LOG = "data/raw/ai_messy_code.jsonl"
CLEAN_PROMPTS = "data/raw/clean_code_prompts.jsonl"
OUT_PATH = "data/processed"
SEED = 22

#row order of both raw files: 4000 clean, 4000 messy, 2000 comp
SOURCES = ["clean"] * 4000 + ["messy"] * 4000 + ["comp"] * 2000
MESSY_PREFIX = "\n\nCODE TO REFACTOR:\n"

os.makedirs(OUT_PATH, exist_ok=True)

def squash(text: str) -> str:
    return "".join(text.split())

#the fields collect_ai_code put in the clean prompt, compared whitespace-insensitively
def clean_prompt_key(docstring: str, signature) -> str:
    text = docstring.strip()
    for q in ('"""', "'''"):
        if text.startswith(q) and text.endswith(q):
            text = text[len(q):-len(q)]
            break

    block = "".join(signature) if isinstance(signature, list) else str(signature)
    return squash(f"DOCSTRING:{text}SIGNATURES:{block}")

#both classes in one frame; a human sample and the AI sample derived from it share a group
def load_linked() -> pd.DataFrame:

    human = pd.read_json(HUMAN_CODE, lines=True)
    ai = pd.read_json(AI_CODE, lines=True)
    human["source"] = ai["source"] = SOURCES
    human["group"] = [f"h{i}" for i in range(len(human))]

    clean_log = pd.read_json(AI_CLEAN_LOG, lines=True).head(4000)
    messy_log = pd.read_json(AI_MESSY_LOG, lines=True).head(4000)
    logged = pd.concat([clean_log["code"], messy_log["code"]], ignore_index=True)
    assert logged.equals(ai["code"].head(8000)), "generation logs do not line up with ai_dataset.jsonl"
    assert messy_log["user_content"].str.startswith(MESSY_PREFIX).all(), "unexpected messy prompt format"

    prompts = pd.read_json(CLEAN_PROMPTS, lines=True)
    prompt_code = {
        clean_prompt_key(docstring, signature): code
        for docstring, signature, code in zip(prompts["docstring"], prompts["signature"], prompts["code"])
        if isinstance(docstring, str)
    }

    #the human code each AI sample was generated from (KeyError = a prompt that cannot be traced)
    origin = [prompt_code[squash(prompt)] for prompt in clean_log["user_content"]]
    origin += [prompt.removeprefix(MESSY_PREFIX).removesuffix("\n\n") for prompt in messy_log["user_content"]]
    origin += [None] * 2000 #comp prompts were never logged

    human_group = dict(zip(zip(human["source"], human["code"]), human["group"]))
    ai["group"] = [
        human_group.get((source, code), f"a{i}")
        for i, (source, code) in enumerate(zip(ai["source"], origin))
    ]

    print(f"AI samples grouped with their human source: {ai['group'].str.startswith('h').sum()}")
    return pd.concat([human, ai], ignore_index=True)

def run_preprocess() -> None:

    df = load_linked()
    print(f"Combined df size: {len(df)}")

    #same normalization standardize_code.py applies; rows it would drop are dropped here, before the split
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        df["norm"] = df["code"].map(strip_docstrings)
    df = df[df["norm"].astype(bool)].reset_index(drop=True)

    #identical code must not sit in two splits: merge those groups, then drop the repeats
    #(code that appears under both labels is dropped entirely)
    for _, groups in df[df.duplicated("norm", keep=False)].groupby("norm")["group"]:
        df.loc[df["group"].isin(groups), "group"] = groups.iloc[0]
    both_labels = df.groupby("norm")["label"].transform("nunique") > 1
    df = df[~both_labels].drop_duplicates("norm").reset_index(drop=True)
    print(f"After dropping unparsable and duplicate samples: {len(df)}")

    #80/10/10 by group, balanced over source x label
    strata = df["source"] + "_" + df["label"].astype(str)
    fold = np.zeros(len(df), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    for k, (_, idx) in enumerate(splitter.split(df, strata, df["group"])):
        fold[idx] = k
    df["split"] = np.select([fold == 0, fold == 1], ["test2", "validate2"], "train2")

    assert df.groupby("group")["split"].nunique().max() == 1, "a group is in more than one split"
    assert df.groupby("norm")["split"].nunique().max() == 1, "identical code is in more than one split"

    for split, part in df.groupby("split"):
        part = part.sample(frac=1, random_state=SEED)
        part[["code", "label", "source", "group"]].to_json(f"{OUT_PATH}/{split}.jsonl", orient="records", lines=True)
        print(f"{split}: {len(part)} samples - {(len(part)/len(df))*100:.1f}%")


def check_balance() -> None:
    groups = {}
    for split in ["train2", "validate2", "test2"]:
        df = pd.read_json(f"{OUT_PATH}/{split}.jsonl", lines=True)
        human = sum(df["label"] == 0)
        llm = sum(df["label"] == 1)
        total = len(df)
        groups[split] = set(df["group"])

        print(f"\n{split.upper()}:")
        print(f"  Total: {total}")
        print(f"  Human: {human} ({human/total*100:.1f}%)")
        print(f"  LLM: {llm} ({llm/total*100:.1f}%)")
        print(df.groupby(["source", "label"]).size().unstack().to_string())

    leaked = (
        (groups["train2"] & groups["validate2"])
        | (groups["train2"] & groups["test2"])
        | (groups["validate2"] & groups["test2"])
    )
    assert not leaked, f"{len(leaked)} groups appear in more than one split"
    print("\nNo group appears in more than one split.")

if __name__ == "__main__":
    run_preprocess()
    check_balance()

