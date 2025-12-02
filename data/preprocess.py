
import os

import pandas as pd
from sklearn.model_selection import train_test_split

HUMAN_CODE = "data/raw/cleaned.jsonl"
AI_CODE = "data/raw/ai_dataset.jsonl"
OUT_PATH = "data/processed"

os.makedirs(OUT_PATH, exist_ok=True)

def run_preprocess() -> None:

    human_code_df = pd.read_json(HUMAN_CODE, lines=True)
    llm_code_df = pd.read_json(AI_CODE, lines=True)

    combined_df = pd.concat([human_code_df, llm_code_df])
    print(f"Combined df size: {len(combined_df)}")

    #shuffle
    combined_df = combined_df.sample(frac=1, random_state=22).reset_index(drop=True)

    train, temp = train_test_split(combined_df, test_size=0.2, random_state=22, stratify=combined_df["label"])
    test, validate = train_test_split(temp, test_size=0.5, random_state=22, stratify=temp["label"])

    train.to_json(f"{OUT_PATH}/train2.jsonl", orient="records", lines=True)
    test.to_json(f"{OUT_PATH}/test2.jsonl", orient="records", lines=True)
    validate.to_json(f"{OUT_PATH}/validate2.jsonl", orient="records", lines=True)


    print(f"Train: {len(train)} samples - {(len(train)/len(combined_df))*100:.1f}%")
    print(f"Test: {len(test)} samples - {(len(test)/len(combined_df))*100:.1f}%")
    print(f"Validate: {len(validate)} samples - {(len(validate)/len(combined_df))*100:.1f}%")


def check_balance() -> None:
    for split in ["train2", "validate2", "test2"]:
        df = pd.read_json(f"{OUT_PATH}/{split}.jsonl", lines=True)
        human = sum(df["label"] == 0)
        llm = sum(df["label"] == 1)
        total = len(df)

        print(f"\n{split.upper()}:")
        print(f"  Total: {total}")
        print(f"  Human: {human} ({human/total*100:.1f}%)")
        print(f"  LLM: {llm} ({llm/total*100:.1f}%)")

if __name__ == "__main__":
    run_preprocess()
    check_balance()

