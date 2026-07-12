"""CLI for Aegis"""

import argparse
import json
from argparse import RawDescriptionHelpFormatter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

from transformers import logging as hf_logging

from aegis.predictor import Predictor

hf_logging.set_verbosity_error()

try:
    __version__ = version("aegis-detect")
except PackageNotFoundError:
    __version__ = "unknown"

def get_code_input(text: Optional[str], file_path: Optional[str]) -> str:
    if text is not None:
        if not text.strip():
            raise SystemExit("Error: --text cannot be empty")
        return text

    file = Path(file_path).expanduser()
    if not file.exists():
        raise SystemExit(f"Error: File not found: {file_path}")
    return file.read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Classify Python code as AI or human",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
        "Examples:\n"
        "  aegis --file path/to/code.py\n"
        "  aegis --text 'def add(a, b): return a + b'\n"
        "  aegis --file script.py --json > result.json\n"
        "  aegis --text 'your code snippet' --threshold 0.7\n"
        )
    )

    #creates a mutually exclusive input group where only either --text or --file is allowed
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Code snippet as text (quote if it contains spaces or special characters).")
    input_group.add_argument("--file", type=str, help="Path to code file")

    #output format
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    #allows users to adjust a threshold
    parser.add_argument("--threshold", type=float, help="Custom threshold to classify as AI")

    #allows users to check version
    parser.add_argument("--version", action="version", version=f"aegis {__version__}", help="Display the version of Aegis")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    #parses args
    args = build_parser().parse_args(argv)

    #gets the code to analyze
    code = get_code_input(args.text, args.file)

    #loads model and makes a prediction
    try:
        predictor = Predictor(threshold=args.threshold)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")

    result = predictor.predict(code)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Prediction: {result['prediction']} | human={result['human']} | ai={result['ai']}")


if __name__ == "__main__":
    main()
