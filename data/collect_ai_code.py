import asyncio
import json
from textwrap import dedent
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
CLEAN_FILE_PATH = "data/raw/clean_code_prompts.jsonl"
CLEAN_AI_FILE_PATH = "data/raw/ai_dataset.jsonl"
EMPTY_RESPONSE_PATH = "data/raw/empty_responses.jsonl"
MAX_OUTPUT_TOKENS = 50000
MODEL = "gpt-5.1-codex-mini"
MAX_CONCURRENT = 200

client = AsyncOpenAI()

async def generate_code(user_content: str, system_prompt: str, semaphore: asyncio.Semaphore) -> str | None:

    prompt = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    async with semaphore:
        try:
            response = await client.responses.create(
                model=MODEL,
                input=prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                reasoning={"effort": "low"}
            )

            print(f"Tokens: {response.usage.total_tokens}, Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
            result = response.output_text

            if result and result.strip():
                return result

            with open(EMPTY_RESPONSE_PATH, "a") as f:
                data = {"request": user_content}
                f.write(json.dumps(data) + "\n")

            return None
        except Exception as e:
            print(f"API Error: {e}")
            return None

async def create_code(content_list: List[str], out_path: str, prompt: str, semaphore: asyncio.Semaphore | None = None) -> None:
    try:
        print(f"Loaded {len(content_list)} docstrings. Starting code generation...")

        shared_semaphore = semaphore or asyncio.Semaphore(MAX_CONCURRENT)

        async def wrap_task(original_data):
            if original_data:
                ai_code = await generate_code(
                    user_content=original_data,
                    system_prompt=prompt,
                    semaphore=shared_semaphore
                )
                return original_data, ai_code

            return original_data, None

        wrapped_tasks = []

        for content in content_list:
            if not content:
                continue

            try:
                if content:
                    wrapped_tasks.append(wrap_task(content))
            except Exception as e:
                print(f"Skipping sample: {e}")

        buffer = []
        with open(out_path, "a") as outfile:
            count = 0
            for task in asyncio.as_completed(wrapped_tasks):

                info = await task
                if not info:
                    continue

                og_data, code = info


                if code:
                    buffer.append({"code": code, "label": 1})

                if len(buffer) >= 10:
                    for doc in buffer:
                        outfile.write(json.dumps(doc) + "\n")

                    count += len(buffer)
                    outfile.flush()
                    print(f"Saved {count} snippets to disk...")
                    buffer.clear()

            if buffer:
                for doc in buffer:
                    outfile.write(json.dumps(doc) + "\n")

                count += len(buffer)
                outfile.flush()
                buffer.clear()

        print("Finished collecting snippets")

    except Exception as e:
        print(f"Failed to generate snippets: {e}")

def normalize_input(clean_input_path: str = CLEAN_FILE_PATH) -> List[str]:

    with open(clean_input_path, "r") as clean_input:

        clean_prompts = []
        for line in clean_input:
            data = json.loads(line)
            docstring = data.get("docstring")
            signatures = data.get("signatures")

            if not docstring or not signatures:
                continue

            text = docstring.strip()
            for q in ('"""', "'''"):
                if text.startswith(q) and text.endswith(q):
                    text = text[len(q):-len(q)].strip()
                    break

            if isinstance(signatures, list):
                signature_block = "\n".join(sig.strip() for sig in signatures if sig and sig.strip())
            else:
                signature_block = str(signatures).strip()

            if not signature_block:
                continue

            clean_prompt = dedent(f"""
            DOCSTRING:
            {text}

            SIGNATURES:
            {signature_block}
            """).strip()
            clean_prompts.append(clean_prompt)

        return clean_prompts

CLEAN_PROMPT = dedent("""You are a Senior Python Software Engineer.
Your task is to implement a Python function that STRICTLY follows the provided docstring specification and declared signatures.

### INSTRUCTIONS:
1. **Logic:** Implement exactly the algorithm described in the docstring. Do not improvise or optimize beyond what is asked.
2. **Style:** Write clean, professional, PEP-8 compliant code.
   - Use descriptive variable names (e.g., `user_index` instead of `i`).
   - Use type hints for arguments and return values.
   - Do not add comments unless necessary for complex logic.
3. **Format:** Return ONLY the Python code. Do not use markdown blocks. Do not include the docstring itself in the output (it will be added later).

You will receive the docstring text and the function signatures in the user message.
""").strip()

async def main() -> None:
    clean = normalize_input()

    await asyncio.gather(
        create_code(clean, CLEAN_AI_FILE_PATH, CLEAN_PROMPT)
    )


if __name__ == "__main__":
    asyncio.run(main())

