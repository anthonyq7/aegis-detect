import json
from textwrap import dedent
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
CLEAN_FILE_PATH = "data/raw/clean_prompts.jsonl"
CLEAN_OUT_PATH = "data/raw/clean_code_prompts.jsonl"
MAX_OUTPUT_TOKENS = 4096
MODEL = "gpt-5-mini"
DOCSTRING_TARGET = 15000
MAX_CONCURRENT = 200

client = AsyncOpenAI()

SYSTEM_PROMPT = dedent("""
You are an Expert Senior Software Engineer.
Your task is to read the provided Python function and write a comprehensive "Implementation Spec" in the form of a docstring.
This docstring will be used to instruct a Junior Developer to reconstruct this exact function.

### 1. FOREIGN LANGUAGE & QUALITY CHECK
Analyze the variable names, function name, and comments.
- If the code uses **Non-English** variable names (e.g., 'compteur', 'calcular', '你好'), return EXACTLY the string: "SKIP".
- If the code is just data (a list of numbers), boilerplate, or incoherent, return EXACTLY the string: "SKIP".
- If you return "SKIP", output **nothing else**.

### 2. DOCSTRING GENERATION RULES
Write a docstring that describes **WHAT** the function does and **HOW** it is structured logically.

**CRITICAL: Preserve Structural Integrity**
- **Do not** just summarize the output (e.g., "Returns sorted list").
- **DO** describe the algorithm and control flow used (e.g., "Implements a bubble sort by iterating through the list with nested loops, comparing adjacent elements and swapping them").
- **DO** mention specific data structures used (e.g., "Uses a stack to track opening brackets").
- **DO** specify the inferred input types and return type (e.g., "Accepts a dictionary mapping strings to integers").
- **DO NOT** write literal code snippets or give away the solution (e.g., do NOT write `for i in range(len(arr))`). Use natural language.

**The Goal:** The Junior Developer should be able to write code that looks **structurally identical** to the original (same loops, same complexity) based solely on your description.

### 3. OUTPUT FORMAT
- Return **ONLY** the docstring text.
- Do not include "Here is the docstring" or any conversational filler.
- Do not use markdown code blocks (```).
""")

async def generate_docstring(code: str, semaphore: asyncio.Semaphore) -> str | None:

    prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": code
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
            
            print(response.incomplete_details)
            print(response.status)
            return None
        except Exception as e:
            print(f"API Error: {e}")
            return None

async def create_docstrings() -> None:
    try:
        if not os.path.exists(CLEAN_FILE_PATH):
            print(f"Input file not found: {CLEAN_FILE_PATH}")
            return
        
        with open(CLEAN_FILE_PATH, "r") as f:
            candidates = f.readlines()
        
        print(f"Loaded {len(candidates)} candidates. Starting docstring generation...")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        count = 0

        async def wrap_task(original_data):
            code = original_data.get("code", "")
            if code:
                docstring = await generate_docstring(code, semaphore)
                return original_data, docstring
                
            return original_data, None
        
        wrapped_tasks = []
        for line in candidates:
            try:
                data = json.loads(line)
                wrapped_tasks.append(wrap_task(data))
            except:
                continue

        buffer = []
        with open(CLEAN_OUT_PATH, "w") as outfile:
            for task in asyncio.as_completed(wrapped_tasks):

                if count >= DOCSTRING_TARGET:
                    break
                
                original_data, docstring = await task
                if docstring and docstring.strip().upper() != "SKIP":
                    code = original_data.get("code")
                    sig = original_data.get("signatures")
                    buffer.append({"code": code, "signatures": sig, "docstring": docstring})

                if len(buffer) >= 10:
                    for doc in buffer:
                        outfile.write(json.dumps(doc) + "\n")
                    count += len(buffer)
                    outfile.flush()
                    print(f"Saved {count} docstrings to disk...")
                    buffer.clear()
            
            if buffer:
                for doc in buffer:
                    outfile.write(json.dumps(doc) + "\n")
                
                count += len(buffer)
                outfile.flush()
                buffer.clear()
            
        print("Finished collecting docstrings")
    except:
        print("Failed to generate docstrings")

if __name__ == "__main__":
    asyncio.run(create_docstrings())