import json
import re
import time
from datetime import datetime, timedelta
import ollama
from tqdm import tqdm

# ── Model ──────────────────────────────────────────────────────────────────────

MODEL_ID = "llama3.2:3b"

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_PER_LEVEL = 1250   # 4 levels × 1250 = 5000 total
BATCH_SIZE       = 50     # questions per call → 25 batches per level

# ── Load seed data ─────────────────────────────────────────────────────────────
with open("/Users/sambit_03/Desktop/RYKYU/Data/feed_data.json") as f:
    seed_data = json.load(f)

# ── Level definitions ──────────────────────────────────────────────────────────
LEVELS = {
    "level_1": {
        "name": "Addition and Subtraction",
        "description": (
            "Addition and subtraction problems for 2nd–3rd grade students. "
            "Start with single-digit numbers, then gradually introduce 2-digit numbers "
            "(e.g. 23 + 45, 87 - 34), carrying/borrowing, and finally 3-digit numbers "
            "(e.g. 124 + 357, 500 - 278). No negative answers."
        ),
    },
    "level_2": {
        "name": "Multiplication",
        "description": (
            "Multiplication problems for 2nd–3rd grade students. "
            "Begin with times-tables (1–10), then progress to multiplying a 2-digit number "
            "by a 1-digit number (e.g. 23 * 4, 56 * 7), and finally 2-digit x 2-digit "
            "(e.g. 12 * 13, 24 * 15). Increasing difficulty throughout."
        ),
    },
    "level_3": {
        "name": "Division",
        "description": (
            "Division problems for 2nd–3rd grade students. "
            "Start with exact division using times-table facts (e.g. 36 / 6), "
            "then introduce dividing 2-digit numbers by 1-digit numbers (e.g. 48 / 4, 72 / 8), "
            "and finally 3-digit numbers by 1-digit numbers (e.g. 144 / 6). "
            "All divisions must be exact (no remainders)."
        ),
    },
    "level_4": {
        "name": "Mixed Operations",
        "description": (
            "Mixed arithmetic problems for 2nd–3rd grade students combining addition, subtraction, "
            "multiplication, and division. "
            "Include problems across a wide difficulty range: single-digit operations, "
            "2-digit operations, and some 3-digit operations. "
            "Use operators: +, -, *, /. No negative answers. Exact division only (no remainders)."
        ),
    },
}

SEED_EXAMPLES = {
    lvl: data["questions"] for lvl, data in seed_data["levels"].items()
    if lvl in LEVELS
}

# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(level_key: str, batch_index: int, batch_size: int = BATCH_SIZE) -> str:
    level = LEVELS[level_key]
    examples = SEED_EXAMPLES.get(level_key, [])
    example_str = "\n".join(f'  "{q}"' for q in examples[:10])

    return f"""You are a math question generator for a children's educational AI training dataset.

Level: {level["name"]}
Description: {level["description"]}

Seed examples (style reference only — do NOT repeat these):
{example_str}

Generate exactly {batch_size} NEW math questions for this level.
- Batch {batch_index + 1}: questions should be progressively harder than previous batches.
- Each question must be a single arithmetic expression string (e.g. "23 + 47", "6 * 8", "144 / 12").
- Use / for division, * for multiplication, + for addition, - for subtraction.
- No negative answers. No remainders for division.
- Vary the numbers — avoid repeating the same question twice across the dataset.
- Return ONLY a valid JSON array of strings, no explanation, no markdown fences.

Example output format:
["12 + 34", "56 - 23", "7 * 8", "48 / 6"]
"""

# ── Answer calculator ──────────────────────────────────────────────────────────
def calculate_answer(question: str) -> str | None:
    """Compute the answer locally — guarantees correctness regardless of LLM output."""
    try:
        expr = question.replace("÷", "/").replace("×", "*").replace("x", "*")
        result = eval(expr, {"__builtins__": {}})  # noqa: S307 — controlled input
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 4))
    except Exception:
        return None

# ── Bulletproof Extractor ──────────────────────────────────────────────────────
def extract_questions(text: str) -> list[str]:
    """
    Attempts to extract questions by first parsing cleaned JSON. 
    If that fails, it uses a relaxed regex fallback to scrape math expressions.
    """
    # 1. Try to find and clean a JSON array
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        json_str = re.sub(r',\s*\]', ']', json_str) # Fix trailing commas
        
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                return parsed
        except json.JSONDecodeError:
            try:
                # Fix single quotes
                json_str_double = json_str.replace("'", '"')
                parsed = json.loads(json_str_double)
                if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                    return parsed
            except json.JSONDecodeError:
                pass # Proceed to regex fallback

    # 2. Aggressive Regex Fallback
    # Matches numbers followed by any math symbol (+, -, *, /, x, X, ÷) and another number.
    # It ignores equations that have equal signs by only extracting the left side.
    pattern = r'\d+\s*[\+\-\*/xX÷]\s*\d+(?:\s*[\+\-\*/xX÷]\s*\d+)*'
    matches = re.findall(pattern, text)
    
    seen = set()
    result = []
    for match in matches:
        # Standardize operators and spacing
        clean_match = match.replace('x', '*').replace('X', '*').replace('÷', '/')
        clean_match = " ".join(clean_match.split())
        
        if clean_match not in seen:
            seen.add(clean_match)
            result.append(clean_match)
            
    return result

def fetch_batch(
    level_key: str,
    batch_index: int,
    batch_size: int = BATCH_SIZE,
    pbar: tqdm = None,
) -> list[str]:
    prompt = build_prompt(level_key, batch_index, batch_size)

    for attempt in range(3):
        try:
            if pbar:
                pbar.set_postfix_str(f"Generating... (attempt {attempt+1}/3)", refresh=True)

            response = ollama.chat(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.7, # Lowered slightly for more predictable formatting
                    "top_p": 0.90,
                    "num_predict": 2048,
                },
            )
            raw = response.get("message", {}).get("content", "").strip()

            questions = extract_questions(raw)
            
            if questions:
                return questions[:batch_size]
            else:
                # If it completely fails, print exactly what the model said so we can diagnose it
                if attempt == 2 and pbar:
                    pbar.write(f"\n[DEBUG] LLM output contained no math. Raw output:\n{raw[:500]}...\n")
                raise ValueError("No math equations found in response text.")

        except Exception as e:
            if pbar:
                pbar.write(f"  Error attempt {attempt+1}: {e} — retrying...")
            time.sleep(1)

    if pbar:
        pbar.write(f"  Batch {batch_index} failed after 3 attempts — skipping.")
    return []

# ── Level generator ────────────────────────────────────────────────────────────
def generate_level(
    level_key: str,
    target: int = TARGET_PER_LEVEL,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    level_name = LEVELS[level_key]["name"]

    print(f"\n{'='*60}")
    print(f"  Level  : {level_name}")
    print(f"  Target : {target:,} questions  |  Batch size: {batch_size}")
    print(f"  Batches: ~{-(-target // batch_size)}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    all_qa             = []
    seen_questions     = set()
    batch_index        = 0
    duplicates_skipped = 0
    invalid_skipped    = 0
    start_time         = time.time()

    with tqdm(
        total=target,
        desc=f"  {level_name[:20]:<20} batch   1",
        unit="q",
        colour="green",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    ) as pbar:
        while len(all_qa) < target:
            current_batch_size = min(batch_size, (target - len(all_qa)) + 10)
            batch_start = time.time()

            pbar.set_description(f"  {level_name[:20]:<20} batch {batch_index+1:>3}")

            questions = fetch_batch(level_key, batch_index, current_batch_size, pbar)

            added_this_batch = 0
            for q in questions:
                if q in seen_questions:
                    duplicates_skipped += 1
                    continue
                answer = calculate_answer(q)
                if answer is None:
                    invalid_skipped += 1
                    continue
                seen_questions.add(q)
                all_qa.append({"question": q, "answer": answer})
                added_this_batch += 1
                pbar.update(1)
                if len(all_qa) >= target:
                    break

            batch_elapsed = time.time() - batch_start

            pbar.set_postfix({
                "batch"  : batch_index + 1,
                "added"  : added_this_batch,
                "dupes"  : duplicates_skipped,
                "invalid": invalid_skipped,
                "time"   : f"{batch_elapsed:.1f}s",
            }, refresh=True)

            batch_index += 1

    total_elapsed = time.time() - start_time
    elapsed_str   = str(timedelta(seconds=int(total_elapsed)))

    print(f"\n  Done — {len(all_qa):,} questions in {elapsed_str}")
    print(f"     Batches used    : {batch_index}")
    print(f"     Duplicates skip : {duplicates_skipped}")
    print(f"     Invalid skip    : {invalid_skipped}")
    print(f"     Avg/batch       : {total_elapsed / max(batch_index, 1):.1f}s")

    return all_qa[:target]

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    # Verify Ollama is running and model is available
    try:
        models = ollama.list()
        available = [m["model"] for m in models.get("models", [])]
        if not any(MODEL_ID in m for m in available):
            print(f"\n[ERROR] Model '{MODEL_ID}' not found locally.")
            print(f"   Run this first:  ollama pull {MODEL_ID}\n")
            return
        print(f"\n[OK] Model '{MODEL_ID}' is ready.")
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running:  ollama serve\n")
        return

    total_levels = len(LEVELS)
    total_target = total_levels * TARGET_PER_LEVEL

    print(f"\n{'#'*60}")
    print(f"  Math Q&A Dataset Generator  —  Local Llama via Ollama")
    print(f"  Model   : {MODEL_ID}  (running on your Mac)")
    print(f"  Levels  : {total_levels}  |  {TARGET_PER_LEVEL:,} questions each")
    print(f"  Total   : {total_target:,} Q&A pairs")
    print(f"  Delay   : none — no rate limits locally")
    print(f"  Run at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    output        = {"levels": {}}
    overall_start = time.time()

    with tqdm(
        total=total_levels,
        desc="  Overall progress",
        unit="level",
        colour="cyan",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} levels [{elapsed}]",
        dynamic_ncols=True,
        position=0,
    ) as overall_pbar:
        for i, level_key in enumerate(LEVELS):
            qa_pairs = generate_level(level_key)
            output["levels"][level_key] = {
                "name"     : LEVELS[level_key]["name"],
                "questions": qa_pairs,
            }

            # Save after every level
            with open("/Users/sambit_03/Desktop/RYKYU/Data/training_data_partial.json", "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            tqdm.write(
                f"\n  Saved -> training_data_partial.json  ({i+1}/{total_levels} levels done)\n"
            )

            overall_pbar.update(1)
            overall_pbar.set_postfix_str(
                f"Last: {LEVELS[level_key]['name']}", refresh=True
            )

    # Final save
    with open("/Users/sambit_03/Desktop/RYKYU/Data/training_data_5000.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_elapsed = str(timedelta(seconds=int(time.time() - overall_start)))

    print(f"\n{'#'*60}")
    print(f"  ALL DONE — training_data_5000.json")
    print(f"  Total time : {total_elapsed}")
    print(f"{'-'*60}")
    for level_key, data in output["levels"].items():
        print(f"  {data['name']:<30} {len(data['questions']):>5,} Q&A pairs")
    print(f"{'-'*60}")
    total_q = sum(len(d["questions"]) for d in output["levels"].values())
    print(f"  {'TOTAL':<30} {total_q:>5,} Q&A pairs")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()