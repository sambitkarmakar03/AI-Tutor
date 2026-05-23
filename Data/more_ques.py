import json
import random
import re
import time
from datetime import datetime, timedelta
import ollama
from tqdm import tqdm

random.seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID         = "llama3.2:3b"
TARGET           = 1850
BATCH_SIZE       = 35  # REDUCED: Keeps the small LLM focused and formatting clean
INPUT_PATH       = "/Users/sambit_03/Desktop/RYKYU/Data/training_data_reclassified.json"
OUTPUT_PATH      = "/Users/sambit_03/Desktop/RYKYU/Data/training_data_topped_up.json"
PARTIAL_PATH     = "/Users/sambit_03/Desktop/RYKYU/Data/training_data_topup_partial.json"

# ── Load existing reclassified data ───────────────────────────────────────────
with open(INPUT_PATH) as f:
    data = json.load(f)

# ── Regex validators (Properly Escaped) ───────────────────────────────────────
RE_LEVEL_1 = re.compile(r"^\s*\d+\s*[\+\-]\s*\d+\s*$")
RE_LEVEL_2 = re.compile(r"^\s*\d+\s*[\*×xX]\s*\d+\s*$")
RE_LEVEL_3 = re.compile(r"^\s*\d+\s*[/÷]\s*\d+\s*$")
RE_LEVEL_4 = re.compile(r"^\s*\d+\s*[\+\-\*×/÷xX]\s*\d+\s*$")

LEVEL_REGEX = {
    "level_2": RE_LEVEL_2,
    "level_3": RE_LEVEL_3,
    "level_4": RE_LEVEL_4,
}

# ── Answer calculator ──────────────────────────────────────────────────────────
def calculate_answer(question: str) -> str | None:
    try:
        expr = question.replace("÷", "/").replace("×", "*").replace("x", "*").replace("X", "*")
        result = eval(expr, {"__builtins__": {}})
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        elif isinstance(result, int):
            return str(result)
        return None
    except Exception:
        return None

# ── Bulletproof Extractor ──────────────────────────────────────────────────────
def extract_questions(text: str) -> list[str]:
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        json_str = re.sub(r',\s*\]', ']', json_str) 
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            try:
                parsed = json.loads(json_str.replace("'", '"'))
                if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                    return [str(q).strip() for q in parsed if str(q).strip()]
            except json.JSONDecodeError:
                pass

    pattern = r'\d+\s*[\+\-\*/xX×÷]\s*\d+(?:\s*[\+\-\*/xX×÷]\s*\d+)*'
    matches = re.findall(pattern, text)
    
    seen = set()
    result = []
    for match in matches:
        clean_match = match.replace('x', '*').replace('X', '*').replace('×', '*').replace('÷', '/')
        clean_match = " ".join(clean_match.split())
        if clean_match not in seen:
            seen.add(clean_match)
            result.append(clean_match)
            
    return result

# ── Level descriptions for prompts ────────────────────────────────────────────
LEVEL_DESCRIPTIONS = {
    "level_2": {
        "name": "Multiplication",
        "description": "Multiplication problems for 2nd–3rd grade. Mix of times-tables (1–10), 2-digit × 1-digit, and 2-digit × 2-digit. Use * only.",
        "example": '["3 * 7", "14 * 5", "23 * 4", "12 * 13", "9 * 8"]',
    },
    "level_3": {
        "name": "Division",
        "description": "Division problems for 2nd–3rd grade. Mix of 2-digit ÷ 1-digit, and 3-digit ÷ 1-digit. ALL divisions must be exact — no remainders. Use / only.",
        "example": '["36 / 6", "48 / 4", "72 / 8", "144 / 6", "125 / 5"]',
    },
    "level_4": {
        "name": "Mixed Operations",
        "description": "Mixed arithmetic for 2nd–3rd grade. Use +, -, *, /. Each question uses exactly ONE operator. Mix of single, 2-digit, and 3-digit numbers. No negative answers. Exact division only.",
        "example": '["23 + 47", "87 - 34", "6 * 8", "48 / 4", "124 + 357"]',
    },
}

# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(level_key: str, batch_index: int, batch_size: int, seen: set) -> str:
    lvl = LEVEL_DESCRIPTIONS[level_key]
    existing = data["levels"][level_key]["questions"]
    samples  = random.sample(existing, min(5, len(existing)))
    examples = ", ".join(f'"{q["question"]}"' for q in samples)

    # DYNAMIC INJECTION: Force the LLM to use different numbers every batch to prevent duplicate loops
    focus_digit = random.randint(2, 15)
    focus_large = random.choice([random.randint(20, 50), random.randint(100, 250), random.randint(300, 500)])

    return f"""You are a math question generator for a children's educational AI training dataset.

Level: {lvl["name"]}
Description: {lvl["description"]}

Style examples (do NOT repeat these):
{examples}

Generate exactly {batch_size} NEW math questions for this level.
- BATCH FOCUS: To ensure dataset variety, include numbers related to {focus_digit} and numbers around {focus_large}.
- Each question: a single arithmetic expression string using exactly ONE operator.
- Use / for division, * for multiplication, + for addition, - for subtraction.
- No negative answers. No remainders for division. Integer answers only.
- Return ONLY a valid JSON array of strings — no explanation, no markdown.

Example output:
{lvl["example"]}
"""

# ── Ollama caller ──────────────────────────────────────────────────────────────
def fetch_batch(level_key: str, batch_index: int, batch_size: int,
                seen: set, pbar: tqdm) -> list[str]:
    prompt = build_prompt(level_key, batch_index, batch_size, seen)

    for attempt in range(3):
        try:
            pbar.set_postfix_str(f"Generating... (attempt {attempt+1}/3)", refresh=True)
            response = ollama.chat(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.95, # INCREASED: Forces more variety in numbers
                    "top_p": 0.95, 
                    "num_predict": 2048
                },
            )
            raw = response["message"]["content"].strip()

            questions = extract_questions(raw)
            if questions:
                return questions[:batch_size]
            else:
                if attempt == 2 and pbar:
                    pass # Silenced the heavy debug print to keep console clean
                raise ValueError("No math equations found in response text.")

        except Exception as e:
            time.sleep(1)

    return []

# ── Top-up generator ───────────────────────────────────────────────────────────
def topup_level(level_key: str) -> list[dict]:
    existing   = data["levels"][level_key]["questions"]
    have       = len(existing)
    need       = TARGET - have
    level_name = LEVEL_DESCRIPTIONS[level_key]["name"]
    validator  = LEVEL_REGEX[level_key]

    if need <= 0:
        print(f"\n  ✅ {level_name} already has {have} questions — no top-up needed.")
        return existing

    print(f"\n{'═'*60}")
    print(f"  Level   : {level_name}")
    print(f"  Have    : {have}  |  Need: {need} more  |  Target: {TARGET}")
    print(f"  Batches : ~{-(-need // BATCH_SIZE)}")
    print(f"  Started : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'═'*60}")

    seen              = {q["question"].strip() for q in existing}
    new_qa            = []
    batch_index       = 0
    dupes_skipped     = 0
    invalid_skipped   = 0
    regex_skipped     = 0
    start_time        = time.time()

    with tqdm(
        total=need,
        desc=f"  {level_name[:20]:<20} batch   1",
        unit="q",
        colour="green",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    ) as pbar:
        while len(new_qa) < need:
            current_batch = min(BATCH_SIZE, (need - len(new_qa)) + 10)
            batch_start   = time.time()

            pbar.set_description(f"  {level_name[:20]:<20} batch {batch_index+1:>3}")
            questions = fetch_batch(level_key, batch_index, current_batch, seen, pbar)

            added = 0
            for q in questions:
                if q in seen:
                    dupes_skipped += 1
                    continue
                if not validator.match(q):
                    regex_skipped += 1
                    continue
                answer = calculate_answer(q)
                if answer is None:
                    invalid_skipped += 1
                    continue
                seen.add(q)
                new_qa.append({"question": q, "answer": answer})
                added += 1
                pbar.update(1)
                if len(new_qa) >= need:
                    break

            pbar.set_postfix({
                "added"  : added,
                "dupes"  : dupes_skipped,
                "invalid": invalid_skipped,
                "regex✗" : regex_skipped,
                "⏱"      : f"{time.time()-batch_start:.1f}s",
            }, refresh=True)

            batch_index += 1

    elapsed = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"\n  ✅ Generated {len(new_qa)} new questions in {elapsed}")
    print(f"     Dupes skipped   : {dupes_skipped}")
    print(f"     Regex rejected  : {regex_skipped}")
    print(f"     Invalid answers : {invalid_skipped}")

    return existing + new_qa

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    try:
        models    = ollama.list()
        available = [m["model"] for m in models.get("models", [])]
        if not any(MODEL_ID in m for m in available):
            print(f"\n❌ Model '{MODEL_ID}' not found. Run: ollama pull {MODEL_ID}")
            return
        print(f"\n✓ Model '{MODEL_ID}' is ready.")
    except Exception as e:
        print(f"\n❌ Cannot connect to Ollama: {e}")
        print("   Run: ollama serve")
        return

    print(f"\n{'█'*60}")
    print(f"  Top-Up Generator  —  Target: {TARGET} per level")
    print(f"{'─'*60}")
    for lvl_key, lvl_data in data["levels"].items():
        have = len(lvl_data["questions"])
        gap  = max(0, TARGET - have)
        name = lvl_data["name"]
        print(f"  {name:<30} have {have:>4}  →  need {gap:>4} more")
    print(f"{'█'*60}\n")

    output        = {"levels": {}}
    overall_start = time.time()

    output["levels"]["level_1"] = data["levels"]["level_1"]

    for level_key in ["level_2", "level_3", "level_4"]:
        output["levels"][level_key] = {
            "name"     : data["levels"][level_key]["name"],
            "questions": topup_level(level_key),
        }
        with open(PARTIAL_PATH, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = str(timedelta(seconds=int(time.time() - overall_start)))

    print(f"\n{'█'*60}")
    print(f"  ✅ ALL DONE — training_data_topped_up.json")
    print(f"  Total time : {elapsed}")
    print(f"{'─'*60}")
    total = 0
    for lvl_key, lvl_data in output["levels"].items():
        count  = len(lvl_data["questions"])
        total += count
        status = "✅" if count >= TARGET else "⚠️ "
        print(f"  {status} {lvl_data['name']:<28} {count:>5} Q&A pairs")
    print(f"{'─'*60}")
    print(f"  {'TOTAL':<30} {total:>5} Q&A pairs")
    print(f"{'█'*60}\n")

if __name__ == "__main__":
    main()