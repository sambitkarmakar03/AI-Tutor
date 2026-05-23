
import json
import random

random.seed(42)

INPUT_PATH  = "/Users/sambit_03/Desktop/RYKYU/Data/training_data_topped_up.json"
OUTPUT_PATH = "/Users/sambit_03/Desktop/RYKYU/Data/training_data_alpaca.json"

# ── Level metadata ─────────────────────────────────────────────────────────────
LEVEL_META = {
    "level_1": {
        "label"      : "Level 1",
        "topic"      : "addition and subtraction",
        "difficulty" : "easy",
    },
    "level_2": {
        "label"      : "Level 2",
        "topic"      : "multiplication",
        "difficulty" : "medium",
    },
    "level_3": {
        "label"      : "Level 3",
        "topic"      : "division",
        "difficulty" : "medium",
    },
    "level_4": {
        "label"      : "Level 4",
        "topic"      : "mixed operations",
        "difficulty" : "hard",
    },
}

# ── Instruction templates — varied so the model doesn't overfit one phrasing ──
INSTRUCTION_TEMPLATES = [
    "Solve this math problem.",
    "What is the answer to this arithmetic question?",
    "Calculate the result.",
    "Help the student solve this problem.",
    "Find the answer to this math question.",
    "Solve the following arithmetic expression.",
    "What does this equal?",
    "Work out the answer.",
]

# ── System prompt the model will see during inference ─────────────────────────
SYSTEM_PROMPT = (
    "You are a friendly math tutor for children in 2nd and 3rd grade. "
    "When given a math problem, respond with only the numeric answer. "
    "Be encouraging and accurate."
)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(INPUT_PATH) as f:
    data = json.load(f)

# ── Convert to Alpaca format ───────────────────────────────────────────────────
alpaca_records = []

for level_key, level_data in data["levels"].items():
    meta = LEVEL_META[level_key]
    questions = level_data["questions"]

    for item in questions:
        q = item["question"].strip()
        a = item["answer"].strip()

        instruction = random.choice(INSTRUCTION_TEMPLATES)

        record = {
            "system"     : SYSTEM_PROMPT,
            "instruction": instruction,
            "input"      : q,
            "output"     : a,
            "level"      : meta["label"],
            "topic"      : meta["topic"],
            "difficulty" : meta["difficulty"],
        }
        alpaca_records.append(record)

# Shuffle so all levels are mixed during training
random.shuffle(alpaca_records)

# ── Save ───────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(alpaca_records, f, indent=2, ensure_ascii=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'█'*55}")
print(f"  ✅ Saved → training_data_alpaca.json")
print(f"{'─'*55}")
print(f"  Total records : {len(alpaca_records)}")
print(f"\n  Sample record:")
sample = alpaca_records[0]
for k, v in sample.items():
    print(f"    {k:<12}: {v}")
print(f"{'─'*55}")
print(f"{'█'*55}\n")