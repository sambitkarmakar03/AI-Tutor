import json
import random
import streamlit as st
import ollama

st.set_page_config(
    page_title="Math Tutor",
    page_icon="🧮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID  = "math-tutor"
DATA_PATH = "/Users/sambit_03/Desktop/RYKYU/Data/training_data_alpaca.json"
PASS_MARK = 0.8
FAIL_MARK = 0.5
QUIZ_SIZE = 10

LEVELS = {
    "level_1": {"name": "Addition & Subtraction", "emoji": "➕", "color": "#2563EB"},
    "level_2": {"name": "Multiplication",          "emoji": "✖️", "color": "#16A34A"},
    "level_3": {"name": "Division",                "emoji": "➗", "color": "#D97706"},
    "level_4": {"name": "Mixed Operations",        "emoji": "🔀", "color": "#7C3AED"},
}
LEVEL_ORDER = ["level_1", "level_2", "level_3", "level_4"]

CORRECT = ["Great job! ✅", "Correct! ✅", "Well done! ✅", "Excellent! ✅"]
WRONG   = ["Not quite.", "Incorrect.", "Almost there.", "Not right."]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #F9FAFB !important;
    color: #111827 !important;
}

[data-testid="stAppViewContainer"] > .main > div { padding-top: 2rem; }

/* Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.15s ease !important;
    border: 1px solid #E5E7EB !important;
    background: white !important;
    color: #374151 !important;
}
.stButton > button:hover {
    border-color: #D1D5DB !important;
    background: #F9FAFB !important;
}
.stButton > button[kind="primary"] {
    background: #111827 !important;
    color: white !important;
    border-color: #111827 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1F2937 !important;
    border-color: #1F2937 !important;
}

/* Input */
.stTextInput > div > div > input {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    border-radius: 8px !important;
    border: 1px solid #E5E7EB !important;
    background: white !important;
    padding: 0.8rem !important;
    color: #111827 !important;
}
.stTextInput > div > div > input:focus {
    border-color: #111827 !important;
    box-shadow: 0 0 0 2px #11182722 !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    background: white !important;
}

/* Hide chrome */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    level_map = {"Level 1":"level_1","Level 2":"level_2","Level 3":"level_3","Level 4":"level_4"}
    out = {lvl: [] for lvl in LEVEL_ORDER}
    for item in raw:
        lk = level_map.get(item.get("level",""))
        q  = item.get("input","").strip()
        a  = str(item.get("output","")).strip()
        if lk and q and a:
            out[lk].append({"question": q, "answer": a})
    return out

def ask_model(question):
    try:
        r = ollama.chat(
            model=MODEL_ID,
            messages=[
                {"role":"system","content":"You are a math tutor. Respond with only the numeric answer."},
                {"role":"user","content":f"Solve this math problem.\n{question}"},
            ],
            options={"temperature":0.1,"num_predict":16},
        )
        return r["message"]["content"].strip()
    except:
        return "—"

def check_answer(user, correct):
    try:
        return str(int(float(user.strip()))) == str(correct).strip()
    except:
        return user.strip() == str(correct).strip()

# ── State ──────────────────────────────────────────────────────────────────────
def init():
    for k, v in {
        "page":"welcome","kid_name":"","current_level":"level_1",
        "quiz_questions":[],"q_index":0,
        "round_score":0,"round_total":0,
        "answered":False,"feedback":"","feedback_ok":True,
        "history":[],"hint":"","show_hint":False,
        "stars":{"level_1":0,"level_2":0,"level_3":0,"level_4":0},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def start_round(data):
    pool = data[st.session_state.current_level]
    st.session_state.update({
        "quiz_questions": random.sample(pool, min(QUIZ_SIZE, len(pool))),
        "q_index":0,"round_score":0,"round_total":0,
        "answered":False,"feedback":"","hint":"","show_hint":False,"page":"quiz",
    })

def stars_for(p):
    return 3 if p>=0.9 else 2 if p>=0.7 else 1 if p>=0.5 else 0

# ── Welcome ────────────────────────────────────────────────────────────────────
def page_welcome():
    st.markdown("## 🧮 Math Tutor")
    st.markdown("Practice arithmetic and move through the levels.")
    st.divider()

    name = st.text_input("Your name", placeholder="Enter your name...")
    st.write("")
    if st.button("Start →", type="primary"):
        if name.strip():
            st.session_state.kid_name = name.strip()
            start_round(load_data())
            st.rerun()
        else:
            st.warning("Please enter your name.")

    st.write("")
    st.markdown("**Levels**")
    for lk, li in LEVELS.items():
        st.markdown(f"{li['emoji']} &nbsp; **{li['name']}**", unsafe_allow_html=True)

# ── Quiz ───────────────────────────────────────────────────────────────────────
def page_quiz(data):
    lvl  = st.session_state.current_level
    li   = LEVELS[lvl]
    idx  = st.session_state.q_index
    tot  = len(st.session_state.quiz_questions)
    item = st.session_state.quiz_questions[idx]
    q, a = item["question"], item["answer"]

    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**{li['emoji']} {li['name']}**")
    with c2:
        st.markdown(f"<p style='text-align:right;color:#6B7280;font-size:0.9rem'>{idx+1} / {tot}</p>", unsafe_allow_html=True)

    st.progress((idx) / tot)
    st.write("")

    # Question
    color = li["color"]
    st.markdown(
        f"<div style='background:white;border:1px solid #E5E7EB;border-radius:12px;"
        f"padding:2.5rem;text-align:center;margin:0.5rem 0'>"
        f"<p style='color:#6B7280;font-size:0.8rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:0.8rem'>Question</p>"
        f"<p style='font-size:3.2rem;font-weight:700;color:{color};margin:0;line-height:1'>{q}</p>"
        f"<p style='color:#9CA3AF;font-size:0.9rem;margin-top:0.6rem'>= ?</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Answer
    if not st.session_state.answered:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            user_in = st.text_input("Answer", placeholder="Type here...",
                                    key=f"a_{idx}", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Submit", use_container_width=True, type="primary", key=f"s_{idx}"):
                    if user_in.strip():
                        ok = check_answer(user_in, a)
                        st.session_state.round_total += 1
                        if ok:
                            st.session_state.round_score += 1
                        st.session_state.feedback    = random.choice(CORRECT) if ok else f"{random.choice(WRONG)} The answer is **{a}**."
                        st.session_state.feedback_ok = ok
                        st.session_state.history.append({"question":q,"correct":ok,"user_ans":user_in,"right_ans":a})
                        st.session_state.answered = True
                        st.rerun()
                    else:
                        st.warning("Type an answer first.")
            with c2:
                if st.button("Hint", use_container_width=True, key=f"h_{idx}"):
                    with st.spinner("Thinking..."):
                        st.session_state.hint      = ask_model(q)
                        st.session_state.show_hint = True
                    st.rerun()

        if st.session_state.show_hint and st.session_state.hint:
            st.info(f"💡 Hint: the answer is **{st.session_state.hint}**")

    else:
        if st.session_state.feedback_ok:
            st.success(st.session_state.feedback)
        else:
            st.error(st.session_state.feedback)

        st.write("")
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            is_last = idx + 1 >= tot
            if st.button("See Results" if is_last else "Next →",
                         use_container_width=True, type="primary", key=f"n_{idx}"):
                if is_last:
                    pct = st.session_state.round_score / st.session_state.round_total
                    s   = stars_for(pct)
                    st.session_state.stars[lvl] = max(st.session_state.stars[lvl], s)
                    st.session_state.page = "result"
                else:
                    st.session_state.update({
                        "q_index": idx+1,"answered":False,
                        "feedback":"","hint":"","show_hint":False,
                    })
                st.rerun()

# ── Result ─────────────────────────────────────────────────────────────────────
def page_result(data):
    score   = st.session_state.round_score
    total   = st.session_state.round_total
    pct     = score / total if total else 0
    lvl     = st.session_state.current_level
    li      = LEVELS[lvl]
    lvl_idx = LEVEL_ORDER.index(lvl)
    stars   = stars_for(pct)

    st.markdown(f"## Results")
    st.markdown(f"**{st.session_state.kid_name}** — {li['emoji']} {li['name']}")
    st.divider()

    # Score
    c1, c2, c3 = st.columns(3)
    c1.metric("Correct",  f"{score}")
    c2.metric("Total",    f"{total}")
    c3.metric("Score",    f"{pct*100:.0f}%")

    # Stars
    star_str = "⭐" * stars + "☆" * (3 - stars)
    st.markdown(f"<p style='font-size:1.8rem;margin:0.5rem 0'>{star_str}</p>", unsafe_allow_html=True)
    st.divider()

    # Progression
    if pct >= PASS_MARK:
        if lvl_idx < len(LEVEL_ORDER) - 1:
            nl  = LEVEL_ORDER[lvl_idx + 1]
            nli = LEVELS[nl]
            st.success(f"Great work! Moving up to {nli['emoji']} **{nli['name']}**.")
            st.session_state.current_level = nl
        else:
            st.success("🏆 You've completed all levels!")
            st.balloons()
    elif pct < FAIL_MARK and lvl_idx > 0:
        nl  = LEVEL_ORDER[lvl_idx - 1]
        nli = LEVELS[nl]
        st.warning(f"Let's practice {nli['emoji']} **{nli['name']}** a bit more.")
        st.session_state.current_level = nl
    else:
        st.info("Keep practicing to level up!")

    st.write("")

    # Breakdown
    with st.expander("Question breakdown"):
        for i, h in enumerate(st.session_state.history[-total:], 1):
            icon = "✅" if h["correct"] else "❌"
            if h["correct"]:
                st.markdown(f"{icon} &nbsp; **{h['question']}** = {h['right_ans']}")
            else:
                st.markdown(f"{icon} &nbsp; **{h['question']}** — you answered `{h['user_ans']}`, correct: `{h['right_ans']}`")

    with st.expander("Stars"):
        for lk, linfo in LEVELS.items():
            s = st.session_state.stars[lk]
            st.markdown(f"{linfo['emoji']} **{linfo['name']}** — {'⭐'*s}{'☆'*(3-s)}")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Play Again", use_container_width=True, type="primary"):
            start_round(data)
            st.rerun()
    with c2:
        if st.button("Home", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init()
    data = load_data()
    page = st.session_state.page
    if   page == "welcome": page_welcome()
    elif page == "quiz":    page_quiz(data)
    elif page == "result":  page_result(data)

if __name__ == "__main__":
    main()