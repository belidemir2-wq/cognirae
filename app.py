import json
import os
import re
import io
import time
import random
from typing import Dict, List, Any, Tuple

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Configure OpenAI client; API key is expected in the OPENAI_API_KEY environment variable.
client = OpenAI()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
IS_DEV = os.getenv("COGNIRAE_DEV", "0") == "1"
ANSWER_ALLOWED_FORMATS = {"MCQ", "SAQ"}


def difficulty_intensity(d: int) -> float:
        """Convert slider value (1-10) to an exponential intensity in [0,1]."""
        x = max(0.0, min(1.0, (d - 1) / 9))
        k = x ** 2.2  # steeper near 10
        return k


def difficulty_band_text(d: int) -> str:
        """Deterministic band instructions for the model."""
        return (
                "Difficulty bands to follow strictly:\n"
                "1-2 (Foundational): simple recall/identification; one concept only; minimal traps.\n"
                "3-4 (Core): basic application; one concept with a small twist.\n"
                "5-6 (Standard): application with reasoning; one main concept + one supporting detail; distractors target misconceptions.\n"
                "7-8 (Advanced): synthesis/comparison; integrate one related secondary concept briefly in correct reasoning; multi-step logic; nuanced distractors.\n"
                "9-10 (Elite): high-level yet fair; deeper reasoning or conditions; secondary concept strengthens reasoning; may require counterpoint/exception; avoid obscure trivia."
        )


def normalize_format(fmt: str) -> str:
    """Normalize UI format labels to canonical tokens."""
    if not isinstance(fmt, str):
        return "MCQ"
    lowered = fmt.lower()
    if "mcq" in lowered:
        return "MCQ"
    if "saq" in lowered:
        return "SAQ"
    if "leq" in lowered:
        return "LEQ"
    if "frq" in lowered:
        return "FRQ"
    return "MCQ"


UI_TO_CANONICAL = {
    "Multiple Choice (MCQ)": "MCQ",
    "Short Answer (SAQ)": "SAQ",
    "Short Answer Question (SAQ)": "SAQ",
    "Long Essay Question (LEQ)": "LEQ",
    "Free Response (FRQ)": "FRQ",
}


def selected_format_from_ui(fmt_value: str | None) -> str:
    """Map the select value to canonical format tokens."""
    if fmt_value in UI_TO_CANONICAL:
        return UI_TO_CANONICAL[fmt_value]
    return normalize_format(fmt_value or "")


def build_prompt(subject: str, fmt: str, difficulty: int, details: str) -> str:
        """Create a strict JSON-generation prompt for the chosen format and difficulty with exponential scaling."""
        k = difficulty_intensity(difficulty)
        bands = difficulty_band_text(difficulty)
        brief_integration = (
                "Brief subject integration: include one secondary, related concept in the correct answer or reasoning"
                " (e.g., a condition, exception, related theory, historical context, or mechanism) without drifting off-topic."
        )

        canonical_fmt = normalize_format(fmt)

        schemas = {
            "MCQ": {
                "schema": '{"format":"MCQ","items":[{"id":1,"stem":"...","choices":{"A":"...","B":"...","C":"...","D":"..."},"answer":"A","explanation":"..."}] }',
                "rules": [
                    "Exactly 3 items.",
                    "Each item has a stem and exactly four choices labeled A-D.",
                    "Return only MCQ items. No SAQ/FRQ/LEQ."
                ],
            },
            "SAQ": {
                "schema": '{"format":"SAQ","items":[{"id":1,"prompt":"...","parts":["A ...","B ...","C ..."],"answer":"...","rubric":"..."}] }',
                "rules": [
                    "Exactly 3 SAQ prompts.",
                    "AP-style; include parts A/B/C if useful, but no multiple-choice letters.",
                    "No choices object, no A-D options, no 'Correct answer' strings.",
                    "Return only SAQ items. No MCQ/FRQ/LEQ."
                ],
            },
            "LEQ": {
                "schema": '{"format":"LEQ","items":[{"id":1,"prompt":"An LEQ prompt with a task verb (evaluate/compare/assess).","requirements":["thesis","contextualization","evidence","reasoning","complexity"],"rubric":"...","answer":"Concise thesis + outline of argument","thesis_guidance":"How to frame the thesis"}]}',
                "rules": [
                    "Exactly 1 LEQ prompt.",
                    "Use strong task verbs (evaluate, compare, assess) and period/context if relevant.",
                    "Include a requirements list (thesis, contextualization, evidence, reasoning, complexity).",
                    "Provide a concise rubric and a short answer/thesis outline (not a full essay).",
                    "No choices object, no A-D options, no 'Correct answer', no MCQ elements.",
                    "Return only LEQ items. No MCQ/SAQ/FRQ."
                ],
            },
            "FRQ": {
                "schema": '{"format":"FRQ","items":[{"id":1,"prompt":"...","rubric":"..."}] }',
                "rules": [
                    "Exactly 3 FRQ prompts.",
                    "No choices object, no A-D options, no MCQ elements.",
                    "Return only FRQ items. No MCQ/SAQ/LEQ."
                ],
            },
        }

        fmt_rules = schemas.get(canonical_fmt, schemas["MCQ"])
        rules_block = "\n".join(f"- {r}" for r in fmt_rules["rules"])

        return f"""
You are an expert educator generating practice exam questions only.
Subject/topic: {subject}
Question format (canonical): {canonical_fmt}
Difficulty slider: {difficulty}/10; intensity k (0-1): {k:.3f}. As difficulty approaches 10, increase cognitive demand exponentially.
Additional constraints: {details}

{bands}
{brief_integration}
If you cannot increase difficulty without becoming obscure, increase reasoning complexity and distractor nuance instead of adding random facts.

Strict JSON schema for this format:
{fmt_rules['schema']}

Format-specific rules (hard constraints):
{rules_block}

Difficulty application rules (use k):
- Concepts: at low k use one concept; mid k add one supporting detail; high k integrate one brief related concept into correct reasoning.
- Reasoning depth: higher k -> multi-step logic/comparison/synthesis; lower k -> recall/identification.
- Distractors (MCQ): higher k -> more plausible, target misconceptions; still exactly one correct option.
- Prompt constraints (SAQ/FRQ/LEQ): higher k -> clearer task verbs, added conditions, comparisons, or scenario-based application; avoid obscurity.

Global rules:
- Do NOT write full essays.
- Return STRICT JSON only (no markdown, no commentary, no prose outside JSON).
- JSON must have top-level keys: format, items.
"""


def parse_response(raw: str) -> Dict[str, Any]:
    """Parse the JSON response, raising a helpful error on failure."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse model response as JSON: {exc}\nRaw response:\n{raw}")


def _contains_mcq_pattern(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"\b[A-D][\.\):]\s", text)) or "Correct answer" in text


def is_mcq_like(text: str) -> bool:
    """Detect MCQ-like patterns such as A., A), or explicit 'Correct answer'."""
    if not text:
        return False
    patterns = [r"\bA[\.\)]\s", r"\bB[\.\)]\s", r"\bC[\.\)]\s", r"\bD[\.\)]\s", r'"[A-D]"\s*:\s']
    if any(re.search(p, text) for p in patterns):
        return True
    if "Correct answer" in text or "choices" in text or "options" in text:
        return True
    return _contains_mcq_pattern(text)


def validate_mcq(parsed: Dict[str, Any]) -> Tuple[bool, str]:
    if parsed.get("format") != "MCQ":
        return False, "format field is not MCQ"
    items = parsed.get("items")
    if not isinstance(items, list) or not items:
        return False, "items missing"
    for item in items:
        choices = item.get("choices") if isinstance(item, dict) else None
        if not isinstance(choices, dict):
            return False, "choices missing"
        letters = ["A", "B", "C", "D"]
        if any(l not in choices or not isinstance(choices[l], str) or not choices[l].strip() for l in letters):
            return False, "choices must have A-D text"
        ans = item.get("answer") if isinstance(item, dict) else None
        if ans not in letters:
            return False, "answer must be A-D"
    return True, ""


def validate_saq(parsed: Dict[str, Any], raw_text: str) -> Tuple[bool, str]:
    if parsed.get("format") != "SAQ":
        return False, "format field is not SAQ"
    if _contains_mcq_pattern(raw_text):
        return False, "MCQ pattern detected"
    items = parsed.get("items")
    if not isinstance(items, list) or not items:
        return False, "items missing"
    for item in items:
        if not isinstance(item, dict):
            return False, "item not object"
        if "choices" in item:
            return False, "choices object present"
    return True, ""


def validate_frq(parsed: Dict[str, Any], raw_text: str) -> Tuple[bool, str]:
    if parsed.get("format") != "FRQ":
        return False, "format field is not FRQ"
    if _contains_mcq_pattern(raw_text):
        return False, "MCQ pattern detected"
    items = parsed.get("items")
    if not isinstance(items, list) or not items:
        return False, "items missing"
    for item in items:
        if isinstance(item, dict) and "choices" in item:
            return False, "choices object present"
    return True, ""


def validate_leq(parsed: Dict[str, Any], raw_text: str) -> Tuple[bool, str]:
    ok, reason = validate_frq(parsed, raw_text)
    if not ok:
        return ok, reason
    items = parsed.get("items") or []
    if len(items) != 1:
        return False, "LEQ must be exactly 1 item"
    return True, ""


def validate_by_format(parsed: Dict[str, Any], raw_text: str, fmt: str) -> Tuple[bool, str]:
    canonical = normalize_format(fmt)
    if canonical == "MCQ":
        return validate_mcq(parsed)
    if canonical == "SAQ":
        return validate_saq(parsed, raw_text)
    if canonical == "FRQ":
        return validate_frq(parsed, raw_text)
    if canonical == "LEQ":
        return validate_leq(parsed, raw_text)
    return False, "Unknown format"


def validate_output(selected_format: str, parsed: Dict[str, Any], raw_text: str) -> Tuple[bool, str]:
    """Deterministic validation of model output against expected format."""
    canonical = normalize_format(selected_format)
    if not isinstance(parsed, dict):
        return False, "Response is not a JSON object"
    if parsed.get("format") != canonical:
        return False, "format field mismatch"

    items = parsed.get("items")
    if not isinstance(items, list) or not items:
        return False, "items missing"

    # Catch MCQ leakage for non-MCQ formats
    if canonical != "MCQ" and is_mcq_like(raw_text):
        return False, "MCQ pattern detected in non-MCQ format"

    if canonical == "LEQ":
        if len(items) != 1:
            return False, "LEQ must have exactly one item"
        item = items[0] if isinstance(items[0], dict) else {}
        if "choices" in item:
            return False, "LEQ item contains choices"
        if "correct" in raw_text.lower():
            return False, "LEQ output mentions correct answer"
        if is_mcq_like(raw_text):
            return False, "MCQ pattern detected in LEQ"
        prompt_val = item.get("prompt") if isinstance(item, dict) else None
        if not isinstance(prompt_val, str) or not prompt_val.strip():
            return False, "LEQ prompt missing"
        requirements = item.get("requirements") if isinstance(item, dict) else None
        if not isinstance(requirements, list) or not requirements or not all(isinstance(x, str) and x.strip() for x in requirements):
            return False, "LEQ requirements missing or invalid"
        rubric_val = item.get("rubric") if isinstance(item, dict) else None
        if rubric_val is not None and (not isinstance(rubric_val, str) or not rubric_val.strip()):
            return False, "LEQ rubric invalid"
        answer_val = item.get("answer") if isinstance(item, dict) else None
        answer_outline = item.get("answer_outline") if isinstance(item, dict) else None
        if answer_val is not None and (not isinstance(answer_val, str) or not answer_val.strip()):
            return False, "LEQ answer invalid"
        if answer_outline is not None and (not isinstance(answer_outline, str) or not answer_outline.strip()):
            return False, "LEQ answer_outline invalid"
        return True, ""

    # Reuse existing validators for other formats
    return validate_by_format(parsed, raw_text, canonical)


def repair_prompt(original_prompt: str, fmt: str, reason: str) -> str:
    canonical = normalize_format(fmt)
    schema_hints = {
        "MCQ": '{"format":"MCQ","items":[{"id":1,"stem":"...","choices":{"A":"...","B":"...","C":"...","D":"..."},"answer":"A","explanation":"..."}]}',
        "SAQ": '{"format":"SAQ","items":[{"id":1,"prompt":"...","parts":["A ...","B ...","C ..."],"answer":"...","rubric":"..."}]}',
        "FRQ": '{"format":"FRQ","items":[{"id":1,"prompt":"...","rubric":"..."}]}',
        "LEQ": '{"format":"LEQ","items":[{"id":1,"prompt":"An LEQ prompt with a task verb (evaluate/compare/assess).","requirements":["thesis","contextualization","evidence","reasoning","complexity"],"rubric":"...","answer":"Concise thesis + outline of argument","thesis_guidance":"How to frame the thesis"}]}',
    }
    schema_hint = schema_hints.get(canonical, "Follow the required schema exactly.")
    return (
        original_prompt
        + "\n\nCORRECTION: You violated the required output. Reason: "
        + reason
        + "\nYou violated the required "
        + canonical
        + " format. Return ONLY "
        + canonical
        + " JSON in the specified schema. No MCQ elements. Output JSON only. Use this schema: "
        + schema_hint
    )


def looks_like_assignment(text: str) -> bool:
    """Heuristic to catch pasted assignments or questions to avoid cheating."""
    lowered = text.lower()
    triggers = [
        "assignment",
        "homework",
        "worksheet",
        "quiz",
        "exam",
        "test",
        "points",
        "due",
        "submit",
        "prompt",
        "question:",
        "answer:",
        "answer this",
        "solve",
        "please help",
        "show work",
    ]
    if any(t in lowered for t in triggers):
        return True
    # Long text with multiple questions is likely an upload/paste.
    if len(text) > 220 and text.count("?") >= 2:
        return True
    return False


def question_without_preface(text: str) -> bool:
    """Detect direct questions unless clearly framed as examples for new generation."""
    lowered = text.strip().lower()
    if not lowered:
        return False

    allow_prefixes = [
        "generate a question along the lines of",
        "here's a question that's in the format of our exam, make other like it",
        "here is a question that's in the format of our exam, make other like it",
    ]
    if any(lowered.startswith(p) for p in allow_prefixes):
        return False

    question_triggers = [
        "?",
        "what is",
        "solve",
        "differentiate",
        "integrate",
        "derivative",
        "answer",
        "compute",
        "evaluate",
        "prove",
    ]
    return any(t in lowered for t in question_triggers)


def classify_intent(subject: str, details: str, fmt: str | None) -> Tuple[str, float, str]:
    """Classify intent into ANSWER_SEEKING, QUESTION_GENERATION, or AMBIGUOUS.

    Rules (deterministic, explainable):
    - Allow generation verbs/format/unit cues to pass as QUESTION_GENERATION.
    - Answer-seeking requires multiple strong signals (direct question, minimal context, answer verbs).
    - Otherwise, mark AMBIGUOUS and redirect to safe generation.
    """

    text = f"{subject}\n{details}".strip()
    lowered = text.lower()

    allow_prefixes = [
        "generate a question along the lines of",
        "here's a question that's in the format of our exam, make other like it",
        "here is a question that's in the format of our exam, make other like it",
        "use past questions as examples",
    ]

    generation_verbs = ["generate", "create", "write", "make", "build", "draft", "practice", "produce"]
    format_tokens = ["mcq", "multiple choice", "saq", "short answer", "frq", "free response", "leq", "long essay"]
    exam_tokens = ["ap", "sat", "ib", "unit", "chapter", "module", "exam-style", "ap-style"]

    answer_verbs = ["answer", "solve", "what is", "define", "explain", "give me the answer", "provide the answer", "give me the solution"]
    numbered_q_tokens = ["here is my question", "q:", "#1", "1)", "select the correct answer", "correct option"]

    has_generation_verb = any(v in lowered for v in generation_verbs)
    has_format = any(t in lowered for t in format_tokens)
    has_exam = any(t in lowered for t in exam_tokens)
    has_allow_prefix = any(lowered.startswith(p) for p in allow_prefixes)

    # Direct-question signals
    question_like = bool(re.search(r"\b(what|who|when|where|why|how|which|did|do|does|is|are|can|could|will|would|should)\b", lowered))
    has_qmark = "?" in text
    short_len = len(text) < 180
    has_answer_verb = any(v in lowered for v in answer_verbs)
    has_numbered = any(t in lowered for t in numbered_q_tokens)
    looks_assignment = looks_like_assignment(text)
    unprefaced_question = question_without_preface(details)
    mcq_pattern = bool(re.search(r"\b(a\)|b\)|c\)|d\))", lowered))

    # Positive allowance: clear generation intent
    if has_allow_prefix or (has_generation_verb and (has_format or has_exam)):
        return ("QUESTION_GENERATION", 0.82, "Explicit request to generate new questions with format/topic cues.")
    if has_format and has_exam:
        return ("QUESTION_GENERATION", 0.76, "Uses exam/unit plus format to request new practice questions.")
    if has_generation_verb and not (has_answer_verb or has_numbered) and not has_qmark:
        return ("QUESTION_GENERATION", 0.7, "Generation verbs without answer-seeking cues.")

    # Strong answer-seeking: require multiple signals
    answer_signals = sum(
        [
            1 if has_qmark and question_like else 0,
            1 if has_answer_verb else 0,
            1 if has_numbered else 0,
            1 if unprefaced_question else 0,
            1 if mcq_pattern else 0,
        ]
    )

    if looks_assignment and (has_qmark or has_answer_verb or has_numbered):
        return ("ANSWER_SEEKING", 0.86, "Looks like an assignment/worksheet with direct question cues.")

    if answer_signals >= 2 and short_len:
        return ("ANSWER_SEEKING", 0.8, "Direct question with answer-seeking language and little context.")

    # Ambiguous default: play it safe but generate new questions
    return ("AMBIGUOUS", 0.5, "Not clearly answer-seeking; will generate fresh practice instead.")


def derive_topic(subject: str, details: str) -> str:
    """Convert question-like input into a neutral topic label."""
    candidate = subject.strip() or details.strip()
    if not candidate:
        return "practice topic"

    text = re.sub(r"\?+$", "", candidate).strip()
    text = re.sub(r"^(what|who|when|where|why|how|which|did|do|does|is|are|can|could|will|would|should)\s+", "", text, flags=re.I)
    text = re.sub(r"^(q[:)]\s*|question[:)]\s*)", "", text, flags=re.I)
    if len(text.split()) <= 3:
        return text or "practice topic"
    return text


def generate_safe_support(topic: str) -> Dict[str, Any]:
    """Produce safe guidance instead of answering a copied question."""
    safe_prompt = {
        "role": "user",
        "content": (
            "Provide help for the topic: "
            f"{topic}. Return JSON with keys mini_lesson (<=80 words), practice_questions (3-5 items,"
            " no answers), checklist (3-6 bullets), hint_ladder (3 short steps). Do NOT answer or solve"
            " the user's question. Do not include the original wording."
        ),
    }

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Academic integrity coach: never solve or restate the user's pasted question."
                    " Give only mini-lesson, new practice questions (without answers), study checklist,"
                    " and a 3-step hint ladder. Keep it brief and safe. Return strict JSON."
                ),
            },
            safe_prompt,
        ],
        temperature=0.4,
    )
    raw = response.choices[0].message.content
    return parse_response(raw)


def _wrap_lines(c: canvas.Canvas, text: str, x: int, y: int, max_width: int, leading: int = 14) -> int:
    """Draw wrapped text and return the next y position."""
    if not text:
        return y
    from reportlab.lib.utils import simpleSplit  # local import to keep top clean

    lines = simpleSplit(text, "Helvetica", 11, max_width)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def build_pdf_bytes(topic: str, fmt: str, difficulty: int, data: Dict[str, Any], include_answers: bool) -> bytes:
    """Build a PDF in memory from generated questions."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    margin_x = 50
    y = height - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_x, y, "Practice Questions")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, f"Topic: {topic or '—'}")
    y -= 14
    c.drawString(margin_x, y, f"Format: {fmt or '—'} | Difficulty: {difficulty}/10")
    y -= 20

    canonical = data.get("format") if isinstance(data, dict) else normalize_format(fmt)
    items: List[Dict[str, Any]] = data.get("items", []) if isinstance(data, dict) else []
    for idx, q in enumerate(items, start=1):
        if y < 120:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 60

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin_x, y, f"Question {idx}")
        y -= 16
        c.setFont("Helvetica", 11)
        if canonical == "MCQ":
            y = _wrap_lines(c, q.get("stem", ""), margin_x, y, max_width=int(width - 2 * margin_x))
            y -= 6
            choices = q.get("choices", {}) if isinstance(q, dict) else {}
            for label in ["A", "B", "C", "D"]:
                line = f"{label}. {choices.get(label, '')}"
                y = _wrap_lines(c, line, margin_x + 14, y, max_width=int(width - 2 * margin_x - 14))
            y -= 4
            if include_answers:
                c.setFont("Helvetica-Oblique", 11)
                y = _wrap_lines(c, f"Answer: {q.get('answer', '')}", margin_x, y, max_width=int(width - 2 * margin_x))
                if "explanation" in q:
                    y = _wrap_lines(c, f"Explanation: {q.get('explanation', '')}", margin_x, y, max_width=int(width - 2 * margin_x))
                c.setFont("Helvetica", 11)
                y -= 6
        else:
            y = _wrap_lines(c, q.get("prompt", ""), margin_x, y, max_width=int(width - 2 * margin_x))
            if canonical == "SAQ":
                parts = q.get("parts", []) if isinstance(q, dict) else []
                for part in parts:
                    y = _wrap_lines(c, part, margin_x + 10, y, max_width=int(width - 2 * margin_x - 10))
            if include_answers:
                c.setFont("Helvetica-Oblique", 11)
                if "answer" in q:
                    y = _wrap_lines(c, f"Answer: {q.get('answer', '')}", margin_x, y, max_width=int(width - 2 * margin_x))
                if "rubric" in q:
                    y = _wrap_lines(c, f"Rubric: {q.get('rubric', '')}", margin_x, y, max_width=int(width - 2 * margin_x))
                if "thesis_guidance" in q:
                    y = _wrap_lines(c, f"Thesis guidance: {q.get('thesis_guidance', '')}", margin_x, y, max_width=int(width - 2 * margin_x))
                c.setFont("Helvetica", 11)
                y -= 6

        y -= 8

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def render_questions(data: Dict[str, Any], show_answers: bool, selected_format: str) -> None:
    canonical = data.get("format") if isinstance(data, dict) else "MCQ"
    if canonical != normalize_format(selected_format):
        st.error("Format mismatch between selection and response; output suppressed.")
        return

    def render_leq_frq_guidance(fmt: str) -> None:
        st.info("Answer generation is disabled for LEQ/FRQ to save credits. Use this rubric-style checklist:")
        st.markdown("**Thesis:** Takes a clear position; addresses the prompt directly.")
        st.markdown("**Evidence:** Multiple specific, relevant examples; tie each back to the claim.")
        st.markdown("**Reasoning:** Explain how evidence supports the claim; avoid mere lists.")
        st.markdown("**Complexity / Counterargument:** Acknowledge limits or counterpoints and respond briefly.")
        st.markdown("**Planning outline:**\n- Claim/thesis\n- 2-3 evidence points + link-back sentences\n- Brief counter/limit and response\n- Closing sentence tying back to prompt")

    items: List[Dict[str, Any]] = data.get("items", []) if isinstance(data, dict) else []
    for idx, q in enumerate(items, start=1):
        st.markdown(f"### Question {idx}")

        if canonical == "MCQ":
            st.markdown(q.get("stem", ""))
            choices = q.get("choices", {}) if isinstance(q, dict) else {}
            for letter in ["A", "B", "C", "D"]:
                st.markdown(f"- **{letter}.** {choices.get(letter, '')}")
        elif canonical == "SAQ":
            st.markdown(q.get("prompt", ""))
            parts = q.get("parts", []) if isinstance(q, dict) else []
            for part in parts:
                st.markdown(f"- {part}")
        elif canonical == "LEQ":
            st.markdown(q.get("prompt", ""))
            requirements = q.get("requirements", []) if isinstance(q, dict) else []
            if requirements:
                st.markdown("**Requirements:**")
                for elem in requirements:
                    st.markdown(f"- {elem}")
        else:
            st.markdown(q.get("prompt", ""))

        if show_answers:
            with st.expander("Show answer & guidance", expanded=False):
                if "answer" in q:
                    st.markdown(f"**Answer:** {q.get('answer', '')}")
                elif "answer_outline" in q:
                    st.markdown(f"**Answer outline:** {q.get('answer_outline', '')}")
                if "explanation" in q:
                    st.markdown(f"**Explanation:** {q.get('explanation', '')}")
                if "rubric" in q:
                    st.markdown(f"**Rubric:** {q.get('rubric', '')}")
                if "thesis_guidance" in q:
                    st.markdown(f"**Thesis guidance:** {q.get('thesis_guidance', '')}")

    # For LEQ/FRQ, always surface rubric-style guidance instead of answers.
    if canonical not in ANSWER_ALLOWED_FORMATS:
        render_leq_frq_guidance(canonical)


def render_safe_support(data: Dict[str, Any]) -> None:
    """Render mini-lesson, fresh practice questions, checklist, and hint ladder without answers."""
    mini = data.get("mini_lesson", "")
    practice = data.get("practice_questions", [])
    checklist = data.get("checklist", [])
    ladder = data.get("hint_ladder", [])

    if mini:
        st.subheader("Mini-lesson")
        st.write(mini)

    if practice:
        st.subheader("New practice questions (no answers)")
        for idx, item in enumerate(practice, 1):
            st.markdown(f"{idx}. {item}")

    if checklist:
        st.subheader("Study checklist")
        for item in checklist:
            st.markdown(f"- {item}")

    if ladder:
        st.subheader("Hint ladder")
        for step_idx, hint in enumerate(ladder, 1):
            st.markdown(f"Hint {step_idx}: {hint}")


# --- Streamlit UI ---
st.set_page_config(page_title="Cognirae", page_icon="🧠", layout="wide")

# Base styling for cards, toolbar, and subtle dividers
st.markdown(
    """
    <style>
        .c-card {
            background: #0f1115;
            border: 1px solid #1d222a;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
            margin-bottom: 14px;
        }
        .c-card h3 { margin-bottom: 8px; }
        .c-section-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
        .c-helper { color: #8891a1; font-size: 13px; margin-bottom: 8px; }
        .c-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 8px 12px;
            font-size: 13px;
        }
        .c-pill { background: #131822; border: 1px solid #1f2632; padding: 8px 10px; border-radius: 10px; }
        .c-toolbar { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-top: 6px; }
        .c-divider { height: 1px; background: #1f2632; margin: 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo + title centered
header = st.container()
with header:
    left, center, right = st.columns([1, 1, 1])
    with center:
        try:
            st.image("cognirae_logo.png", width=120)
        except Exception:
            pass  # If the logo file is missing, skip quietly.
        st.markdown(
            """
            <div style="text-align:center;">
                <h1 style="margin-bottom:6px;">Cognirae</h1>
                <p style="color:#8e98a7; margin-top:0;">Generate tailored practice questions without full essays. Reveal answers only when you're ready.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if "last_run" not in st.session_state:
    st.session_state.last_run = None
if "data" not in st.session_state:
    st.session_state.data = None
if "questions" not in st.session_state:
    st.session_state.questions = None
if "show_answers" not in st.session_state:
    st.session_state.show_answers = False
if "safe_support" not in st.session_state:
    st.session_state.safe_support = None
if "questions_generated" not in st.session_state:
    st.session_state.questions_generated = False
if "answers_revealed" not in st.session_state:
    st.session_state.answers_revealed = False
if "selected_format" not in st.session_state:
    st.session_state.selected_format = "MCQ"
if "dev_debug" not in st.session_state:
    st.session_state.dev_debug = False

topic_choices = [
    "Algebra - Quadratic functions",
    "US History - Reconstruction",
    "Biology - Cellular respiration",
    "Chemistry - Stoichiometry",
    "English - Literary analysis",
]

# Sidebar: global context & integrity
with st.sidebar:
    st.markdown("### Cognirae")
    st.caption("Academic-safe practice question generator.")
    st.markdown("---")
    st.markdown("**Usage notes**")
    st.caption("- Enter a topic, not a question.\n- Do not paste live quiz or homework items.\n- Reveal answers only when ready.")
    st.markdown("---")
    topic_pick = st.selectbox("Quick topic chooser", ["(choose a topic)"] + topic_choices)
    attest = st.checkbox(
        "I confirm I am not pasting a live quiz or homework question.",
        help="We must take these measures to prevent cheating on assignments.",
    )
    if IS_DEV:
        dev_debug = st.checkbox("Dev: show debug selection", value=st.session_state.get("dev_debug", False))
        st.session_state.dev_debug = dev_debug
    else:
        st.session_state.dev_debug = False

# Non-typable dropdown via custom HTML <select>; typing is blocked via JS keydown preventDefault.
OPTIONS = [
    "Multiple Choice (MCQ)",
    "Short Answer (SAQ)",
    "Long Essay Question (LEQ)",
    "Free Response (FRQ)",
]

fmt_html = """
<style>
    :root {
        --cognirae-bg: #0f1115;
        --cognirae-border: #2b2f36;
        --cognirae-border-focus: #6aa0ff;
        --cognirae-text: #f5f7fa;
        --cognirae-subtle: #c3c7cf;
    }
    .cognirae-select-wrap {
        display: flex;
        flex-direction: column;
        gap: 6px;
        font-family: inherit;
        color: var(--cognirae-text);
    }
    .cognirae-select-label {
        font-size: 14px;
        font-weight: 600;
        color: var(--cognirae-subtle);
        letter-spacing: 0.1px;
    }
    .cognirae-select-shell {
        position: relative;
    }
    .cognirae-select {
        appearance: none;
        -webkit-appearance: none;
        -moz-appearance: none;
        width: 100%;
        padding: 12px 44px 12px 14px;
        border-radius: 12px;
        border: 1px solid var(--cognirae-border);
        background: var(--cognirae-bg);
        font-size: 14px;
        color: var(--cognirae-text);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.14);
        transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease, color 120ms ease;
    }
    .cognirae-select:hover {
        border-color: #3b414a;
        background: #14171c;
    }
    .cognirae-select:focus {
        outline: none;
        border-color: var(--cognirae-border-focus);
        box-shadow: 0 0 0 3px rgba(106, 160, 255, 0.18);
        background: #161a20;
    }
    .cognirae-select option {
        background: #0f1115;
        color: var(--cognirae-text);
    }
    .cognirae-arrow {
        position: absolute;
        right: 14px;
        top: 50%;
        transform: translateY(-50%);
        width: 10px;
        height: 6px;
        pointer-events: none;
    }
    .cognirae-arrow::before {
        content: "";
        position: absolute;
        inset: 0;
        margin: auto;
        width: 0;
        height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 6px solid #c3c7cf;
    }
</style>
<div class="cognirae-select-wrap">
    <label for="fmt" class="cognirae-select-label">Question format</label>
    <div class="cognirae-select-shell">
        <select id="fmt" class="cognirae-select">
            {options}
        </select>
        <span class="cognirae-arrow"></span>
    </div>
</div>
<script>
    const sel = document.getElementById('fmt');
    const send = (val) => window.parent.postMessage({isStreamlitMessage:true, type:'streamlit:setComponentValue', value: val}, '*');
    sel.addEventListener('change', e => send(e.target.value));
    sel.addEventListener('keydown', e => { e.preventDefault(); return false; });
    sel.addEventListener('keypress', e => { e.preventDefault(); return false; });
    sel.addEventListener('keyup', e => { e.preventDefault(); return false; });
    send(sel.value);
</script>
""".replace('{options}', '\n'.join([f'<option value="{o}">{o}</option>' for o in OPTIONS]))

left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    with st.container():
        st.markdown('<div class="c-card">', unsafe_allow_html=True)
        st.markdown('<div class="c-section-title">Setup</div>', unsafe_allow_html=True)
        topic_entry = st.text_input(
            "Topic statement (not a question)",
            placeholder="AP Psychology Unit 8: Learning and Cognition",
            help="Generate new practice questions. Do not paste questions expecting answers.",
        )
        st.markdown('<div class="c-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="c-section-title">Constraints</div>', unsafe_allow_html=True)
        details = st.text_area(
            "Constraints and preferences (Generate new practice questions. Do not paste questions expecting answers.)",
            placeholder="Practice SAQs, medium difficulty; emphasize application; mirror past exam style",
            help="Add guidance like difficulty, style, or focus areas. Avoid asking for answers to specific questions.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="c-card">', unsafe_allow_html=True)
        st.markdown('<div class="c-section-title">Question style</div>', unsafe_allow_html=True)
        fmt_label = st.selectbox("Question format", OPTIONS, index=0)
        difficulty = st.slider("Difficulty", min_value=1, max_value=10, value=6, help="1 = very easy, 10 = exam-level difficulty")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="c-card">', unsafe_allow_html=True)
        st.markdown('<div class="c-section-title">Generation</div>', unsafe_allow_html=True)
        # Generate is enabled initially; after first generation it is disabled to steer users to Regenerate.
        generate = st.button(
            "Generate Questions",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.questions_generated,
            key="generate_btn",
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Resolve selected format immediately after UI controls
selected_format = selected_format_from_ui(fmt_label if isinstance(fmt_label, str) else None)
st.session_state.selected_format = selected_format
if IS_DEV and st.session_state.dev_debug:
    st.write("DEBUG selected_label:", fmt_label)
    st.write("DEBUG selected_format:", selected_format)

# Subject/topic resolution using sidebar quick pick as fallback
subject = topic_entry.strip() or (topic_pick if topic_pick != "(choose a topic)" else "")

with right_col:
    # Live summary panel
    st.markdown('<div class="c-card">', unsafe_allow_html=True)
    st.markdown('<div class="c-section-title">Live summary</div>', unsafe_allow_html=True)
    summary_fmt = selected_format
    st.markdown(
        f"""
        <div class="c-summary">
            <div class="c-pill"><strong>Topic</strong><br>{subject or '—'}</div>
            <div class="c-pill"><strong>Format</strong><br>{summary_fmt or '—'}</div>
            <div class="c-pill"><strong>Difficulty</strong><br>{difficulty if 'difficulty' in locals() else '—'}/10</div>
            <div class="c-pill"><strong>Constraints</strong><br>{'Provided' if details else 'None'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Results card with toolbar
    st.markdown('<div class="c-card">', unsafe_allow_html=True)
    st.markdown('<div class="c-section-title">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="c-helper">Use the toolbar to manage output.</div>', unsafe_allow_html=True)

    can_reveal = (
        st.session_state.questions_generated
        and not st.session_state.answers_revealed
        and st.session_state.data is not None
        and normalize_format(selected_format) in ANSWER_ALLOWED_FORMATS
    )
    can_regen = st.session_state.questions_generated
    can_actions = st.session_state.questions_generated

    toolbar_cols = st.columns(5, gap="small")
    reveal = toolbar_cols[0].button("Reveal Answers", disabled=not can_reveal, use_container_width=True, key="reveal_btn")
    copy_clicked = toolbar_cols[1].button("Copy Questions", disabled=not can_actions, use_container_width=True)
    with toolbar_cols[2]:
        if can_actions and st.session_state.questions:
            pdf_bytes = build_pdf_bytes(
                subject,
                selected_format,
                difficulty,
                st.session_state.questions,
                include_answers=st.session_state.answers_revealed,
            )
        else:
            pdf_bytes = None
        st.download_button(
            "Download PDF",
            data=pdf_bytes or b"",
            file_name=("practice_questions_with_answers.pdf" if st.session_state.answers_revealed else "practice_questions.pdf"),
            mime="application/pdf",
            disabled=not can_actions or pdf_bytes is None,
            use_container_width=True,
        )
    regenerate = toolbar_cols[3].button("Regenerate", disabled=not can_regen, use_container_width=True, key="regen_btn")
    clear_clicked = toolbar_cols[4].button(
        "Clear",
        disabled=st.session_state.data is None and st.session_state.safe_support is None,
        use_container_width=True,
    )

    # Status will be rendered after button logic to reflect up-to-date session state.

    if copy_clicked and st.session_state.data:
        st.info("Select and copy from the rendered questions below.")

    if clear_clicked:
        st.session_state.data = None
        st.session_state.questions = None
        st.session_state.safe_support = None
        st.session_state.show_answers = False
        st.session_state.questions_generated = False
        st.session_state.answers_revealed = False
        st.info("Cleared the current results.")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

if generate or regenerate:
    if not subject.strip():
        st.warning("Please enter a topic (not a question).")
    elif not attest:
        st.warning("Please confirm you are not pasting a live quiz or homework question. We must take these measures to prevent cheating on assignments.")
    else:
        intent, confidence, intent_reason = classify_intent(subject, details, fmt_label if isinstance(fmt_label, str) else None)

        if intent == "ANSWER_SEEKING":
            st.session_state.data = None
            st.session_state.questions = None
            st.session_state.safe_support = None
            st.session_state.questions_generated = False
            st.session_state.answers_revealed = False
            st.error(
                "I can’t provide direct answers to copied questions. "
                + intent_reason
                + " We must take these measures to prevent cheating on assignments."
            )
            safe_topic = derive_topic(subject, details)
            with st.spinner("Preparing safe study guidance..."):
                try:
                    support = generate_safe_support(safe_topic)
                    st.session_state.safe_support = support
                except Exception as exc:  # noqa: BLE001
                    st.warning("Could not generate safe support right now. Please try again.")
                    st.session_state.safe_support = None
            if st.session_state.safe_support:
                render_safe_support(st.session_state.safe_support)

        else:
            st.session_state.show_answers = False
            st.session_state.answers_revealed = False
            st.session_state.safe_support = None

            # For ambiguous, convert to a neutral topic and surface a notice; still generate practice questions only.
            topic_for_prompt = subject
            if intent == "AMBIGUOUS":
                topic_for_prompt = derive_topic(subject, details)
                st.info(
                    "To keep this a practice tool, I generated new questions instead of answering a specific prompt."
                )

            status_box = st.empty()
            cycle_msgs = ["Activating sarabot.ai", "Low storage!!! Now replacing following files: Systems32.exe, googlechrome.app, redtubeMidgets.mp4", "Installing malaware onto system hard drive"]
            random.shuffle(cycle_msgs)
            for msg in cycle_msgs:
                status_box.info(msg)
                time.sleep(1.4)

            prompt = build_prompt(topic_for_prompt, selected_format, difficulty, details)
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic-safe question generator. Refuse to solve or restate user-submitted"
                            " questions or assignments. Only create fresh practice questions. Provide answer keys"
                            " in JSON but rely on the UI to reveal them; do not present answers in prose."
                        ),
                    },
                    {
                        "role": "system",
                        "content": "Return strict JSON only.",
                    },
                    {
                        "role": "system",
                        "content": (
                            f"Generate only {selected_format} in the exact schema. For LEQ, include a requirements list"
                            " (thesis, contextualization, evidence, reasoning, complexity) and never include choices,"
                            " A-D letters, or 'Correct answer'. Output JSON only."
                        ),
                    },
                    {
                        "role": "system",
                        "content": (
                            "Academic integrity: If the user provides assignment, quiz, test, or worksheet text, or a direct"
                            " question to solve, refuse and ask for a topic summary instead. Only generate fresh practice"
                            " questions; never solve, rewrite, or answer submitted questions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            status_box.empty()
            raw = response.choices[0].message.content
            try:
                parsed = parse_response(raw)
                valid, reason = validate_output(selected_format, parsed, raw)
                if not valid:
                    correction = repair_prompt(prompt, selected_format, reason)
                    repair_response = client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an academic-safe question generator. Refuse to solve or restate user-submitted"
                                    " questions or assignments. Only create fresh practice questions. Provide answer keys"
                                    " in JSON but rely on the UI to reveal them; do not present answers in prose."
                                ),
                            },
                            {"role": "system", "content": "Return strict JSON only."},
                            {
                                "role": "system",
                                "content": (
                                    f"Generate only {selected_format} in the exact schema. For LEQ, include a requirements list"
                                    " (thesis, contextualization, evidence, reasoning, complexity) and never include choices,"
                                    " A-D letters, or 'Correct answer'. Output JSON only."
                                ),
                            },
                            {
                                "role": "system",
                                "content": (
                                    "Academic integrity: If the user provides assignment, quiz, test, or worksheet text, or a direct"
                                    " question to solve, refuse and ask for a topic summary instead. Only generate fresh practice"
                                    " questions; never solve, rewrite, or answer submitted questions."
                                ),
                            },
                            {"role": "user", "content": correction},
                        ],
                        temperature=0.4,
                    )
                    raw = repair_response.choices[0].message.content
                    parsed = parse_response(raw)
                    valid, reason = validate_output(selected_format, parsed, raw)

                if not valid:
                    raise ValueError(f"Response failed validation: {reason}")

                st.session_state.data = parsed
                st.session_state.questions = parsed
                st.session_state.last_run = topic_for_prompt
                st.session_state.questions_generated = True
                st.session_state.answers_revealed = False
                st.success("Questions generated. Use 'Reveal Answers' when ready.")
                st.rerun()  # Ensure UI re-renders with updated state so buttons/status reflect availability.
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                st.session_state.data = None
                st.session_state.questions = None
                st.session_state.questions_generated = False
                st.session_state.answers_revealed = False

if reveal and st.session_state.data is not None and not st.session_state.answers_revealed:
    canonical = normalize_format(st.session_state.selected_format)
    if canonical not in ANSWER_ALLOWED_FORMATS:
        st.info("Answer generation is disabled for LEQ/FRQ to save credits. Review the rubric guidance below.")
    else:
        # Reveal answers once per generation for allowed formats
        st.session_state.show_answers = True
        st.session_state.answers_revealed = True
        st.rerun()

if st.session_state.data:
    canonical = normalize_format(st.session_state.selected_format)
    show_ans = st.session_state.show_answers if canonical in ANSWER_ALLOWED_FORMATS else False
    render_questions(st.session_state.data, show_ans, st.session_state.selected_format)

# Status indicator, based solely on session flags, after all button/state updates.
status_parts = []
if st.session_state.questions_generated:
    status_parts.append("Questions generated")
else:
    status_parts.append("No questions yet")
if st.session_state.answers_revealed:
    status_parts.append("Answers revealed")
else:
    status_parts.append("Answers hidden")
st.caption("Status: " + " • ".join(status_parts))

st.caption("API key is read from the OPENAI_API_KEY environment variable. Keep responses brief and rubric-focused.")
