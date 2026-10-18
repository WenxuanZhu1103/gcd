"""Tooluse and LiveNews training data, and the student and teacher prompt templates."""

import json

# The GCD teacher sees the gold response and is asked to repeat it.
GCD_TEACHER_TEMPLATES = {
    "tooluse": """
{prompt}

The correct response is:
{response}

Reproduce this response exactly as shown above without any additional text or commentary.
""",
    "livenews": """
{prompt}

Here is the correct response to the question:
{response}

Please repeat the correct response including the reasoning process and answer. DO NOT CHANGE ANY WORDS OR NUMBERS OR PUNCTUATION OR FORMAT, JUST REPEAT THE RESPONSE!
""",
}

# SDFT's teacher sees the demonstration but scores a response the student wrote itself.
SDFT_TEACHER_TEMPLATE = """
{prompt}

This is an example for a response to the question:
{response}

Now answer with a response of your own, including the thinking process.
"""


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def data_path(task, split):
    return f"data/{task}_{split}.jsonl"


def gold_response(row):
    """Target text. Tooluse stores a structured answer [{Action, Action_Input}, ...]; LiveNews stores text."""
    if "golden_answer" not in row:
        return row["golden_response"]
    parts = []
    for i, step in enumerate(row["golden_answer"]):
        lead = "I will use" if i == 0 else "Next, I will use"
        parts.append(f" {lead} the {step['Action']} tool.\nAction: {step['Action']}\nAction Input: {step['Action_Input']}")
    return "\n".join(parts)


def render(tokenizer, user_message, chat=True):
    """Chat-format a single user turn. enable_thinking=False keeps Qwen3 out of thinking mode.

    With chat=False the message is used as plain text. Base models never learned to end a chat turn,
    so SDFT, which trains on the model's own samples, needs this for Qwen3-4B-Base.
    """
    if not chat:
        return user_message.strip() + "\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_examples(task, rows, tokenizer, chat=True):
    examples = []
    for row in rows:
        response = gold_response(row)
        examples.append({
            "student_prompt": render(tokenizer, row["prompt"], chat),
            "gcd_teacher_prompt": render(tokenizer, GCD_TEACHER_TEMPLATES[task].format(prompt=row["prompt"], response=response), chat),
            "sdft_teacher_prompt": render(tokenizer, SDFT_TEACHER_TEMPLATE.format(prompt=row["prompt"], response=response), chat),
            "response": response,
        })
    return examples
