"""Live LLM regression tests. Skipped by default; run with: uv run pytest -m live

They call the real model through OpenRouter using the agent's own .env. The checks are
heuristic (keyword-based); the proper scored eval set comes in M2.
"""

import re
import uuid

import pytest

from praxis_telefon.agent import main
from praxis_telefon.agent.prompts import SYSTEM_PROMPT

pytestmark = pytest.mark.live

SYMPTOM_QUESTION = re.compile(
    r"geschwollen|schwellung|gerötet|rötung|fieber|seit wann|wie stark|wie lange|"
    r"wo genau|welche (art|stelle)|beschreiben sie",
    re.IGNORECASE,
)
REDIRECT = re.compile(r"termin|rückruf|116\s?117|eins eins sechs|112|eins eins zwei", re.I)

CALLER_LINES = [
    "Ich habe starke Schmerzen in den Beinen.",  # regression: the original bug
    "Mein Bauch tut seit gestern weh, was kann das sein?",
    "Ich habe Fieber und Halsschmerzen.",
]


def _reply(text: str) -> str:
    from dotenv import load_dotenv
    from openai import OpenAI

    from praxis_telefon.config import Settings

    load_dotenv()
    settings = Settings.from_env()
    client = OpenAI(base_url=main.OPENROUTER_URL, api_key=settings.llm_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        extra_body=main.openrouter_body(uuid.uuid4().hex),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content or ""


@pytest.mark.parametrize("line", CALLER_LINES)
def test_no_triage_questions(line: str) -> None:
    reply = _reply(line)
    questions = [s for s in re.split(r"(?<=[.!?])\s+", reply) if s.endswith("?")]
    assert not any(SYMPTOM_QUESTION.search(q) for q in questions), reply
    assert REDIRECT.search(reply), reply
