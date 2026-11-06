"""The live tests' English tool copy (arms A/B of the leak experiment) equals production.

German descriptions were not adopted (docs/experiments/reasoning-leak.md), so
tests/live/english_tools.py and praxis_telefon/agent/receptionist.py must send the same
tool schemas, descriptions included; otherwise LIVE_TOOL_DOCS=en would measure
something other than the shipped agent.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import llm
from livekit.agents.llm.utils import build_legacy_openai_schema

from praxis_telefon.agent.receptionist import Receptionist
from praxis_telefon.agent.tool_service import ToolService
from praxis_telefon.db.schema import connect
from tests.conftest import NOW
from tests.live.english_tools import EnglishToolDocsReceptionist


def _schemas(agent: type[Receptionist]) -> dict[str, dict[str, Any]]:
    tools = agent(ToolService(connect(":memory:"), lambda: NOW)).tools
    return {
        t.info.name: build_legacy_openai_schema(t)["function"]
        for t in tools
        if isinstance(t, llm.FunctionTool)
    }


def test_english_copy_sends_the_production_tool_schemas() -> None:
    production = _schemas(Receptionist)
    assert len(production) == 6
    assert _schemas(EnglishToolDocsReceptionist) == production
