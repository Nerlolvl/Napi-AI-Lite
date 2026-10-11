"""Helper functions for the chat pipeline (filter step, context flattening)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.prompt_builder import FILTER_SYSTEM_PROMPT
from soft_learning.rule_extractor import categorize_rule, is_usable_rule

from app.config import FILTER_MODEL, PROVIDER_CFG, TEMPS
from app.state import engine

log = logging.getLogger("napi")

_REFLECTED_RULE_RE = re.compile(r"\[REFLECTED_RULE:\s*([^\]]+)\]")


def helper_model_ready(model: str) -> bool:
    """A helper (teacher/filter/reasoning) is usable only with a real provider."""
    return bool(PROVIDER_CFG.get("base_url") and model)


async def model_filter(text: str) -> dict | None:
    """Run the filter model and parse its JSON payload; None on any failure."""
    try:
        raw = await engine.chat(
            messages=[
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            model=FILTER_MODEL,
            temperature=TEMPS.get("filter", 0.0),
            max_tokens=500,
        )
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
    except Exception as exc:
        log.warning("Model filter failed: %s", exc)
    return None


def rules_from_filter(filter_data: dict) -> list[dict[str, str]]:
    """Extract quality-checked reflected rules from a filter memory_note."""
    entries = []
    note = filter_data.get("memory_note", "")
    if note:
        rules = [r for r in _REFLECTED_RULE_RE.findall(note) if is_usable_rule(r)]
        for rule in rules:
            entries.append({"rule": rule, "category": categorize_rule(rule), "source": "filter"})
        if not rules and is_usable_rule(note):
            entries.append({"rule": note.strip(), "category": categorize_rule(note), "source": "filter"})
    return entries


def build_learning_context(
    knowledge_chunks: list[dict[str, str]],
    reflected_rules: list[str],
) -> str | None:
    """Compact knowledge + rules text for grounded evaluation and revision."""
    parts = []
    if knowledge_chunks:
        kb = "\n".join(
            f"- {chunk['title']}: {chunk['content'][:200]}" for chunk in knowledge_chunks
        )
        parts.append(f"Knowledge base excerpts used:\n{kb}")
    if reflected_rules:
        rules = "\n".join(f"- {rule}" for rule in reflected_rules)
        parts.append(f"Reflected rules applied:\n{rules}")
    return "\n\n".join(parts) or None


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Flatten a chat message list to plain text for the reasoning model."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"[{role}]: {content}")
    return "\n".join(parts)
