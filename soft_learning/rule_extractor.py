from __future__ import annotations

import re
from typing import Any

from soft_learning.teacher_api import Evaluation


def extract_rules_from_evaluation(evaluation: Evaluation) -> list[str]:
    """Extract [REFLECTED_RULE: ...] from the teacher's memory_note field."""
    note = evaluation.memory_note or ""
    if not note:
        return []

    rules = re.findall(r"\[REFLECTED_RULE:\s*([^\]]+)\]", note)
    if rules:
        return rules

    if note.strip():
        return [note.strip()]

    return []


def extract_rules_from_critique(evaluation: Evaluation) -> list[str]:
    """Derive short rules from critique and revision instructions."""
    rules: list[str] = []

    critique = evaluation.critique.strip()
    revision = evaluation.revision_instructions.strip()

    if critique:
        short = critique.split(".")[0].strip()
        if short and len(short) < 200:
            rules.append(short)

    if revision and revision != critique:
        short = revision.split(".")[0].strip()
        if short and len(short) < 200:
            rules.append(short)

    return rules


def categorize_rule(rule: str) -> str:
    """Assign a category tag to a reflected rule for organized storage."""
    rule_lower = rule.lower()

    categories = {
        "language": ["язык", "language", "lang", "ru", "en", "pl", "перевод", "translate"],
        "safety": ["безопасн", "safety", "filter", "фильтр", "harm", "malicious"],
        "accuracy": ["fact", "факт", "точност", "accuracy", "ошибк", "incorrect", "wrong"],
        "conciseness": ["кратк", "concise", "short", "длин", "verbose", "вод", "water"],
        "design": ["ui", "ux", "дизайн", "design", "figma", "layout", "иерарх", "hierarchy"],
        "code": ["python", "код", "code", "function", "class", "import", "api"],
        "media": ["аниме", "anime", "кино", "cinema", "фильм", "movie", "дорам", "drama"],
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in rule_lower:
                return category

    return "general"


def process_evaluation(evaluation: Evaluation) -> list[dict[str, str]]:
    """Full pipeline: extract rules, categorize, return structured entries."""
    entries: list[dict[str, str]] = []

    for rule in extract_rules_from_evaluation(evaluation):
        entries.append({
            "rule": rule,
            "category": categorize_rule(rule),
            "source": "teacher_note",
        })

    for rule in extract_rules_from_critique(evaluation):
        entries.append({
            "rule": rule,
            "category": categorize_rule(rule),
            "source": "teacher_critique",
        })

    return entries