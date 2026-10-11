from __future__ import annotations

import re
from typing import Any

from soft_learning.teacher_api import Evaluation

MIN_RULE_LENGTH = 10
MAX_RULE_LENGTH = 200

# Rules that carry no actionable information.
_GENERIC_RULES = {
    "be better",
    "do better",
    "improve",
    "improve answer",
    "better answer",
    "answer correctly",
    "be accurate",
    "be useful",
}


def is_usable_rule(rule: str) -> bool:
    """Reject noise rules: wrong length, generic phrases, or empty filler."""
    clean = rule.strip()
    if not (MIN_RULE_LENGTH <= len(clean) <= MAX_RULE_LENGTH):
        return False
    if clean.lower() in _GENERIC_RULES:
        return False
    # A rule made of a single word cannot carry a lesson.
    if len(clean.split()) < 2:
        return False
    return True


def extract_rules_from_evaluation(evaluation: Evaluation) -> list[str]:
    """Extract [REFLECTED_RULE: ...] from the teacher's memory_note field."""
    note = evaluation.memory_note or ""
    if not note:
        return []

    rules = [r for r in re.findall(r"\[REFLECTED_RULE:\s*([^\]]+)\]", note) if is_usable_rule(r)]
    if rules:
        return rules

    if note.strip() and is_usable_rule(note):
        return [note.strip()]

    return []


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
    """Full pipeline: extract quality-checked rules from the teacher's
    memory_note (the sanctioned lesson channel), categorize them."""
    entries: list[dict[str, str]] = []

    for rule in extract_rules_from_evaluation(evaluation):
        entries.append({
            "rule": rule,
            "category": categorize_rule(rule),
            "source": "teacher_note",
        })

    return entries