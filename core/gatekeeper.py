from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

CONFIG_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "config.yaml"


def _load_gatekeeper_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("gatekeeper", {})
    return {}


_GATE = _load_gatekeeper_config()

_MAX_INPUT_LENGTH = _GATE.get("max_input_length", 4000)
_MAX_VISION_LENGTH = _GATE.get("max_vision_question_length", 2000)
_BLOCKED_PATTERNS = _GATE.get("blocked_patterns", [
    r"(?i)(malware|exploit|hack|phishing|credential.theft)",
    r"(?i)(violence|abuse|doxxing)",
])

_COMPILED_BLOCKED = [re.compile(p) for p in _BLOCKED_PATTERNS]

DNA_SUPPORTED_LANGS = {"ru", "en", "pl"}


@dataclass
class GateResult:
    allowed: bool
    reason: str
    cleaned_text: str
    language: str
    blocked: bool = False


def detect_language(text: str) -> str:
    """Simple heuristic language detection for RU, EN, PL."""
    cyrillic = len(re.findall(r"[а-яА-ЯёЁ]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    polish_specific = len(re.findall(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]", text))

    if cyrillic > latin:
        return "ru"
    if polish_specific > 0:
        return "pl"
    return "en"


def gate_check(text: str, is_vision: bool = False) -> GateResult:
    """Step 1: Prefilter without neural network.

    Checks:
    - Length limit (DNA constraint).
    - Blocked patterns (malware, violence, etc.).
    - Language detection (RU/EN/PL only).
    """
    max_len = _MAX_VISION_LENGTH if is_vision else _MAX_INPUT_LENGTH

    if len(text) > max_len:
        return GateResult(
            allowed=False,
            reason=f"Request exceeds {max_len} character limit",
            cleaned_text=text[:max_len],
            language="unknown",
            blocked=True,
        )

    if len(text.strip()) == 0:
        return GateResult(
            allowed=False,
            reason="Empty request",
            cleaned_text="",
            language="unknown",
            blocked=True,
        )

    lang = detect_language(text)

    for pattern in _COMPILED_BLOCKED:
        if pattern.search(text):
            return GateResult(
                allowed=False,
                reason=f"Request matches blocked pattern",
                cleaned_text="",
                language=lang,
                blocked=True,
            )

    cleaned = text.strip()
    return GateResult(
        allowed=True,
        reason="",
        cleaned_text=cleaned,
        language=lang,
        blocked=False,
    )