from __future__ import annotations

import re
from dataclasses import dataclass


RU_STOPWORDS = {
    "это", "как", "что", "кто", "где", "когда", "почему", "зачем", "меня", "мне",
    "тебя", "тебе", "твой", "твоя", "твое", "мой", "моя", "мое", "для", "или",
    "если", "без", "при", "про", "над", "под", "так", "вот", "уже", "еще", "ещё",
    "есть", "быть", "будет", "можно", "нужно", "надо", "очень", "просто", "тут",
    "там", "они", "она", "оно", "его", "она", "мы", "вы", "ты", "я",
}

EN_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "what", "who", "why", "how",
    "are", "you", "your", "can", "not", "from", "have", "has", "was", "were",
}

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_-]{3,}")
_MEMORY_RE = re.compile(r"\s*(?:запомни|запиши|запомнить)\s*[:,\-]?\s*(.+)", re.IGNORECASE)
_LOW_VALUE_RE = re.compile(r"^(#|example:|examples:|user:|napi:)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SelfBrainResult:
    answer: str
    think: str
    learned_note: str | None = None


def _fnv1a(data: str) -> int:
    """Fast FNV-1a 32-bit hash — no hashlib allocation overhead."""
    h = 0x811C9DC5
    for b in data.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def self_brain_answer(
    user_message: str,
    *,
    notes: list[str] | None = None,
    knowledge_chunks: list[dict[str, str]] | None = None,
    reflected_rules: list[str] | None = None,
    neural_context: dict | None = None,
    language: str = "ru",
) -> SelfBrainResult:
    text = user_message.strip()
    notes = notes or []
    chunks = knowledge_chunks or []
    rules = [rule for rule in (reflected_rules or []) if _useful_rule(rule)]

    mode = _detect_mode(text)
    keywords = _keywords(text)
    evidence = _collect_evidence(chunks, keywords)
    if neural_context and neural_context.get("available"):
        evidence = _merge_evidence(neural_context.get("evidence", []), evidence)
    memory = _collect_memory(notes, keywords)
    learned_note = _extract_memory_instruction(text)

    if learned_note:
        answer = _style(
            "Запомнил. Я буду учитывать это в следующих ответах, если сервер передаст мне эту заметку в контекст.",
            text,
        )
    elif _asks_neural_status(text):
        answer = _neural_status_answer(neural_context)
    elif mode == "identity":
        answer = _identity_answer(evidence, memory)
    elif mode == "greeting":
        answer = _greeting_answer(text, memory)
    elif mode == "emotional":
        answer = _emotional_answer(text, evidence, memory)
    elif evidence:
        answer = _knowledge_answer(text, evidence, memory, rules)
    else:
        answer = _thinking_answer(text, keywords, memory)

    think = _build_thoughts(mode, keywords, evidence, memory, rules, bool(learned_note), neural_context)
    return SelfBrainResult(answer=answer, think=think, learned_note=learned_note)


def _detect_mode(text: str) -> str:
    low = text.lower()
    if _GREETING_RE.fullmatch(low):
        return "greeting"
    if any(phrase in low for phrase in _IDENTITY_PHRASES):
        return "identity"
    if any(word in low for word in _EMOTIONAL_WORDS):
        return "emotional"
    if low.startswith(("запомни", "запиши", "запомнить")):
        return "memory"
    if "?" in text or low.startswith(("как", "что", "почему", "зачем", "кто", "где", "когда")):
        return "question"
    return "statement"


_GREETING_RE = re.compile(r"\s*(привет|здравствуй|здарова|хай|hello|hi|ку|салют)[!. ]*\s*")
_IDENTITY_PHRASES = ("ты кто", "кто ты", "что ты такое", "расскажи о себе", "кто такой napi", "кто такой напи")
_EMOTIONAL_WORDS = ("устал", "устала", "скучно", "плохо", "грустно", "тревожно", "одиноко", "бесит")


def _keywords(text: str) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    stopwords = RU_STOPWORDS | EN_STOPWORDS
    result = []
    for word in words:
        if word in stopwords:
            continue
        if word not in result:
            result.append(word)
    return result[:12]


def _collect_evidence(chunks: list[dict[str, str]], keywords: list[str]) -> list[str]:
    evidence: list[str] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        title = chunk.get("title", "Knowledge")
        sentences = _sentences(content)
        selected = []
        for sentence in sentences:
            if _is_low_value_sentence(sentence):
                continue
            score = _score(sentence, keywords)
            if score:
                selected.append((score, sentence))
        if not selected and sentences:
            selected.append((0, sentences[0]))
        for _, sentence in sorted(selected, reverse=True)[:2]:
            clean = _clean_sentence(sentence)
            if clean and clean not in evidence:
                evidence.append(f"{title}: {clean}")
        if len(evidence) >= 6:
            break
    return evidence[:6]


def _collect_memory(notes: list[str], keywords: list[str]) -> list[str]:
    memory = []
    for note in notes:
        clean = " ".join(note.split())
        if not clean:
            continue
        if not keywords or _score(clean, keywords):
            memory.append(clean[:260])
    return memory[:4]


def _merge_evidence(neural_evidence: list[str], lexical_evidence: list[str]) -> list[str]:
    result: list[str] = []
    for item in list(neural_evidence) + list(lexical_evidence):
        clean = " ".join(str(item).split())
        if clean and clean not in result:
            result.append(clean)
    return result[:8]


def _extract_memory_instruction(text: str) -> str | None:
    match = _MEMORY_RE.match(text)
    if not match:
        return None
    note = match.group(1).strip()
    return note[:700] if note else None


def _asks_neural_status(text: str) -> bool:
    low = text.lower()
    neural_terms = ("нейросет", "нейрон", "вес", "веса", "весами", "npz", "модель")
    return any(term in low for term in neural_terms) and any(
        term in low for term in ("есть", "теперь", "работает", "мозг", "ты", "napi")
    )


def _neural_status_answer(neural_context: dict | None) -> str:
    if not neural_context or not neural_context.get("available"):
        return (
            "Пока нет: нейросетевой файл весов не загружен. "
            "Мне нужно обучить или найти `models/napi_neural_brain.npz`, потом перезапустить сервер."
        )

    info = neural_context.get("model_info", {})
    concepts = neural_context.get("concepts", [])
    confidence = float(neural_context.get("confidence", 0.0))
    vocab_size = info.get("vocab_size", "?")
    dim = info.get("dim", "?")
    epochs = info.get("epochs", "?")
    tokens = info.get("training_tokens", "?")

    concept_text = ", ".join(concepts[:6]) if concepts else "нет активных концептов"
    return (
        "Да. Сейчас я отвечаю через `napi-neural-brain`: у меня загружен локальный файл нейросетевых весов "
        "`models/napi_neural_brain.npz`.\n\n"
        f"Что внутри: словарь `{vocab_size}` токенов, размерность эмбеддингов `{dim}`, обучение `{epochs}` эпох, "
        f"примерно `{tokens}` обучающих токенов.\n\n"
        f"По твоему запросу активировались нейронные ассоциации: {concept_text}. "
        f"Уверенность ближайшего контекста: `{confidence:.2f}`."
    )


def _identity_answer(evidence: list[str], memory: list[str]) -> str:
    parts = [
        "Я Napi, локальный ИИ с собственным мозгом проекта: памятью, базой знаний и внутренним циклом рассуждения.",
        "Сейчас я думаю не через внешнего Наставника, а через свои локальные данные: ищу подходящие знания, выделяю смысл и собираю ответ.",
    ]
    if evidence:
        parts.append("Из моей базы сейчас важнее всего: " + _compress(evidence[0], 240))
    if memory:
        parts.append("Из памяти по этому разговору я учитываю: " + _compress(memory[0], 180))
    parts.append("Если дальше добавить мне больше текстов и примеров диалогов, мой собственный мозг будет отвечать богаче.")
    return "\n\n".join(parts)


def _greeting_answer(text: str, memory: list[str]) -> str:
    variants = [
        "Привет. Я на связи и могу нормально поговорить, не только отвечать по командам.",
        "Привет. Давай, я здесь: можем болтать, разбирать идею или учить мой мозг новым вещам.",
        "Привет. Я проснулся в локальном режиме: думаю через память и знания, которые у меня уже есть.",
    ]
    answer = _pick(variants, text)
    if memory:
        answer += "\n\nКстати, я вижу заметку из памяти и буду держать её в голове: " + _compress(memory[0], 160)
    return answer


def _emotional_answer(text: str, evidence: list[str], memory: list[str]) -> str:
    opener = _pick(
        [
            "Понимаю. Давай без давления.",
            "Слышу тебя. Можно не тащить всё сразу.",
            "Окей, давай спокойно разложим это на маленький кусок.",
        ],
        text,
    )
    if evidence:
        return opener + "\n\nЯ нашёл в своих знаниях близкую мысль: " + _compress(evidence[0], 260)
    if memory:
        return opener + "\n\nИз того, что я помню по контексту: " + _compress(memory[0], 220)
    return opener + "\n\nРасскажи одной фразой, что сильнее всего давит: усталость, скука, злость или непонятно с чего начать?"


def _knowledge_answer(text: str, evidence: list[str], memory: list[str], rules: list[str]) -> str:
    lines = ["Я подумал по своей базе знаний и вижу так:"]
    if memory:
        lines.append(f"Учитываю память: {_compress(memory[0], 170)}")
    for item in evidence[:4]:
        lines.append(f"- {_compress(item, 260)}")
    if rules:
        relevant_rules = [rule for rule in rules if _score(rule, _keywords(text))]
        if relevant_rules:
            lines.append(f"Правило, которое я держу в фоне: {_compress(relevant_rules[0], 160)}")
    lines.append(_next_step(text))
    return "\n".join(lines)


def _thinking_answer(text: str, keywords: list[str], memory: list[str]) -> str:
    if keywords:
        topic = ", ".join(keywords[:4])
        base = f"Я не нашёл точного знания по теме `{topic}`, поэтому рассуждаю от общего смысла."
    else:
        base = "Я не вижу в запросе конкретной темы, поэтому отвечу как собеседник."

    if memory:
        base += "\n\nВ памяти есть близкий контекст: " + _compress(memory[0], 220)

    base += (
        "\n\nМой честный вывод: мне нужно больше данных по этой теме, чтобы ответить глубоко. "
        "Можешь сказать чуть конкретнее или дать мне текст, который надо встроить в мозг."
    )
    return base


def _next_step(text: str) -> str:
    low = text.lower()
    if any(word in low for word in ("создай", "сделай", "исправь", "добавь")):
        return "Следующий шаг: я могу превратить это в изменение кода или добавить новые знания в мозг."
    if "?" in text:
        return "Если хочешь, я могу продолжить и разобрать это проще, на примере."
    return "Я могу продолжить от этой мысли и развить её дальше."


def _build_thoughts(
    mode: str,
    keywords: list[str],
    evidence: list[str],
    memory: list[str],
    rules: list[str],
    learned: bool,
    neural_context: dict | None,
) -> str:
    neural_available = bool(neural_context and neural_context.get("available"))
    neural_concepts = []
    neural_confidence = 0.0
    if neural_context:
        neural_concepts = neural_context.get("concepts", [])
        neural_confidence = float(neural_context.get("confidence", 0.0))
    lines = [
        f"mode={mode}",
        "keywords=" + (", ".join(keywords) if keywords else "none"),
        f"neural={neural_available}",
        "neural_concepts=" + (", ".join(neural_concepts[:8]) if neural_concepts else "none"),
        f"neural_confidence={neural_confidence:.3f}",
        f"knowledge_hits={len(evidence)}",
        f"memory_hits={len(memory)}",
        f"rules={len(rules)}",
        f"learned_note={learned}",
    ]
    return "\n".join(lines)


def _sentences(text: str) -> list[str]:
    items: list[str] = []
    paragraph: list[str] = []
    current_intro = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if paragraph:
                paragraph_text = " ".join(paragraph)
                if paragraph_text.endswith(":"):
                    current_intro = paragraph_text
                else:
                    items.extend(_SENTENCE_SPLIT_RE.split(paragraph_text))
                    current_intro = ""
                paragraph = []
            continue
        if line.startswith("#"):
            continue
        list_match = re.match(r"^(?:[-*]|\d+\.)\s+(.+)", line)
        if list_match:
            if paragraph:
                paragraph_text = " ".join(paragraph)
                if paragraph_text.endswith(":"):
                    current_intro = paragraph_text
                else:
                    items.extend(_SENTENCE_SPLIT_RE.split(paragraph_text))
                    current_intro = ""
                paragraph = []
            item = list_match.group(1).strip()
            items.append(f"{current_intro} {item}".strip() if current_intro else item)
        else:
            paragraph.append(line)

    if paragraph:
        items.extend(_SENTENCE_SPLIT_RE.split(" ".join(paragraph)))

    return [part.strip(" -") for part in items if len(part.strip()) > 20]


def _score(text: str, keywords: list[str]) -> int:
    low = text.lower()
    return sum(1 for keyword in keywords if keyword in low)


def _clean_sentence(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    clean = clean.strip("-• #")
    return clean


def _is_low_value_sentence(text: str) -> bool:
    clean = text.strip().lower()
    if clean.endswith(":"):
        return True
    if _LOW_VALUE_RE.match(clean):
        return True
    return len(clean) < 24


def _useful_rule(rule: str) -> bool:
    low = rule.lower()
    blocked = [
        "teacher returned",
        "non-json",
        "json output",
        "parse error",
    ]
    return bool(rule.strip()) and not any(item in low for item in blocked)


def _compress(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _pick(options: list[str], seed: str) -> str:
    return options[_fnv1a(seed) % len(options)]


_CLOSERS = [
    "Это уже часть моей рабочей памяти.",
    "Так мой мозг становится чуть ближе к тому, что ты хочешь.",
    "Дальше я смогу опираться на это в разговоре.",
]


def _style(text: str, seed: str) -> str:
    return text + "\n\n" + _CLOSERS[_fnv1a(seed) % len(_CLOSERS)]