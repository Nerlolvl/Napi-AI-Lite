from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "napi_neural_brain.npz"

STOPWORDS = {
    "это", "как", "что", "кто", "где", "когда", "почему", "зачем", "меня", "мне",
    "тебя", "тебе", "твой", "твоя", "твоё", "мой", "моя", "моё", "для", "или",
    "если", "без", "при", "про", "над", "под", "так", "вот", "уже", "ещё", "еще",
    "есть", "быть", "будет", "можно", "нужно", "надо", "очень", "просто", "тут",
    "там", "они", "она", "оно", "его", "мы", "вы", "ты", "the", "and", "for",
    "with", "that", "this", "what", "who", "why", "how", "are", "you", "your",
}

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_-]{3,}")


@dataclass(frozen=True)
class NeuralThought:
    available: bool
    confidence: float
    concepts: list[str]
    evidence: list[str]
    model_info: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "confidence": self.confidence,
            "concepts": self.concepts,
            "evidence": self.evidence,
            "model_info": self.model_info,
        }


class NeuralBrain:
    """Tiny local neural semantic brain based on learned token embeddings.

    Uses float16 internally to cut memory usage in half (critical for 2 GB RAM).
    Lazy-loads on first .think() call instead of at server startup.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded and self.embeddings is not None and bool(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            data = np.load(self.model_path, allow_pickle=False)
            vocab = [str(x) for x in data["vocab"].tolist()]
            embeddings = data["embeddings"].astype(np.float16)
            metadata_raw = str(data["metadata"].tolist())
            self.vocab = vocab
            self.token_to_id = {token: i for i, token in enumerate(vocab)}
            self.embeddings = _normalize_rows(embeddings)
            self.metadata = json.loads(metadata_raw)
            self._loaded = True
            return True
        except Exception:
            self._loaded = False
            return False

    def think(
        self,
        query: str,
        *,
        knowledge_chunks: list[dict[str, str]] | None = None,
        notes: list[str] | None = None,
    ) -> NeuralThought:
        if not self.available or self.embeddings is None:
            return NeuralThought(False, 0.0, [], [], {})

        qvec = self.encode(query)
        if qvec is None:
            return NeuralThought(True, 0.0, [], [], self.metadata)

        concepts = self.nearest_tokens(qvec, limit=8)
        evidence = self.rank_evidence(qvec, knowledge_chunks or [], notes or [])
        confidence = evidence[0][0] if evidence else 0.0
        return NeuralThought(
            available=True,
            confidence=float(max(0.0, min(1.0, float(confidence)))),
            concepts=concepts,
            evidence=[item for _, item in evidence[:6]],
            model_info=self.metadata,
        )

    def encode(self, text: str) -> np.ndarray | None:
        if self.embeddings is None:
            return None
        ids = [self.token_to_id[token] for token in _tokenize(text) if token in self.token_to_id]
        if not ids:
            return None
        vec = self.embeddings[ids].mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return None
        return vec / norm

    def nearest_tokens(self, qvec: np.ndarray, limit: int = 8) -> list[str]:
        if self.embeddings is None:
            return []
        scores = self.embeddings.astype(np.float32) @ qvec
        order = np.argsort(-scores)
        result = []
        for idx in order:
            token = self.vocab[int(idx)]
            if token in STOPWORDS or len(token) < 3:
                continue
            result.append(token)
            if len(result) >= limit:
                break
        return result

    def rank_evidence(
        self,
        qvec: np.ndarray,
        knowledge_chunks: list[dict[str, str]],
        notes: list[str],
    ) -> list[tuple[float, str]]:
        candidates: list[tuple[str, str]] = []
        for chunk in knowledge_chunks:
            title = chunk.get("title", "Knowledge")
            for sentence in split_sentences(chunk.get("content", "")):
                candidates.append((title, sentence))
        for note in notes:
            candidates.append(("Memory", note))

        if not candidates:
            return []

        texts = [sent for _, sent in candidates]
        vecs: list[np.ndarray | None] = []
        for text in texts:
            vecs.append(self.encode(text))

        scored: list[tuple[float, str]] = []
        for i, ((title, sentence), svec) in enumerate(zip(candidates, vecs)):
            if svec is None:
                continue
            score = float(np.dot(qvec, svec))
            if score > 0.12:
                clean = " ".join(sentence.split())
                scored.append((score, f"{title}: {clean}"))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    items: list[str] = []
    paragraph: list[str] = []
    intro = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if paragraph:
                joined = " ".join(paragraph)
                if joined.endswith(":"):
                    intro = joined
                else:
                    items.extend(_split_plain(joined))
                    intro = ""
                paragraph = []
            continue
        if line.startswith("#"):
            continue
        match = re.match(r"^(?:[-*]|\d+\.)\s+(.+)", line)
        if match:
            if paragraph:
                joined = " ".join(paragraph)
                if joined.endswith(":"):
                    intro = joined
                else:
                    items.extend(_split_plain(joined))
                    intro = ""
                paragraph = []
            body = match.group(1).strip()
            items.append(f"{intro} {body}".strip() if intro else body)
        else:
            paragraph.append(line)

    if paragraph:
        items.extend(_split_plain(" ".join(paragraph)))

    clean_items = []
    for item in items:
        clean = item.strip(" -#")
        if len(clean) >= 24 and not clean.endswith(":"):
            clean_items.append(clean)
    return clean_items


def _split_plain(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return matrix / norms