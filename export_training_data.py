"""Export teacher-approved dialogues into a fine-tune dataset (JSONL).

Pairs (user question, assistant answer) are read from the messages table.
Only answers whose metadata score meets --min-score — the score assigned by
the soft-learning teacher — are exported, so the dataset contains only the
lessons Napi got right. This is the first step of distillation: the dataset
feeds train_distill.py, whose fine-tuned model replaces the external
chat provider over time.

Output is OpenAI-style chat JSONL, accepted by most SFT tools:
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.prompt_builder import NAPI_SYSTEM_PROMPT
from storage.db_manager import NapiBrain

DEFAULT_DB_PATH = "./storage/napi_brain.db"
DEFAULT_OUTPUT = "./data/distill_dataset.jsonl"
DEFAULT_MIN_SCORE = 8.0
MIN_ANSWER_CHARS = 20


def load_pairs(
    rows: Iterable[Any],
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[tuple[str, str]]:
    """Pair adjacent user/assistant messages, keeping high-score answers.

    Rules:
    - a user message is kept as pending until an assistant reply arrives;
    - [vision] questions are skipped (image content cannot be distilled);
    - answers without a teacher score, or below min_score, are dropped;
    - trivially short answers are dropped;
    - duplicate questions (case-insensitive) are deduplicated, first wins.
    """
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    seen_questions: set[str] = set()

    for row in rows:
        role = row["role"]
        content = (row["content"] or "").strip()

        if role == "user":
            pending_user = None if content.startswith("[vision]") else content
            continue
        if role != "assistant" or pending_user is None:
            continue

        metadata = json.loads(row["metadata"] or "{}")
        score = metadata.get("score")
        answer_ok = (
            score is not None
            and float(score) >= min_score
            and len(content) >= MIN_ANSWER_CHARS
        )

        if answer_ok and pending_user.lower() not in seen_questions:
            seen_questions.add(pending_user.lower())
            pairs.append((pending_user, content))

        pending_user = None

    return pairs


def build_example(system: str, user: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def export_dataset(
    rows: Sequence[Any],
    output_path: str,
    min_score: float = DEFAULT_MIN_SCORE,
    system_prompt: str = NAPI_SYSTEM_PROMPT,
) -> int:
    """Write pairs as JSONL and return how many examples were written."""
    pairs = load_pairs(rows, min_score=min_score)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for user, assistant in pairs:
            example = build_example(system_prompt, user, assistant)
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(pairs)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Export teacher-approved dialogues into a fine-tune dataset."
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        return 1

    brain = NapiBrain(args.db)
    count = export_dataset(brain.all_messages(), args.output, min_score=args.min_score)
    brain.close()

    print(f"Exported examples: {count}")
    print(f"Dataset: {args.output}")
    print("Next step: python train_distill.py --dataset", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
