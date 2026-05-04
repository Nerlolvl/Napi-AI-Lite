# Napi Self Brain

Napi Self Brain is the local reasoning layer used when no GGUF model or remote
provider is configured. It is not a scripted FAQ. It runs a small reasoning
cycle:

1. Detect the user's conversational mode.
2. Extract keywords and topic signals.
3. Retrieve relevant knowledge from SQLite FTS5.
4. Inspect memory notes and reflected rules.
5. Select evidence sentences from the retrieved chunks.
6. Compose an answer from evidence, memory, and the current request.
7. Save useful memory notes when the user explicitly says "запомни".

Self Brain is fully local and belongs to Napi. It does not call a teacher model,
OpenAI-compatible API, Ollama, LM Studio, or any other provider.

Limits:
- It has no neural weights.
- It cannot match a full LLM in abstract reasoning or free-form generation.
- Its intelligence grows mostly from better documents, examples, notes, and code.

Good use:
- basic conversation;
- identity and capability questions;
- answers grounded in Napi's knowledge base;
- local memory instructions;
- project-specific reasoning where the needed facts are stored in `knowledge/`.

To make Self Brain smarter, add high-quality `.md` or `.txt` files to
`knowledge/`, then run:

```powershell
python ingest_knowledge.py
```

