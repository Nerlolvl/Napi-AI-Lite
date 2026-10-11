# ⚡ Napi AI Lite

A local AI system with a **Stateless Architecture** — designed for 2 CPU / 2 GB RAM / ~100 GB SSD with room to scale.

Works with **any OpenAI-compatible provider**: LM Studio, Ollama, vLLM, Together AI, Groq, OpenAI, and others.

---

## 📂 Project Structure

```
napi_ai/
│
├── app/                              # 🌐 API layer (FastAPI)
│   ├── __init__.py
│   ├── config.py                     # ⚙️ Settings from config.yaml (single parse, shared with core/*)
│   ├── state.py                      # 🔁 Singletons: DB, inference engine, neural brain + GC counter
│   ├── schemas.py                    # 📋 Pydantic request/response models
│   ├── services.py                   # 🔧 Pipeline helpers (filter, rules, context text)
│   └── routers/                      # 🚏 Endpoints
│       ├── chat.py                   # POST /chat, /vision, /feedback
│       ├── knowledge.py              # /knowledge/add, /ingest, /search
│       └── system.py                 # GET /health
│
├── core/                             # 🧠 Local compute core
│   ├── __init__.py
│   ├── engine.py                     # Inference: GGUF + OpenAI-compatible provider (persistent httpx client)
│   ├── gatekeeper.py                 # Prefilter (protection against memory exhaustion)
│   ├── neural_brain.py               # Own neural brain (NumPy skip-gram, float16)
│   ├── prompt_builder.py             # Final prompt assembly (DNA + Tags + Query)
│   └── self_brain.py                 # Autonomous brain without an LLM (keys + memory + FNV-1a hash)
│
├── storage/                          # 💽 External brain (SQLite WAL, persistent connection)
│   ├── __init__.py
│   ├── db_manager.py                 # SQLite CRUD (single connection, FTS5)
│   └── napi_brain.db                 # SQLite DB (tag, rule, and knowledge storage)
│
├── soft_learning/                    # 🎓 Asynchronous "soft learning" module
│   ├── __init__.py
│   ├── teacher_api.py                # Teacher API client (any provider)
│   └── rule_extractor.py             # Parses Teacher critique into short rules
│
├── models/                           # 📦 Weights directory
│   ├── napi_neural_brain.npz         # NumPy embeddings (float16, ~2400 tokens)
│   └── README.txt                    # Instructions for loading a GGUF model
│
├── knowledge/                        # 📚 Knowledge base (markdown/txt for RAG)
│   └── *.md                          # Topic documents (core, personality, style, RAG, etc.)
│
├── config.yaml                       # ⚙️ Configuration (limits, languages, Teacher settings)
├── config_loader.py                  # 📋 The only config.yaml parser (used by all code)
├── main.py                           # 🚀 Entry point: FastAPI app assembly + graceful shutdown
├── chat_terminal.py                  # 💬 Terminal chat client
├── ingest_knowledge.py               # 📥 Loads knowledge/ into SQLite
├── export_training_data.py           # 📤 Exports teacher-approved dialogues to a fine-tune dataset
├── train_distill.py                  # 🎓 LoRA fine-tuning on the distilled dataset (GPU)
├── train_neural_brain.py             # 🧠 Neural weight training (local tool, git-ignored)
├── check_brain.py                    # 🔍 DB diagnostics (local tool, git-ignored)
├── data/                             # 📊 Exported datasets / adapters (git-ignored)
├── start_napi_chat.bat               # 🪟 Quick launch on Windows
├── tests/                            # 🧪 pytest suite (rule hygiene, learning, distillation)
├── requirements.txt                  # 📚 Dependencies
├── requirements-dev.txt              # 🧪 Dev dependencies (pytest)
└── README.md                         # 📖 Documentation
```

---

## ⚙️ How the System Works (Request Lifecycle)

Every request passes through **5 strict stages** that protect the system from OOM (Out of Memory):

### Step 1: Prefiltering (Gatekeeper)
The user sends a request in RU, EN, or PL.
`core/gatekeeper.py` checks string length, language, and blocking patterns.
If the request exceeds the limit (4000 characters) or contains forbidden content, it is rejected instantly — **without touching the neural network**.

### Step 2: Context Assembly (Prompt Builder)
`core/prompt_builder.py` takes the request and assembles the prompt:
1. **DNA** — the system prompt `[SYSTEM_PROMPT_NAPI_CORE_V1]`
2. **Knowledge** — chunks from `napi_brain.db` (FTS5 keyword search)
3. **Experience** — rules from the Reflection Diary (`[REFLECTED_RULE: ...]`)
4. Glues everything into the final prompt

### Step 3: Thinking and Generation (Inference)
`core/engine.py` passes the prompt to the engine:
- Local GGUF model (if enabled in `config.yaml`)
- Remote provider via the OpenAI-compatible API
- Napi first generates an invisible `<THINK>...</THINK>` block
- Then the final text answer is generated
- If generation reaches `max_visible_tokens` (512 by default), the process is stopped

### Step 4: Persistence and Cleanup (Stateless Drop)
As soon as the answer is delivered to the user:
- The `<THINK>` block is stripped from the visible answer
- The message and notes are saved to SQLite (persistent connection)
- GC runs once every 50 requests, not on every answer
- Napi **forgets** the request context. Memory is free again

### Step 5: Background Soft Learning
If the Teacher is active in `config.yaml`:
- The Teacher evaluates the answer **with full context**: the user question, the answer, the knowledge excerpts Napi used, and the rules it applied — so critique is grounded, not blind
- If the score is below `min_score_for_revision` (default 8), the answer is rewritten with the critique **and** the same knowledge context, keeping the revision consistent with the knowledge base
- Lessons are stored as `[REFLECTED_RULE: ...]` in `reflection_diary` — only from the teacher's `memory_note` channel, and only after a quality gate (length bounds, no generic filler)
- When an answer scores well, the rules that shaped it get a score bonus (`reinforce_bonus`) — rules that correlate with good answers grow stronger

**Rule hygiene** keeps the diary useful over time:
- **Dedup** — equivalent rules (case-insensitive) reinforce score (capped at 3.0) instead of duplicating
- **Score & usage** — selection orders by score and recency; each use bumps `hits` and refreshes `last_used_at`
- **Reinforcement** — good answers strengthen their rules; the teacher repeating a rule strengthens it too
- **TTL** — rules unused for `rule_ttl_days` (default 30) stop appearing in prompts
- **Hard cap** — the table never exceeds `max_rules_total` (default 200); best-scored rules survive

---

## 🛡️ Optimizations for 2 CPU / 2 GB RAM

| Mechanism | Description |
|-----------|-------------|
| Persistent SQLite | One connection instead of a new one per request; WAL + mmap |
| Persistent HTTP | Single httpx.AsyncClient with connection pooling and keepalive |
| float16 embeddings | Neural weights in half precision (×2 RAM savings) |
| Vectorized training | NumPy batches instead of Python loops (×10–100 speedup) |
| FNV-1a hash | Fast hash in self_brain instead of hashlib.sha256 |
| Precompiled regexes | All regexes compiled at import, not per call |
| GC every 50 requests | Instead of gc.collect() on every answer |
| Single config | One YAML parse via config_loader.py |
| Graceful shutdown | DB and HTTP client closed correctly on exit |
| GGUF quantization | 4-bit or 3-bit (local model ~1.5 GB) |
| Semaphore | Max 2 parallel requests to the provider |
| Gatekeeper | Prefiltering without a model call |
| RAG via FTS5 | Only relevant chunks are loaded into memory |
| Rule hygiene | Reflected rules: dedup, score, usage tracking, TTL, hard cap — the diary never bloats |

---

## 🔌 Supported Providers

Napi works with **any OpenAI-compatible API**. In `config.yaml`, set `base_url` and the models in the `engine.provider` section:

| Provider | base_url | Notes |
|----------|----------|-------|
| **LM Studio** | `http://localhost:1234/v1` | Local server on your PC |
| **Ollama** | `http://localhost:11434/v1` | Local inference |
| **vLLM** | `http://localhost:8000/v1` | Fast local server |
| **Together AI** | `https://api.together.xyz/v1` | Cloud inference |
| **Groq** | `https://api.groq.com/openai/v1` | Fast cloud inference |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` / `deepseek-reasoner` |
| **OpenAI** | `https://api.openai.com/v1` | The original OpenAI API |
| **Any other** | `https://your-server/v1` | Any server with `/chat/completions` |

The API key is set via an environment variable (default `NAPI_API_KEY`) or directly in `config.yaml`.
The project loads a `.env` file from the repo root automatically (git-ignored), so the
recommended setup is `.env` containing `NAPI_API_KEY=sk-...`.
For basic chat you only need the primary model: a local GGUF in `engine.local` or `provider.base_url` + `provider.chat_model`.
`teacher_model`, `filter_model`, and `reasoning_model` are optional modules — Napi still answers without them:

```yaml
engine:
  provider:
    api_key_env: "NAPI_API_KEY"
    base_url: "http://localhost:1234/v1"
    chat_model: "my-local-model"          # Napi primary model
    teacher_model: "my-local-model"       # optional: mentor
    filter_model: "my-local-model"        # optional: smart filtering
    reasoning_model: "my-local-model"     # optional: internal analysis
    vision_model: "my-vision-model"       # if the provider supports vision
```

A local GGUF and a remote provider can work **at the same time**: GGUF as primary, the provider as fallback.

### Autonomous Mode (No External LLM)

If neither GGUF nor a provider is configured, Napi runs on its own local brain, `napi-self-brain`.
It uses no Teacher and no external model: it understands the request through key meanings, searches knowledge in SQLite,
extracts relevant fragments, takes memory notes into account, and assembles the answer locally.

If `models/napi_neural_brain.npz` is trained, the local brain runs as `napi-neural-brain`.
This is Napi's own neural layer: NumPy skip-gram embeddings trained on `knowledge/`, memory, and history.

Train/retrain the weights:
```powershell
python train_neural_brain.py
```

---

## 🧪 Distillation (Teaching Napi to Stand on Its Own)

Soft learning turns experience into **rules and knowledge** (prompt context). Distillation
goes further: it turns teacher-approved dialogues into a **fine-tuned local model**, so Napi
needs the external provider less and less. The external model becomes an examiner, not the brain.

The loop:

1. **Chat normally** — every answer is scored by the Teacher (`evaluation_score`).
2. **Export the good ones** into a fine-tune dataset:
   ```powershell
   python export_training_data.py --min-score 8
   # -> ./data/distill_dataset.jsonl (OpenAI-style chat format)
   ```
   Only answers that scored at least `--min-score` are exported; questions are deduplicated.
3. **Fine-tune a small local model** on the dataset. Do this on a GPU machine or Colab/Kaggle —
   the 2 CPU / 2 GB box is for inference, not training:
   ```powershell
   pip install torch transformers peft datasets
   python train_distill.py --dataset ./data/distill_dataset.jsonl
   ```
4. **Convert to GGUF** and point Napi at it (instructions are printed by `train_distill.py`),
   then flip `engine.local.enabled: true` in `config.yaml`.

As the dataset and the rule diary grow, more traffic is served by the local model and the
provider is only needed for new, harder questions.

> Honest note: rules and knowledge are prompt context, not weight updates — they help any model.
> True independence from the external AI comes from this distillation step, where the local model
> actually learns the behavior from teacher-approved answers.

### Escalation (local first, provider backup)

When both a local GGUF model and a provider are configured, Napi answers **local-first**:

1. The local model produces the answer.
2. The Teacher scores it.
3. If the score is **below** `escalation.min_score` (default 7) — the local model did **not know** —
   the same question is re-asked to the provider, re-evaluated, and the better answer is returned.
4. The provider answer is stored with `source: provider` / `escalated: true` and, when it scores well,
   flows straight into the distillation dataset.

This is the self-teaching loop: Napi tries on its own, asks the big model only when unsure, and learns
that answer for next time. With no local model loaded, escalation stays inactive and every request is
served by the provider as before — the behaviour above only activates once you add a GGUF model.

---

## 📦 Dependencies

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
httpx==0.28.1
pydantic==2.10.4
pydantic-settings==2.7.1
python-dotenv==1.0.1
pyyaml==6.0.2
numpy>=2.0.0
```

Optional (for a local GGUF model):
```
llama-cpp-python
```

---

## 🚀 Getting Started

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure config.yaml (base_url, models, API key)

# 3. Set the API key (if needed)
$env:NAPI_API_KEY = "your-key"

# 4. Run the server
python main.py
```

Or via uvicorn:
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Terminal client:
```powershell
python chat_terminal.py
# or
start_napi_chat.bat
```

Load knowledge:
```powershell
python ingest_knowledge.py
python ingest_knowledge.py "C:\path\to\my\docs"
```

Run tests:
```powershell
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

---

## ⚙️ Configuration (config.yaml)

All settings live in a single file:

| Section | What it configures |
|---------|--------------------|
| `dna` | Limits: 4000 chars, 512 tokens, RU/EN/PL languages |
| `engine.local` | GGUF model: path, n_ctx, n_threads |
| `engine.provider` | **Provider**: base_url, models (REQUIRED), timeout, concurrency |
| `gatekeeper` | Prefiltering: length, blocking patterns |
| `storage` | DB path, history/notes/chunks limits, rule hygiene (`rule_ttl_days`, `max_rules_total`) |
| `soft_learning` | Teacher: score threshold, reinforcement bonus, temperatures |
| `reasoning` | THINK block: enable, max_context, max_tokens |
| `escalation` | Local-first routing: enable, `min_score` to escalate to the provider |
| `server` | Host, port, workers |

---

## 🔌 API

### `POST /chat`
```json
{
  "message": "Hi, Napi!",
  "session_id": "default",
  "language": "auto",
  "self_improve": true
}
```

### `POST /vision`
```json
{
  "question": "Rate the UI of this screen",
  "image_url": "https://example.com/image.png",
  "session_id": "default",
  "language": "auto"
}
```

### `POST /feedback`
```json
{
  "session_id": "default",
  "message_id": 1,
  "rating": 1,
  "comment": "The answer was helpful"
}
```

### `POST /knowledge/add`
Add a single document to the knowledge base:
```json
{
  "source": "manual://my-topic",
  "title": "My Topic",
  "content": "Knowledge text that Napi should use in its answers."
}
```

### `POST /knowledge/ingest`
Ingest all `.md` and `.txt` files from a folder:
```json
{
  "directory": "./knowledge"
}
```

### `GET /knowledge/search`
Verify that knowledge is retrievable:
```text
/knowledge/search?q=RAG&limit=3
```

### `GET /health`
```json
{
  "status": "ok",
  "name": "Napi",
  "local_model_loaded": false,
  "neural_brain_loaded": true,
  "neural_vocab_size": 2400,
  "can_chat": true
}
```

---

## 📐 Scaling

| Profile | Concurrency | History | Chunks | Model |
|---------|-------------|---------|--------|-------|
| 2 CPU / 2 GB | 2 | 24 | 6 | GGUF 2B or provider |
| 4 CPU / 4 GB | 4 | 40 | 8 | GGUF 2B + provider |
| 8 CPU / 8 GB | 8 | 60 | 10 | GGUF 7B + provider |
| 16+ CPU / 16+ GB | 16 | 100 | 12 | GGUF 13B + provider |

All limits are changed in `config.yaml` — **no code changes required**.
