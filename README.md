# ⚡ Napi AI Lite

Локальная ИИ-система со **Stateless Architecture** — спроектирована для 2 CPU / 2 GB RAM / ~100 GB SSD с возможностью масштабирования.

Работает с **любым OpenAI-совместимым провайдером**: LM Studio, Ollama, vLLM, Together AI, Groq, OpenAI и другими.

---

## 📂 Структура проекта

```
napi_ai/
│
├── core/                           # 🧠 Локальное вычислительное ядро
│   ├── __init__.py
│   ├── engine.py                   # Инференс: GGUF + OpenAI-совместимый провайдер (постоянный httpx-клиент)
│   ├── gatekeeper.py               # Префильтр (защита от переполнения памяти)
│   ├── neural_brain.py             # Собственный нейросетевой мозг (NumPy skip-gram, float16)
│   ├── prompt_builder.py           # Сборка финального промпта (ДНК + Теги + Запрос)
│   └── self_brain.py               # Автономный мозг без LLM (ключи + память + FNV-1a хеш)
│
├── storage/                        # 💽 Внешний мозг (SQLite WAL, persistent connection)
│   ├── __init__.py
│   ├── db_manager.py               # CRUD-операции для SQLite (одиночное соединение, FTS5)
│   └── napi_brain.db               # SQLite БД (хранилище тегов, правил, знаний)
│
├── soft_learning/                  # 🎓 Модуль асинхронного "мягкого обучения"
│   ├── __init__.py
│   ├── teacher_api.py              # API-клиент Учителя (любой провайдер)
│   └── rule_extractor.py           # Парсер критики Учителя в короткие правила
│
├── models/                         # 📦 Директория для весов
│   ├── napi_neural_brain.npz       # NumPy-эмбеддинги (float16, ~2400 токенов)
│   └── README.txt                  # Инструкция по загрузке GGUF модели
│
├── knowledge/                      # 📚 База знаний (markdown/txt для RAG)
│   ├── napi_ai_core.md
│   ├── napi_self_brain_ru.md
│   ├── napi_personality_ru.md
│   ├── friend_like_boundaries_ru.md
│   ├── small_talk_ru.md
│   ├── conversation_memory_rules.md
│   ├── human_conversation_style_ru.md
│   ├── emotional_support_ru.md
│   ├── uiux_foundations.md
│   ├── prompting_and_answers.md
│   ├── local_llm_operations.md
│   └── rag_and_finetuning.md
│
├── config.yaml                     # ⚙️ Конфигурация (лимиты, языки, настройки Учителя)
├── config_loader.py                # 📋 Единая загрузка конфига (1 парсинг вместо 4)
├── main.py                         # 🚀 Точка входа (FastAPI + graceful shutdown)
├── chat_terminal.py                # 💬 Терминальный клиент чата
├── ingest_knowledge.py             # 📥 Загрузчик knowledge/ в SQLite
├── train_neural_brain.py           # 🧠 Обучение нейросетевых весов (векторизованное numpy)
├── check_brain.py                  # 🔍 Диагностика БД и весов
├── start_napi_chat.bat             # 🪟 Быстрый запуск на Windows
├── requirements.txt                # 📚 Зависимости
└── README.md                       # 📖 Документация
```

---

## ⚙️ Как работает система (Жизненный цикл запроса)

Обработка каждого запроса проходит через **5 строгих этапов**, защищающих систему от OOM (Out of Memory):

### Шаг 1: Префильтрация (Gatekeeper)
Пользователь отправляет запрос на RU, EN или PL.
`core/gatekeeper.py` проверяет длину строки, язык и блокирующие паттерны.
Если запрос превышает лимит (4000 символов) или содержит запрещённый контент — система мгновенно отклоняет его, **не задействуя нейросеть**.

### Шаг 2: Сборка контекста (Prompt Builder)
`core/prompt_builder.py` берёт запрос и собирает промпт:
1. **ДНК** — системный промпт `[SYSTEM_PROMPT_NAPI_CORE_V1]`
2. **Знания** — теги из `napi_brain.db` (FTS5-поиск по ключевым словам)
3. **Опыт** — правила из Дневника Рефлексии (`[REFLECTED_RULE: ...]`)
4. Склеивает всё в финальный промпт

### Шаг 3: Размышление и Генерация (Inference)
`core/engine.py` передаёт промпт в движок:
- Локальная GGUF-модель (если включена в `config.yaml`)
- Удалённый провайдер через OpenAI-совместимый API
- Napi сначала генерирует невидимый блок `<THINK>...</THINK>`
- Затем генерируется финальный текстовый ответ
- Если генерация достигает `max_visible_tokens` (512 по умолчанию) — процесс принудительно останаврывается

### Шаг 4: Сохранение и очистка (Stateless Drop)
Как только ответ выведен пользователю:
- `<THINK>` блок вырезается из видимого ответа
- Сообщение и заметки сохраняются в SQLite (persistent connection)
- GC запускается раз в 50 запросов, не на каждый ответ
- Napi **забывает** контекст запроса. Память снова свободна

### Шаг 5: Фоновое Мягкое Обучение (Soft Learning)
Если в `config.yaml` активен Учитель:
- Связка `[Вопрос + Ответ Napi]` отправляется Учителю
- `soft_learning/rule_extractor.py` парсит критику и создаёт `[REFLECTED_RULE: ...]`
- Правило сохраняется в таблицу `reflection_diary` (SQLite)
- При следующем похожем вопросе Napi прочитает это правило на **Шаге 2**

---

## 🛡️ Оптимизации для 2 CPU / 2 GB RAM

| Механизм | Описание |
|----------|----------|
| Persistent SQLite | Одно соединение вместо нового на каждый запрос; WAL + mmap |
| Persistent HTTP | Один httpx.AsyncClient с connection pooling и keepalive |
| float16 эмбеддинги | Нейросетевые веса в половинной точности (×2 экономия RAM) |
| Векторизованное обучение | NumPy батчи вместо Python-циклов (ускорение ×10-100) |
| FNV-1a хеш | Быстрый хеш в self_brain вместо hashlib.sha256 |
| Прекомпиляция регулярок | Все regex компилируются при импорте, не на каждый вызов |
| GC раз в 50 запросов | Вместо gc.collect() на каждый ответ |
| Единный конфиг | Один парсинг yaml через config_loader.py |
| Graceful shutdown | Корректное закрытие DB и HTTP-клиента при завершении |
| GGUF квантование | 4-битное или 3-битное (локальная модель ~1.5 ГБ) |
| Semaphore | Максимум 2 параллельных запроса к провайдеру |
| Gatekeeper | Префильтрация без вызова модели |
| RAG через FTS5 | Только релевантные чанки загружаются в память |

---

## 🔌 Поддерживаемые провайдеры

Napi работает с **любым OpenAI-совместимым API**. В `config.yaml` в секции `engine.provider` укажи `base_url` и модели:

| Провайдер | base_url | Примечание |
|-----------|----------|------------|
| **LM Studio** | `http://localhost:1234/v1` | Локальный сервер на твоём ПК |
| **Ollama** | `http://localhost:11434/v1` | Локальный инференс |
| **vLLM** | `http://localhost:8000/v1` | Быстрый локальный сервер |
| **Together AI** | `https://api.together.xyz/v1` | Облачный инференс |
| **Groq** | `https://api.groq.com/openai/v1` | Быстрый облачный инференс |
| **OpenAI** | `https://api.openai.com/v1` | Оригинальный OpenAI API |
| **Любой другой** | `https://your-server/v1` | Любой сервер с `/chat/completions` |

API-ключ задаётся через переменную окружения (по умолчанию `NAPI_API_KEY`) или прямо в `config.yaml`.
Для обычного чата нужна только основная модель: локальный GGUF в `engine.local` или `provider.base_url` + `provider.chat_model`.
`teacher_model`, `filter_model` и `reasoning_model` — дополнительные модули, без них Napi всё равно должен отвечать:

```yaml
engine:
  provider:
    api_key_env: "NAPI_API_KEY"
    base_url: "http://localhost:1234/v1"
    chat_model: "my-local-model"          # основная модель Napi
    teacher_model: "my-local-model"       # опционально: наставник
    filter_model: "my-local-model"        # опционально: умная фильтрация
    reasoning_model: "my-local-model"     # опционально: внутренний анализ
    vision_model: "my-vision-model"       # если провайдер поддерживает зрение
```

Локальный GGUF и удалённый провайдер могут работать **одновременно**: GGUF как основной, провайдер как фоллбек.

### Автономный режим (без внешней LLM)

Если ни GGUF, ни провайдер не настроены, Napi запускается через собственный локальный мозг `napi-self-brain`.
Он не использует Наставника и не обращается к внешней модели: понимает запрос через ключевые смыслы, ищет знания в SQLite,
выделяет релевантные фрагменты, учитывает заметки памяти и собирает ответ локально.

Если обучен файл `models/napi_neural_brain.npz`, локальный мозг работает как `napi-neural-brain`.
Это собственный нейросетевой слой Napi: NumPy skip-gram эмбеддинги, обученные на `knowledge/`, памяти и истории.

Обучить/переобучить веса:
```powershell
python train_neural_brain.py
```

---

## 📦 Зависимости

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

Опционально (для локальной GGUF модели):
```
llama-cpp-python
```

---

## 🚀 Запуск

```powershell
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Настроить config.yaml (base_url, модели, API-ключ)

# 3. Задать API-ключ (если нужен)
$env:NAPI_API_KEY = "твой-ключ"

# 4. Запустить сервер
python main.py
```

Или через uvicorn:
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Терминальный клиент:
```powershell
python chat_terminal.py
# или
start_napi_chat.bat
```

Загрузить знания:
```powershell
python ingest_knowledge.py
python ingest_knowledge.py "C:\path\to\my\docs"
```

---

## ⚙️ Конфигурация (config.yaml)

Все настройки в одном файле:

| Секция | Что настраивает |
|--------|----------------|
| `dna` | Лимиты: 4000 символов, 512 токенов, языки RU/EN/PL |
| `engine.local` | GGUF модель: путь, n_ctx, n_threads |
| `engine.provider` | **Провайдер**: base_url, модели (ОБЯЗАТЕЛЬНО), таймаут, конкурентность |
| `gatekeeper` | Префильтрация: длина, блокирующие паттерны |
| `storage` | Путь к БД, лимиты истории, заметок, чанков |
| `soft_learning` | Учитель: порог оценки, температуры |
| `reasoning` | THINK-блок: включение, max_context, max_tokens |
| `server` | Хост, порт, воркеры |

---

## 🔌 API

### `POST /chat`
```json
{
  "message": "Привет, Napi!",
  "session_id": "default",
  "language": "auto",
  "self_improve": true
}
```

### `POST /vision`
```json
{
  "question": "Оцени UI этого экрана",
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
  "comment": "Ответ был полезен"
}
```

### `POST /knowledge/add`
Добавить один документ в базу знаний:
```json
{
  "source": "manual://my-topic",
  "title": "Моя тема",
  "content": "Текст знаний, который Napi должен использовать в ответах."
}
```

### `POST /knowledge/ingest`
Загрузить все `.md` и `.txt` файлы из папки:
```json
{
  "directory": "./knowledge"
}
```

### `GET /knowledge/search`
Проверить, что знания находятся:
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

## 📐 Масштабирование

| Профиль | Конкурентность | История | Чанки | Модель |
|---------|---------------|---------|-------|--------|
| 2 CPU / 2 GB | 2 | 24 | 6 | GGUF 2B или провайдер |
| 4 CPU / 4 GB | 4 | 40 | 8 | GGUF 2B + провайдер |
| 8 CPU / 8 GB | 8 | 60 | 10 | GGUF 7B + провайдер |
| 16+ CPU / 16+ GB | 16 | 100 | 12 | GGUF 13B + провайдер |

Все лимиты меняются в `config.yaml` — **без изменения кода**.