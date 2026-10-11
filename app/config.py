"""Central configuration: every module reads pre-parsed settings from here.

The YAML file itself is parsed exactly once in config_loader (shared with core/*),
this module only exposes the derived settings used by the API layer.
"""

from __future__ import annotations

from config_loader import CONFIG

DNA_CFG = CONFIG.get("dna", {})
STORAGE_CFG = CONFIG.get("storage", {})
SOFT_CFG = CONFIG.get("soft_learning", {})
REASONING_CFG = CONFIG.get("reasoning", {})
SERVER_CFG = CONFIG.get("server", {})

MAX_MESSAGE_LENGTH = DNA_CFG.get("max_message_length", 4000)
MAX_VISIBLE_TOKENS = DNA_CFG.get("max_visible_tokens", 512)

ENABLE_SOFT_LEARNING = SOFT_CFG.get("enabled", True)
MIN_SCORE_REVISION = SOFT_CFG.get("min_score_for_revision", 8)
REINFORCE_BONUS = SOFT_CFG.get("reinforce_bonus", 0.1)
TEMPS = SOFT_CFG.get("temperatures", {})

ENABLE_REASONING = REASONING_CFG.get("enabled", True)
REASONING_MAX_TOKENS = REASONING_CFG.get("max_tokens", 450)
REASONING_MAX_CONTEXT = REASONING_CFG.get("max_context_chars", 12000)

ESCALATION_CFG = CONFIG.get("escalation", {})
ENABLE_ESCALATION = ESCALATION_CFG.get("enabled", True)
ESCALATION_MIN_SCORE = ESCALATION_CFG.get("min_score", 7)

MAX_NOTES = STORAGE_CFG.get("max_memory_notes", 12)
MAX_CHUNKS = STORAGE_CFG.get("max_knowledge_chunks", 6)
MAX_RULES = STORAGE_CFG.get("max_reflected_rules", 8)
NOTE_MAX_LENGTH = STORAGE_CFG.get("note_max_length", 1000)
DB_PATH = STORAGE_CFG.get("database_path", "./storage/napi_brain.db")

RULE_TTL_DAYS = STORAGE_CFG.get("rule_ttl_days", 30)
MAX_RULES_TOTAL = STORAGE_CFG.get("max_rules_total", 200)

PROVIDER_CFG = CONFIG.get("engine", {}).get("provider", {})
CHAT_MODEL = PROVIDER_CFG.get("chat_model", "")
TEACHER_MODEL = PROVIDER_CFG.get("teacher_model", "")
FILTER_MODEL = PROVIDER_CFG.get("filter_model", "")
REASONING_MODEL = PROVIDER_CFG.get("reasoning_model", "")
VISION_MODEL = PROVIDER_CFG.get("vision_model", "")

GC_INTERVAL = SERVER_CFG.get("gc_interval_requests", 50)
