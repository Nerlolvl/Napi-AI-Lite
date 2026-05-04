from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import yaml

log = logging.getLogger("napi.engine")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


CONFIG = _load_config()

_LOCAL = CONFIG.get("engine", {}).get("local", {})
_PROVIDER = CONFIG.get("engine", {}).get("provider", {})

_REQUIRED_MODELS = ["chat_model", "teacher_model", "filter_model", "reasoning_model"]


class LocalEngine:
    """GGUF model inference via llama-cpp-python."""

    def __init__(self) -> None:
        self.model = None
        self._model_path = _LOCAL.get("model_path", "./models/napi-2b-q4_k_m.gguf")
        self._n_ctx = _LOCAL.get("n_ctx", 2048)
        self._n_gpu_layers = _LOCAL.get("n_gpu_layers", 0)
        self._n_threads = _LOCAL.get("n_threads", 2)
        self._max_tokens = _LOCAL.get("max_tokens", 512)
        self._temperature = _LOCAL.get("temperature", 0.55)
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from llama_cpp import Llama

            path = Path(self._model_path)
            if not path.exists():
                log.warning("Local model not found at %s — skipping load", path)
                return
            log.info("Loading local GGUF model from %s", path)
            self.model = Llama(
                model_path=str(path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                n_threads=self._n_threads,
                verbose=False,
            )
            self._loaded = True
            log.info("Local model loaded successfully")
        except ImportError:
            log.warning("llama-cpp-python not installed — local engine disabled")
        except Exception as exc:
            log.error("Failed to load local model: %s", exc)

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        if not self._loaded or self.model is None:
            raise RuntimeError("Local model not loaded")
        result = self.model(
            prompt,
            max_tokens=max_tokens or self._max_tokens,
            temperature=self._temperature,
        )
        return result["choices"][0]["text"] if "choices" in result else str(result)

    @property
    def available(self) -> bool:
        return self._loaded


class ProviderEngine:
    """Remote inference via any OpenAI-compatible API.

    Compatible with: LM Studio, Ollama, vLLM, Together AI,
    Groq, OpenAI, and any server implementing /chat/completions.
    """

    def __init__(self) -> None:
        api_key_env = _PROVIDER.get("api_key_env", "NAPI_API_KEY")
        self.api_key = os.environ.get(api_key_env, "") or _PROVIDER.get("api_key", "")
        self.base_url = _PROVIDER.get("base_url", "")
        self.chat_model = _PROVIDER.get("chat_model", "")
        self.timeout = _PROVIDER.get("timeout_seconds", 90)
        self.max_concurrent = _PROVIDER.get("max_concurrent", 2)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._extra_headers = _PROVIDER.get("extra_headers", {})

        missing_models = [
            key for key in _REQUIRED_MODELS if not _PROVIDER.get(key)
        ]
        if missing_models and _PROVIDER.get("enabled", True):
            log.warning(
                "Config: missing required models: %s. "
                "Set them in config.yaml under engine.provider.",
                ", ".join(missing_models),
            )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for key, value in self._extra_headers.items():
            headers[key] = value
        return headers

    async def chat(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        temperature: float = 0.55,
        max_tokens: int = 512,
    ) -> str:
        if not self.base_url:
            raise RuntimeError(
                "Provider base_url not configured. "
                "Set engine.provider.base_url in config.yaml"
            )
        model = model or self.chat_model
        if not model:
            raise RuntimeError(
                "No model specified. "
                "Set chat_model (and other models) in config.yaml under engine.provider"
            )

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        return data["choices"][0]["message"]["content"]


class InferenceEngine:
    """Unified engine: local GGUF first, remote provider fallback."""

    def __init__(self) -> None:
        self.local = LocalEngine()
        self.remote = ProviderEngine()

    def init_local(self) -> None:
        if _LOCAL.get("enabled", False):
            self.local.load()

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        prefer_local: bool = False,
    ) -> str:
        if prefer_local and self.local.available:
            prompt = self._messages_to_prompt(messages)
            return self.local.generate(
                prompt,
                max_tokens=max_tokens or _LOCAL.get("max_tokens", 512),
            )
        return await self.remote.chat(
            model=model,
            messages=messages,
            temperature=temperature or 0.55,
            max_tokens=max_tokens or 512,
        )

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if role == "system":
                parts.append(f"### System\n{content}")
            elif role == "assistant":
                parts.append(f"### Assistant\n{content}")
            else:
                parts.append(f"### User\n{content}")
        parts.append("### Assistant\n")
        return "\n\n".join(parts)