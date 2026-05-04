from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from config_loader import CONFIG

log = logging.getLogger("napi.engine")

_LOCAL = CONFIG.get("engine", {}).get("local", {})
_PROVIDER = CONFIG.get("engine", {}).get("provider", {})

_OPTIONAL_MODELS = ["teacher_model", "filter_model", "reasoning_model"]


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

    Uses a single persistent httpx.AsyncClient for connection reuse
    and reduced memory allocation on 2-core / 2 GB machines.
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
        self._client: httpx.AsyncClient | None = None

        missing_optional_models = [
            key for key in _OPTIONAL_MODELS if not _PROVIDER.get(key)
        ]
        if missing_optional_models and _PROVIDER.get("enabled", True):
            log.warning(
                "Config: optional helper models are not set: %s. "
                "Teacher/filter/reasoning will be skipped until configured.",
                ", ".join(missing_optional_models),
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=15.0),
                limits=httpx.Limits(
                    max_connections=self.max_concurrent + 1,
                    max_keepalive_connections=self.max_concurrent,
                    keepalive_expiry=120,
                ),
                http2=False,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

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
        client = await self._get_client()
        async with self._semaphore:
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

    async def shutdown(self) -> None:
        await self.remote.close()

    @property
    def remote_ready(self) -> bool:
        return bool(self.remote.base_url and self.remote.chat_model)

    @property
    def can_chat(self) -> bool:
        return self.local.available or self.remote_ready

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        prefer_local: bool = False,
    ) -> str:
        if (prefer_local or not self.remote_ready) and self.local.available:
            prompt = self._messages_to_prompt(messages)
            return self.local.generate(
                prompt,
                max_tokens=max_tokens or _LOCAL.get("max_tokens", 512),
            )
        if not self.remote_ready:
            raise RuntimeError(
                "Napi has no primary model configured. Add a local GGUF model "
                "or set engine.provider.base_url and engine.provider.chat_model in config.yaml"
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