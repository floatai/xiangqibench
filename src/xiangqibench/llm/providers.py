"""Model providers.

Sampling parameters (``temperature``, ``top_p``) and ``max_tokens`` are sent
only when set in the model config; otherwise each endpoint uses its own
defaults, as in the paper runs. The Anthropic API requires ``max_tokens``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from xiangqibench.config import ConfigError, ModelConfig
from xiangqibench.llm.base import Agent, Completion


def _usage(prompt: int | None, completion: int | None, **extra) -> dict:
    prompt, completion = prompt or 0, completion or 0
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, **extra}


def _require_key(cfg: ModelConfig) -> str:
    key = cfg.resolve_api_key()
    if not key:
        hint = f"set ${cfg.api_key_env}" if cfg.api_key_env else "set model.api_key_env"
        raise ConfigError(f"no API key for model {cfg.name!r}: {hint}")
    return key


def _is_filter_false_positive(exc: Exception) -> bool:
    text = str(exc).lower()
    return "invalid_prompt" in text or "usage policy" in text


class OpenAIChatAgent:
    """OpenAI Chat Completions, including any OpenAI-compatible endpoint
    (OpenRouter, vLLM, SGLang, DeepSeek, Together, ...) and Azure OpenAI."""

    def __init__(self, cfg: ModelConfig):
        import openai

        self.name = cfg.name
        self._cfg = cfg
        common: dict[str, Any] = dict(api_key=_require_key(cfg), timeout=cfg.timeout,
                                      max_retries=cfg.max_retries, default_headers=cfg.headers or None)
        self._client: openai.OpenAI
        if cfg.provider == "azure":
            if not cfg.base_url:
                raise ConfigError("azure provider requires model.base_url (the Azure endpoint)")
            self._client = openai.AzureOpenAI(azure_endpoint=cfg.base_url,
                                              api_version=cfg.api_version or "2024-12-01-preview",
                                              **common)
        else:
            self._client = openai.OpenAI(base_url=cfg.base_url, **common)

    def _kwargs(self, messages: Sequence[dict]) -> dict:
        cfg = self._cfg
        kwargs: dict = dict(model=cfg.model_id, messages=list(messages))
        for key in ("max_tokens", "temperature", "top_p", "reasoning_effort"):
            value = getattr(cfg, key)
            if value is not None:
                kwargs[key] = value
        if cfg.extra_body:
            kwargs["extra_body"] = cfg.extra_body
        return kwargs

    def complete(self, messages: Sequence[dict]) -> Completion:
        kwargs = self._kwargs(messages)
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not _is_filter_false_positive(exc):
                raise
            time.sleep(2)
            resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        usage = _usage(resp.usage.prompt_tokens, resp.usage.completion_tokens) if resp.usage else {}
        return Completion(text=msg.content or "", reasoning=reasoning or None,
                          usage=usage, finish_reason=getattr(choice, "finish_reason", None))


class OpenAIResponsesAgent:
    """OpenAI Responses API (``/responses``)."""

    def __init__(self, cfg: ModelConfig):
        import openai

        self.name = cfg.name
        self._cfg = cfg
        self._client = openai.OpenAI(api_key=_require_key(cfg), base_url=cfg.base_url,
                                     timeout=cfg.timeout, max_retries=cfg.max_retries,
                                     default_headers=cfg.headers or None)

    def complete(self, messages: Sequence[dict]) -> Completion:
        cfg = self._cfg
        instructions = "\n\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
        items = [{
            "role": m["role"],
            "content": [{"type": "output_text" if m["role"] == "assistant" else "input_text",
                         "text": m.get("content") or ""}],
        } for m in messages if m["role"] != "system"]
        kwargs: dict = dict(model=cfg.model_id, input=items)
        if cfg.max_tokens is not None:
            kwargs["max_output_tokens"] = cfg.max_tokens
        if instructions:
            kwargs["instructions"] = instructions
        if cfg.reasoning_effort:
            kwargs["reasoning"] = {"effort": cfg.reasoning_effort}
        for key in ("temperature", "top_p"):
            if getattr(cfg, key) is not None:
                kwargs[key] = getattr(cfg, key)
        if cfg.extra_body:
            kwargs["extra_body"] = cfg.extra_body
        resp = self._client.responses.create(**kwargs)

        text, reasoning = [], []
        for item in resp.output or []:
            if item.type == "message":
                text += [c.text for c in item.content if getattr(c, "type", "") in ("output_text", "text")]
            elif item.type == "reasoning":
                reasoning += [s.text for s in (item.summary or []) if getattr(s, "text", None)]
        usage = _usage(resp.usage.input_tokens, resp.usage.output_tokens) if resp.usage else {}
        finish = "stop" if resp.status == "completed" else (
            getattr(resp.incomplete_details, "reason", None) or resp.status)
        return Completion(text="".join(text), reasoning="".join(reasoning) or None,
                          usage=usage, finish_reason=finish)


class AnthropicAgent:
    """Anthropic Messages API. ``prompt_cache`` places an ephemeral cache
    breakpoint on the latest message so the growing prefix is reused."""

    def __init__(self, cfg: ModelConfig):
        import anthropic

        if cfg.max_tokens is None:
            raise ConfigError("the anthropic provider requires model.max_tokens")
        self.name = cfg.name
        self._cfg = cfg
        self._client = anthropic.Anthropic(api_key=_require_key(cfg), base_url=cfg.base_url,
                                           timeout=cfg.timeout, max_retries=cfg.max_retries,
                                           default_headers=cfg.headers or None)

    def complete(self, messages: Sequence[dict]) -> Completion:
        cfg = self._cfg
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [{"role": m["role"], "content": m.get("content") or ""}
                 for m in messages if m["role"] in ("user", "assistant")]
        if cfg.prompt_cache and convo:
            convo[-1] = {"role": convo[-1]["role"], "content": [{
                "type": "text", "text": convo[-1]["content"],
                "cache_control": {"type": "ephemeral"}}]}
        kwargs: dict = dict(model=cfg.model_id, messages=convo, max_tokens=cfg.max_tokens)
        if system:
            kwargs["system"] = system
        for key in ("temperature", "top_p"):
            if getattr(cfg, key) is not None:
                kwargs[key] = getattr(cfg, key)
        if cfg.extra_body:
            kwargs["extra_body"] = cfg.extra_body
        resp = self._client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        reasoning = "".join(getattr(b, "thinking", "") for b in resp.content
                            if getattr(b, "type", None) == "thinking") or None
        usage = {}
        if resp.usage:
            cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            usage = _usage((resp.usage.input_tokens or 0) + cache_write + cache_read,
                           resp.usage.output_tokens,
                           cache_creation_input_tokens=cache_write,
                           cache_read_input_tokens=cache_read)
        return Completion(text=text, reasoning=reasoning, usage=usage,
                          finish_reason=getattr(resp, "stop_reason", None))


def build_agent(cfg: ModelConfig) -> Agent:
    if cfg.provider in ("openai", "azure"):
        return OpenAIChatAgent(cfg)
    if cfg.provider == "openai-responses":
        return OpenAIResponsesAgent(cfg)
    if cfg.provider == "anthropic":
        return AnthropicAgent(cfg)
    raise ConfigError(f"unknown provider {cfg.provider!r}")
