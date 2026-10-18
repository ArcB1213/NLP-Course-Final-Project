import json
from typing import AsyncIterator

import httpx


CONTINUE_PROMPT = "继续完成上一条回答，从中断处接着写，不要重复已经输出的内容，不要输出开场白。"


class IncompleteResponseError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, default_model: str, timeout: float = 60.0,
                 default_thinking: bool | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.default_thinking = default_thinking

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        continue_on_length: bool = False,
        max_continuations: int = 2,
        thinking: bool | None = None,
        require_complete: bool = False,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        working_messages = list(messages)
        content_parts: list[str] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(max_continuations + 1):
                payload = {
                    "model": model or self.default_model,
                    "messages": working_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                effective_thinking = self.default_thinking if thinking is None else thinking
                if effective_thinking is not None and payload["model"].startswith("deepseek"):
                    payload["thinking"] = {"type": "enabled" if effective_thinking else "disabled"}
                response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("DeepSeek returned an empty response")

                choice = choices[0]
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason")
                if require_complete and finish_reason != "stop":
                    raise IncompleteResponseError(f"Incomplete LLM response: {finish_reason}")
                if content:
                    content_parts.append(content)
                elif not content_parts:
                    if finish_reason == "length":
                        raise RuntimeError("LLM output budget exhausted before final content")
                    raise RuntimeError("DeepSeek returned empty message content")

                if not _should_continue(
                    text="".join(content_parts),
                    finish_reason=finish_reason,
                    enabled=continue_on_length,
                    attempt=attempt,
                    max_continuations=max_continuations,
                ):
                    break

                working_messages = [
                    *messages,
                    {"role": "assistant", "content": "".join(content_parts)},
                    {"role": "user", "content": CONTINUE_PROMPT},
                ]

        return "".join(content_parts)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        continue_on_length: bool = True,
        max_continuations: int = 2,
        thinking: bool | None = None,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        working_messages = list(messages)
        content_parts: list[str] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(max_continuations + 1):
                payload = {
                    "model": model or self.default_model,
                    "messages": working_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                }
                effective_thinking = self.default_thinking if thinking is None else thinking
                if effective_thinking is not None and payload["model"].startswith("deepseek"):
                    payload["thinking"] = {"type": "enabled" if effective_thinking else "disabled"}
                finish_reason: str | None = None
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                        else:
                            data_str = line.strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        if choice.get("finish_reason"):
                            finish_reason = choice.get("finish_reason")
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            content_parts.append(text)
                            yield text

                if not content_parts:
                    if finish_reason == "length":
                        raise RuntimeError("LLM output budget exhausted before final content")
                    raise RuntimeError("DeepSeek returned empty message content")

                if not _should_continue(
                    text="".join(content_parts),
                    finish_reason=finish_reason,
                    enabled=continue_on_length,
                    attempt=attempt,
                    max_continuations=max_continuations,
                ):
                    break

                working_messages = [
                    *messages,
                    {"role": "assistant", "content": "".join(content_parts)},
                    {"role": "user", "content": CONTINUE_PROMPT},
                ]


def _should_continue(
    *,
    text: str,
    finish_reason: str | None,
    enabled: bool,
    attempt: int,
    max_continuations: int,
) -> bool:
    if not enabled or attempt >= max_continuations:
        return False
    if finish_reason == "length":
        return True
    return _looks_incomplete(text)


def _looks_incomplete(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith(("，", "、", "：", "；", "（", "(", "【", "[", "-", "—", "/", "的", "和", "与", "及", "或")):
        return True
    if stripped.count("```") % 2 == 1 or stripped.count("**") % 2 == 1:
        return True

    terminal_punctuation = ("。", "！", "？", ".", "!", "?", "）", ")", "】", "]")
    last_line = stripped.splitlines()[-1].strip()
    if not stripped.endswith(terminal_punctuation) and "：" in last_line:
        tail = last_line.rsplit("：", 1)[-1].strip()
        return 0 < len(tail) <= 8
    return False
