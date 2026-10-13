"""LLM client: provider adapters with proper error handling."""

import httpx
from app.core.config import (
    LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    LLM_TIMEOUT_SECONDS, LLM_MAX_OUTPUT_TOKENS,
)


async def test_connection() -> dict:
    """Send a minimal request to verify LLM connectivity."""
    provider = LLM_PROVIDER.lower()
    base_url = LLM_BASE_URL.rstrip("/")
    api_key = LLM_API_KEY
    model = LLM_MODEL

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
            if provider in ("openai", "openai_compatible", "deepseek"):
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                    },
                )
            elif provider == "anthropic":
                resp = await client.post(
                    f"{base_url}/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
            else:
                return {"success": False, "error": f"不支持的 provider: {provider}"}

            if resp.status_code == 200:
                return {"success": True, "provider": provider, "model": model}
            elif resp.status_code == 401:
                return {"success": False, "error": "认证失败 (401)：API Key 无效"}
            elif resp.status_code == 403:
                return {"success": False, "error": "禁止访问 (403)：权限不足"}
            elif resp.status_code == 404:
                return {"success": False, "error": f"模型或端点不存在 (404)"}
            elif resp.status_code == 429:
                return {"success": False, "error": "请求过于频繁 (429)：请稍后重试"}
            else:
                return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except httpx.TimeoutException:
        return {"success": False, "error": f"连接超时 ({LLM_TIMEOUT_SECONDS}s)"}
    except httpx.ConnectError:
        return {"success": False, "error": f"无法连接到 {base_url}"}
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}


async def call_llm(prompt: str) -> tuple[str, dict]:
    """Call LLM and return (response_text, call_metadata).

    Raises on failure - no silent fallback.
    """
    provider = LLM_PROVIDER.lower()
    base_url = LLM_BASE_URL.rstrip("/")
    api_key = LLM_API_KEY
    model = LLM_MODEL

    metadata = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "succeeded": False,
        "latency_ms": None,
        "token_usage": None,
        "status_code": None,
        "error": None,
    }

    import time
    start = time.monotonic()

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        if provider in ("openai", "openai_compatible", "deepseek"):
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": LLM_MAX_OUTPUT_TOKENS,
                },
            )
        elif provider == "anthropic":
            resp = await client.post(
                f"{base_url}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": LLM_MAX_OUTPUT_TOKENS,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    elapsed = (time.monotonic() - start) * 1000
    metadata["latency_ms"] = round(elapsed, 1)
    metadata["status_code"] = resp.status_code

    if resp.status_code == 401:
        metadata["error"] = "Authentication failed (401)"
        raise PermissionError("LLM authentication failed: invalid API key")
    if resp.status_code == 403:
        metadata["error"] = "Forbidden (403)"
        raise PermissionError("LLM access forbidden")
    if resp.status_code == 429:
        metadata["error"] = "Rate limited (429)"
        raise RuntimeError("LLM rate limited, please retry later")
    if resp.status_code >= 400:
        metadata["error"] = f"HTTP {resp.status_code}"
        raise RuntimeError(f"LLM request failed: HTTP {resp.status_code}")

    # Parse response
    if provider in ("openai", "openai_compatible", "deepseek"):
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage")
        if usage:
            metadata["token_usage"] = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
    elif provider == "anthropic":
        data = resp.json()
        text = data["content"][0]["text"]
        usage = data.get("usage")
        if usage:
            metadata["token_usage"] = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }

    metadata["succeeded"] = True
    return text, metadata
