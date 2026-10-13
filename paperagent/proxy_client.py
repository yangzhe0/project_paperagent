import json
import os
import urllib.error
import urllib.request
from typing import Generator, Optional

from paperagent.config import (
    DEFAULT_TIMEOUT,
    PROXY_API_KEY,
    PROXY_URL,
)


class ProxyClient:
    """
    OpenAI 标准兼容协议的学术推理客户端。
    支持非流式与流式调用，并可捕获前沿思考模型的 reasoning_content。
    """

    def __init__(
        self,
        model_name: str,
        url: str = PROXY_URL,
        api_key: str = PROXY_API_KEY,
        timeout: int = DEFAULT_TIMEOUT,
        opener=None,
    ):
        self.model_name = model_name
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.last_reasoning: Optional[str] = None
        self.last_usage: dict = {}

    def _chat_url(self) -> str:
        if self.url.endswith("/chat/completions"):
            return self.url
        if self.url.endswith("/v1"):
            return f"{self.url}/chat/completions"
        return f"{self.url}/v1/chat/completions"

    def build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.2,
    ) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """非流式调用，返回回复文本，并将思考过程记录在 self.last_reasoning。"""
        payload = self.build_payload(prompt, system_prompt=system_prompt, stream=False)
        req = self._build_request(payload)

        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(body or f"Inference HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to Academic Inference Engine: {exc.reason}") from exc

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        self.last_reasoning = message.get("reasoning_content")
        self.last_usage = data.get("usage", {})
        content = message.get("content") or ""
        return content

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """流式调用，逐 token 输出 content，并在结束时保存完整的 reasoning_content。"""
        self.last_reasoning = None
        self.last_usage = {}
        reasoning_chunks = []

        payload = self.build_payload(prompt, system_prompt=system_prompt, stream=True)
        req = self._build_request(payload)

        try:
            resp = self.opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(body or f"Inference HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to Academic Inference Engine: {exc.reason}") from exc

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "error" in chunk_data:
                        raise RuntimeError(chunk_data["error"])

                    usage = chunk_data.get("usage")
                    if usage:
                        self.last_usage = usage

                    choices = chunk_data.get("choices")
                    if choices:
                        delta = choices[0].get("delta", {})
                        rc = delta.get("reasoning_content")
                        if rc:
                            reasoning_chunks.append(rc)
                        c = delta.get("content")
                        if c:
                            yield c
        finally:
            resp.close()
            if reasoning_chunks:
                self.last_reasoning = "".join(reasoning_chunks)

    def check_health(self) -> bool:
        """检查代理健康状态"""
        health_url = f"{self.url.replace('/v1', '')}/health"
        req = urllib.request.Request(health_url, method="GET")
        try:
            with self.opener.open(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _build_request(self, payload: dict) -> urllib.request.Request:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = json.dumps(payload).encode("utf-8")
        return urllib.request.Request(self._chat_url(), data=data, headers=headers, method="POST")
