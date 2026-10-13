import json
import os
from typing import Iterable
from urllib import error, request

from paperagent.config import OLLAMA_TIMEOUT, get_model_runtime


OLLAMA_GENERATE_URL = os.environ.get(
    "PAPERAGENT_OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate",
)


def parse_ollama_stream_lines(lines: Iterable[bytes]):
    for raw_line in lines:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        payload = json.loads(line)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        chunk = payload.get("response")
        if chunk:
            yield chunk


class OllamaClient:
    def __init__(
        self,
        model_name: str,
        url: str = OLLAMA_GENERATE_URL,
        num_ctx: int | None = None,
        think: bool | str | None = None,
        timeout: int = OLLAMA_TIMEOUT,
        opener=None,
    ):
        profile_num_ctx, profile_think = get_model_runtime(model_name)
        self.model_name = model_name
        self.url = url
        self.num_ctx = profile_num_ctx if num_ctx is None else num_ctx
        self.think = profile_think if think is None else think
        self.timeout = timeout
        self.opener = opener or request.build_opener(request.ProxyHandler({}))

    def build_payload(self, prompt: str, stream: bool = False) -> dict:
        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "think": self.think,
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
            },
        }

    def generate(self, prompt: str) -> str:
        response = self._post(self.build_payload(prompt, stream=False))
        result = json.loads(response.read().decode("utf-8"))
        if "error" in result:
            raise RuntimeError(result["error"])
        if "response" not in result:
            raise RuntimeError(json.dumps(result, ensure_ascii=False))
        return result["response"]

    def generate_stream(self, prompt: str):
        response = self._post(self.build_payload(prompt, stream=True))
        yield from parse_ollama_stream_lines(response)

    def list_models(self) -> set[str]:
        """Return locally available Ollama model names."""

        tags_url = f"{self.url.rsplit('/', 1)[0]}/tags"
        req = request.Request(tags_url, method="GET")
        try:
            response = self.opener.open(req, timeout=min(self.timeout, 3))
            payload = json.loads(response.read().decode("utf-8"))
            return {
                model["name"]
                for model in payload.get("models", [])
                if isinstance(model, dict) and model.get("name")
            }
        except (error.HTTPError, error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot query Ollama models: {exc}") from exc

    def _post(self, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return self.opener.open(req, timeout=self.timeout)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(body or f"Ollama HTTP {exc.code}: {exc.reason}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Cannot connect to Ollama: {exc.reason}") from exc
