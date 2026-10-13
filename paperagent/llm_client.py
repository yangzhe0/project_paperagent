from paperagent.config import (
    DEFAULT_BACKEND,
    PROXY_API_KEY,
    PROXY_URL,
    get_backend_for_model,
)

try:
    from paperagent.generation.ollama import OllamaClient
except ImportError:
    from paperagent.ollama_client import OllamaClient

from paperagent.proxy_client import ProxyClient


def create_llm_client(
    model_name: str,
    backend: str | None = None,
    **kwargs,
):
    """
    根据模型名或 backend 自动创建适配的模型客户端。
    - "proxy": 云端学术推理内核 (Gemini 3.8 Flash, Claude Thinking, Gemini Pro 等)
    - "ollama": 本地部署实例 (Qwen 等)
    """
    effective_backend = backend or get_backend_for_model(model_name)
    if effective_backend == "proxy":
        url = kwargs.get("url", PROXY_URL)
        api_key = kwargs.get("api_key", PROXY_API_KEY)
        return ProxyClient(model_name=model_name, url=url, api_key=api_key, **kwargs)
    return OllamaClient(model_name=model_name, **kwargs)
