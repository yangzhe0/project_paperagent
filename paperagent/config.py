from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
PAPER_DIR = DATA_DIR / "papers"
MINERU_DIR = DATA_DIR / "mineru"
PROCESSED_DIR = DATA_DIR / "processed"

VECTORSTORE_DIR = Path(os.environ.get("PAPERAGENT_VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore"))
PAPER_INDEX_PATH = VECTORSTORE_DIR / "paper_index.json"
LOCAL_BGE_M3_DIR = PROJECT_ROOT / "models" / "bge-m3"

# ===== 推理引擎与模型配置 =====
PROXY_URL = os.environ.get("PAPERAGENT_PROXY_URL", "http://127.0.0.1:8000/v1")
PROXY_API_KEY = os.environ.get("PAPERAGENT_PROXY_API_KEY", "")
DEFAULT_BACKEND = os.environ.get("PAPERAGENT_BACKEND", "proxy")
DEFAULT_TIMEOUT = int(os.environ.get("PAPERAGENT_TIMEOUT", "300"))


@dataclass(frozen=True)
class ModelProfile:
    name: str
    num_ctx: int
    think: bool | str
    label: str = ""
    description: str = ""
    backend: str = "proxy"

    @property
    def model_name(self) -> str:
        return self.name


# 统一学术推理模型档位配置字典
MODEL_PROFILE_SPECS = {
    "flash": ModelProfile(
        name=os.environ.get("PAPERAGENT_FLASH_MODEL", "gemini-3.8-flash-high"),
        num_ctx=128000,
        think=True,
        label="⚡ 极速推理",
        description="学术推理内核 · Gemini 3.8 Flash High 超大上下文与敏捷响应",
        backend="proxy",
    ),
    "thinking": ModelProfile(
        name=os.environ.get("PAPERAGENT_THINKING_MODEL", "claude-sonnet-4-6-thinking"),
        num_ctx=128000,
        think="high",
        label="🧠 深度推演",
        description="思维链推理内核 · Claude Sonnet 4.6 Thinking 严密机理与精密星表对比",
        backend="proxy",
    ),
    "pro": ModelProfile(
        name=os.environ.get("PAPERAGENT_PRO_MODEL", "gemini-3.1-pro"),
        num_ctx=128000,
        think=True,
        label="🔬 综合研报",
        description="长文本综合引擎 · Gemini 3.1 Pro 超长跨篇综合与综述提炼",
        backend="proxy",
    ),
}

# 兼容别名映射
MODEL_PROFILE_SPECS["proxy_flash"] = MODEL_PROFILE_SPECS["flash"]
MODEL_PROFILE_SPECS["proxy_thinking"] = MODEL_PROFILE_SPECS["thinking"]
MODEL_PROFILE_SPECS["proxy_pro"] = MODEL_PROFILE_SPECS["pro"]

# 兼容模型名字典映射
MODEL_PROFILES = {key: spec.name for key, spec in MODEL_PROFILE_SPECS.items()}

DEFAULT_MODEL_PROFILE = os.environ.get("PAPERAGENT_MODEL_PROFILE", "flash")
DEFAULT_MODEL = os.environ.get(
    "PAPERAGENT_MODEL_NAME",
    MODEL_PROFILES.get(DEFAULT_MODEL_PROFILE, MODEL_PROFILES["flash"]),
)

# 本地引擎兼容配置
OLLAMA_NUM_CTX = int(os.environ.get("PAPERAGENT_OLLAMA_NUM_CTX", "16384"))
OLLAMA_TIMEOUT = int(os.environ.get("PAPERAGENT_OLLAMA_TIMEOUT", "300"))
OLLAMA_GENERATE_URL = os.environ.get(
    "PAPERAGENT_OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate",
)

# 增强学术检索配置
RAG_MAX_CONTEXT_CHARS = int(os.environ.get("PAPERAGENT_MAX_CONTEXT_CHARS", "8000"))
RAG_TOP_K_DOCS = int(os.environ.get("PAPERAGENT_TOP_K_DOCS", "10"))

BGE_M3_EMBEDDING_MODEL = str(LOCAL_BGE_M3_DIR)
DEFAULT_EMBEDDING_MODEL = os.environ.get("PAPERAGENT_EMBEDDING_MODEL", BGE_M3_EMBEDDING_MODEL)


def get_model_profile(profile_name: str) -> ModelProfile:
    """获取指定 profile 的详细规范。"""
    if profile_name in MODEL_PROFILE_SPECS:
        return MODEL_PROFILE_SPECS[profile_name]
    return MODEL_PROFILE_SPECS[DEFAULT_MODEL_PROFILE]


def get_model_name(profile: str | None = None) -> str:
    """根据 profile 获取对应的模型名。"""
    if not profile:
        return DEFAULT_MODEL
    return MODEL_PROFILES.get(profile, DEFAULT_MODEL)


def get_model_runtime(model_name: str) -> tuple[int, bool | str]:
    """为 OllamaClient 等提供运行时的 num_ctx 与 think 配置。"""
    for spec in MODEL_PROFILE_SPECS.values():
        if spec.name == model_name:
            return spec.num_ctx, spec.think
    return (16384, False)


def get_backend_for_model(model_name: str) -> str:
    """根据模型名称智能判断使用云端推理还是本地 Ollama。"""
    for spec in MODEL_PROFILE_SPECS.values():
        if spec.name == model_name:
            return spec.backend
    if any(m in model_name for m in ["gemini", "claude", "gpt"]):
        return "proxy"
    return "ollama"
