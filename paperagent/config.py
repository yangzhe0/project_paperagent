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

MODEL_PROFILES = {
    "fast": os.environ.get("PAPERAGENT_FAST_MODEL", "qwen2.5:7b-instruct"),
    "deep": os.environ.get("PAPERAGENT_DEEP_MODEL", "qwen3:30b"),
}
DEFAULT_MODEL_PROFILE = os.environ.get("PAPERAGENT_MODEL_PROFILE", "fast")
DEFAULT_MODEL = os.environ.get(
    "PAPERAGENT_MODEL_NAME",
    MODEL_PROFILES.get(DEFAULT_MODEL_PROFILE, MODEL_PROFILES["fast"]),
)
OLLAMA_NUM_CTX = int(os.environ.get("PAPERAGENT_OLLAMA_NUM_CTX", "3072"))
OLLAMA_TIMEOUT = int(os.environ.get("PAPERAGENT_OLLAMA_TIMEOUT", "300"))

BGE_M3_EMBEDDING_MODEL = str(LOCAL_BGE_M3_DIR)
DEFAULT_EMBEDDING_MODEL = os.environ.get("PAPERAGENT_EMBEDDING_MODEL", BGE_M3_EMBEDDING_MODEL)


def get_model_name(profile: str | None = None) -> str:
    """Resolve a UI model profile to an Ollama model name."""

    if not profile:
        return DEFAULT_MODEL
    return MODEL_PROFILES.get(profile, DEFAULT_MODEL)
