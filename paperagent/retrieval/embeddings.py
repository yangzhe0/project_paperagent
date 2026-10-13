"""Local embedding adapter with predictable batching and no network fallback."""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalSentenceEmbeddings(Embeddings):
    def __init__(self, model_path: str, batch_size: int = 32, max_seq_length: int = 1024):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Local embedding model not found: {path}")
        self.model_path = str(path)
        self.batch_size = batch_size
        
        # 智能选择计算设备 (支持 GPU 加速，无卡时自动优化 CPU 多核)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # 优化 CPU 并发推断
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            
        self.model = SentenceTransformer(self.model_path, device=device, local_files_only=True)
        self.model.max_seq_length = max_seq_length

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        # 净化查询，保持与学术文献语义对齐
        cleaned = " ".join(text.strip().split())
        return self.embed_documents([cleaned])[0]
