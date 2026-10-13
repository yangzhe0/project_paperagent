"""Build and load the versioned PaperAgent retrieval index."""

import json
import os
import tempfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from paperagent.config import DEFAULT_EMBEDDING_MODEL, MINERU_DIR, PAPER_DIR, VECTORSTORE_DIR
from paperagent.retrieval.chunking import clean_markdown, split_markdown
from paperagent.retrieval.embeddings import LocalSentenceEmbeddings
from paperagent.retrieval.hybrid import HybridRetriever


INDEX_SCHEMA_VERSION = 2
MIN_CHUNK_CHARS = 80
RETRIEVER_K = int(os.environ.get("PAPERAGENT_RETRIEVER_K", "7"))
RETRIEVER_FETCH_K = int(os.environ.get("PAPERAGENT_RETRIEVER_FETCH_K", "24"))


class KnowledgeBase:
    def __init__(
        self,
        mineru_dir=MINERU_DIR,
        paper_dir=PAPER_DIR,
        vectorstore_dir=VECTORSTORE_DIR,
        rebuild=False,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    ):
        self.mineru_dir = Path(mineru_dir)
        self.paper_dir = Path(paper_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model = str(embedding_model)
        self.embeddings = LocalSentenceEmbeddings(self.embedding_model)
        self.vectorstore = self._load_or_build(rebuild)
        self.documents = self._stored_documents()
        self.retriever = HybridRetriever(
            self.vectorstore,
            self.documents,
            k=RETRIEVER_K,
            fetch_k=RETRIEVER_FETCH_K,
        )

    def get_retriever(self):
        return self.retriever

    def get_docs(self) -> list[Document]:
        documents = []
        for markdown_path in self.find_markdown_files():
            source = self.infer_source_filename(markdown_path)
            paper_title = Path(source).stem
            cleaned = clean_markdown(markdown_path.read_text(encoding="utf-8", errors="ignore"))
            for chunk_id, chunk in enumerate(split_markdown(cleaned), start=1):
                content = chunk["text"]
                if len(content) < MIN_CHUNK_CHARS:
                    continue
                section = chunk["section_title"]
                embedding_text = f"论文：{paper_title}\n章节：{section}\n{content}"
                documents.append(
                    Document(
                        page_content=embedding_text,
                        metadata={
                            "source": source,
                            "filename": source,
                            "parser": "mineru",
                            "chunk_id": chunk_id,
                            "section_title": section,
                            "page": chunk.get("page"),
                            "content": content,
                        },
                    )
                )
        return documents

    def find_markdown_files(self) -> list[Path]:
        if not self.mineru_dir.exists():
            return []
        return sorted(
            path
            for path in self.mineru_dir.rglob("*.md")
            if not any(part.startswith("_") for part in path.relative_to(self.mineru_dir).parts)
        )

    def infer_source_filename(self, markdown_path: Path) -> str:
        pdf_by_stem = {path.stem: path.name for path in self.paper_dir.glob("*.pdf")}
        if markdown_path.stem in pdf_by_stem:
            return pdf_by_stem[markdown_path.stem]
        for parent in markdown_path.parents:
            if parent == self.mineru_dir.parent:
                break
            if parent.name in pdf_by_stem:
                return pdf_by_stem[parent.name]
        return f"{markdown_path.stem}.pdf"

    def _load_or_build(self, rebuild: bool):
        if not rebuild:
            self._validate_index()
            return FAISS.load_local(
                str(self.vectorstore_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        documents = self.get_docs()
        if not documents:
            raise ValueError("No MinerU markdown found. Run scripts/parse_with_mineru.sh first.")
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        metadata = json.dumps(
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "chunk_count": len(documents),
                "embedding_model": self.embedding_model,
                "retrieval": "dense_bm25_rrf",
            },
            ensure_ascii=False,
            indent=2,
        )
        with tempfile.TemporaryDirectory(
            prefix=".paperagent-index-", dir=self.vectorstore_dir.parent
        ) as temporary_dir:
            temporary_path = Path(temporary_dir)
            vectorstore.save_local(str(temporary_path))
            (temporary_path / "index_meta.json").write_text(metadata, encoding="utf-8")
            for filename in ("index.faiss", "index.pkl", "index_meta.json"):
                os.replace(temporary_path / filename, self.vectorstore_dir / filename)
        return vectorstore

    def _validate_index(self):
        required = [self.vectorstore_dir / "index.faiss", self.vectorstore_dir / "index.pkl"]
        if not all(path.exists() for path in required):
            raise ValueError("Knowledge base is not built. Use the Rebuild action first.")
        try:
            metadata = json.loads(self._metadata_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError("Knowledge base format is outdated. Rebuild it once.") from exc
        if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Knowledge base schema changed. Rebuild it once.")

    def _stored_documents(self) -> list[Document]:
        values = getattr(self.vectorstore.docstore, "_dict", {}).values()
        return list(values)

    def _metadata_path(self) -> Path:
        return self.vectorstore_dir / "index_meta.json"


def main():
    knowledge_base = KnowledgeBase(rebuild=True)
    print(f"Rebuilt hybrid index with {len(knowledge_base.documents)} chunks")


if __name__ == "__main__":
    main()
