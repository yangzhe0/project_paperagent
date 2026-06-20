import re
import os
import warnings
from pathlib import Path

# 演示环境使用项目内本地模型，避免运行时临时联网下载 HuggingFace 权重。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from paperagent.config import DEFAULT_EMBEDDING_MODEL, MINERU_DIR, PAPER_DIR, VECTORSTORE_DIR


CHUNK_SIZE = int(os.environ.get("PAPERAGENT_CHUNK_SIZE", "1100"))
CHUNK_OVERLAP = int(os.environ.get("PAPERAGENT_CHUNK_OVERLAP", "180"))
RETRIEVER_SEARCH_K = int(os.environ.get("PAPERAGENT_RETRIEVER_K", "8"))
RETRIEVER_FETCH_K = int(os.environ.get("PAPERAGENT_RETRIEVER_FETCH_K", "30"))
RETRIEVER_DIVERSITY = float(os.environ.get("PAPERAGENT_RETRIEVER_DIVERSITY", "0.4"))


class PaperIngestor:
    """把 MinerU markdown 转成 FAISS 可检索的本地向量库。"""

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
        self.rebuild = rebuild
        self.embedding_model = str(embedding_model)

        # 当前依赖版本会提示 HuggingFaceEmbeddings 未来迁移；演示期先保持锁定版本。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        self.vectorstore = self.get_or_create_vectorstore()

    def get_docs(self):
        # 当前 RAG 只使用 MinerU 的 markdown；图片、JSON、标注 PDF 不参与建库。
        markdown_files = self.find_mineru_markdown_files()
        print(f"Using MinerU markdown: {len(markdown_files)} files")

        docs = []
        for markdown_path in markdown_files:
            docs.extend(self.docs_from_mineru_markdown(markdown_path))
        return docs

    def find_mineru_markdown_files(self):
        if not self.mineru_dir.exists():
            return []
        return sorted(
            path
            for path in self.mineru_dir.rglob("*.md")
            # 排除 _logs 等内部目录，避免把日志误当论文内容。
            if not any(part.startswith("_") for part in path.relative_to(self.mineru_dir).parts)
        )

    def docs_from_mineru_markdown(self, markdown_path):
        source = self.infer_source_filename(markdown_path)
        text = clean_markdown(markdown_path.read_text(encoding="utf-8", errors="ignore"))
        chunks = split_markdown(text)

        docs = []
        for chunk_id, chunk in enumerate(chunks, start=1):
            docs.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": source,
                        "filename": source,
                        "parser": "mineru",
                        "chunk_id": chunk_id,
                        "section_title": chunk["section_title"],
                        "page": chunk.get("page"),
                    },
                )
            )
        return docs

    def infer_source_filename(self, markdown_path):
        pdf_by_stem = {path.stem: path.name for path in self.paper_dir.glob("*.pdf")}
        if markdown_path.stem in pdf_by_stem:
            return pdf_by_stem[markdown_path.stem]

        for parent in markdown_path.parents:
            if parent == self.mineru_dir.parent:
                break
            if parent.name in pdf_by_stem:
                return pdf_by_stem[parent.name]

        return f"{markdown_path.stem}.pdf"

    def get_or_create_vectorstore(self):
        index_path = self.vectorstore_dir / "index.faiss"
        if index_path.exists() and not self.rebuild:
            # LangChain 的 FAISS 索引保存为 pickle，需要显式允许本地反序列化。
            return FAISS.load_local(
                str(self.vectorstore_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        docs = self.get_docs()
        if not docs:
            raise ValueError(
                "No MinerU markdown found. Run scripts/parse_with_mineru.sh first. "
                "PDFs without MinerU markdown are intentionally excluded."
            )

        for doc in docs:
            doc.metadata["chunk_text"] = doc.page_content

        vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.vectorstore_dir))
        return vectorstore

    def get_retriever(self):
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_SEARCH_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_DIVERSITY,
            },
        )


def clean_markdown(text):
    """轻量清洗：保留论文正文结构，去掉孤立页码和分隔线。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if re.fullmatch(r"[-_=]{3,}", stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_markdown(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """先按标题分节，再按段落切 chunk，尽量不把完整段落切碎。"""

    sections = split_by_heading(text)
    chunks = []

    for section_title, section_text in sections:
        for chunk_text in split_long_text(section_text, chunk_size, chunk_overlap):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "section_title": section_title,
                    "text": chunk_text,
                    "page": None,
                }
            )

    return chunks


def split_by_heading(text):
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [("全文", text)]

    sections = []
    first_start = matches[0].start()
    if first_start > 0:
        preface = text[:first_start].strip()
        if preface:
            sections.append(("题名前内容", preface))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group(2).strip()
        sections.append((title, text[start:end].strip()))

    return sections


def split_long_text(text, chunk_size, chunk_overlap):
    if len(text) <= chunk_size:
        return [text]

    blocks = split_paragraphs(text)
    chunks = []
    current = ""

    for block in blocks:
        if len(block) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_by_window(block, chunk_size, chunk_overlap))
            continue

        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            current = block

    if current.strip():
        chunks.append(current.strip())

    return add_overlap(chunks, chunk_overlap)


def split_paragraphs(text):
    parts = re.split(r"\n\s*\n", text)
    return [part.strip() for part in parts if part.strip()]


def split_by_window(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def add_overlap(chunks, chunk_overlap):
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for index in range(1, len(chunks)):
        previous_tail = chunks[index - 1][-chunk_overlap:]
        overlapped.append(f"{previous_tail}\n\n{chunks[index]}")
    return overlapped


def main():
    PaperIngestor(rebuild=True)
    print(f"Rebuilt FAISS index in {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
