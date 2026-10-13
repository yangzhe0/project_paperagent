"""
RAG (Retrieval-Augmented Generation) 问答链模块

支持云端学术推理内核（Gemini 3.8 Flash, Claude Thinking 等）与本地推理引擎。
"""

import time
import unicodedata

from langchain.prompts import PromptTemplate

from paperagent.config import (
    RAG_MAX_CONTEXT_CHARS,
    RAG_TOP_K_DOCS,
    get_backend_for_model,
)
from paperagent.llm_client import create_llm_client
from paperagent.performance import AnswerResult, QueryTiming


class RAGChain:
    """
    RAG 问答链 - 负责向量检索、提示词构建和大模型调用
    """

    def __init__(
        self,
        retriever,
        model_name: str = "gemini-3.8-flash-high",
        ollama_client=None,
        llm_client=None,
    ):
        self.retriever = retriever
        self.model_name = model_name
        self.llm_client = llm_client or ollama_client or create_llm_client(model_name)
        # 兼容旧代码与单元测试
        self.ollama_client = self.llm_client
        self.last_result = None

    def answer(self, question: str) -> str:
        return self.answer_with_trace(question).text

    def answer_with_trace(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        documents = self.retriever.invoke(question)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        max_chars = (
            RAG_MAX_CONTEXT_CHARS
            if get_backend_for_model(self.model_name) == "proxy"
            else 800
        )
        context = self.format_context(documents, max_chars=max_chars)
        include_sources = wants_source_details(question)
        prompt = self.build_prompt(question, context, include_sources=include_sources)

        llm_started = time.perf_counter()
        response = self.llm_client.generate(prompt)
        llm_ms = (time.perf_counter() - llm_started) * 1000

        text = self.finalize_response(response, documents, include_sources)
        reasoning = getattr(self.llm_client, "last_reasoning", None)
        structured_sources = self.extract_sources(documents)

        result = AnswerResult(
            text=text,
            route="rag",
            model_name=self.model_name,
            reasoning=reasoning,
            sources=structured_sources,
            timing=QueryTiming(
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            ),
        )
        self.last_result = result
        return result

    def answer_stream(self, question: str):
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        documents = self.retriever.invoke(question)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        max_chars = (
            RAG_MAX_CONTEXT_CHARS
            if get_backend_for_model(self.model_name) == "proxy"
            else 800
        )
        context = self.format_context(documents, max_chars=max_chars)
        include_sources = wants_source_details(question)
        prompt = self.build_prompt(question, context, include_sources=include_sources)

        chunks = []
        llm_started = time.perf_counter()
        for chunk in self.llm_client.generate_stream(prompt):
            chunks.append(chunk)
            yield chunk
        llm_ms = (time.perf_counter() - llm_started) * 1000

        suffix = ""
        response = "".join(chunks)
        if include_sources and "来源：" not in response and "相关文件：" not in response:
            suffix = f"\n\n{self.format_sources(documents)}"
            yield suffix

        text = response + suffix
        if not text.lstrip().startswith("总结："):
            text = f"总结：基于当前检索到的论文片段生成回答。\n\n{text}"

        reasoning = getattr(self.llm_client, "last_reasoning", None)
        structured_sources = self.extract_sources(documents)

        self.last_result = AnswerResult(
            text=text,
            route="rag",
            model_name=self.model_name,
            reasoning=reasoning,
            sources=structured_sources,
            timing=QueryTiming(
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            ),
        )

    def build_prompt(self, question, context, include_sources=False):
        source_rule = (
            "用户要求展示来源，请在回答末尾加入“来源：”，每条只列 PDF 文件名、页码和一句依据。"
            if include_sources
            else "不要在回答末尾展开文件清单或来源列表；如需引用，只在正文中点到文件名即可。"
        )
        shared_rules = """硬性规则：
1. 只根据“论文片段”回答，不使用外部知识，不补全片段中没有的信息。
2. 不输出思考过程、检索过程、推理草稿、计划步骤或“我需要/首先/接下来”等内部叙述。
3. 如果片段不足以回答，明确写“根据当前片段无法确定”，并说明缺少哪类信息。
4. 中文回答；句子短，直接给结论；不要重复同一篇论文名。
5. Markdown 表格必须简短，单元格用短语，不写长段落。
/no_think"""
        if self.is_compare_question(question):
            template = """你是 PaperAgent，本地论文知识库智能体。任务是基于给定片段做论文对比。
{shared_rules}

输出格式：
总结：一句话说明最核心结论。

对比结论：
| 维度 | 论文 A | 论文 B | 差异总结 |
|---|---|---|---|
| 研究对象 | ... | ... | ... |
| 数据来源 | ... | ... | ... |
| 方法流程 | ... | ... | ... |
| 结论用途 | ... | ... | ... |

来源规则：{source_rule}

论文片段：{context}

问题：{question}
"""
        else:
            template = """你是 PaperAgent，本地论文知识库智能体。任务是基于给定片段回答科研论文问题。
{shared_rules}

输出格式：
总结：一句话说明本次回答的核心结论。

回答：
- 如果问题问作者或论文清单，用项目符号列出论文和文件名。
- 如果问题问数据来源，按“论文/对象/数据来源/星表或处理方法”组织。
- 如果问题问方法总结，按“观测数据、处理方法、主要结果、可复核来源”组织。
- 其它问题用简洁段落或要点回答。

来源规则：{source_rule}

论文片段：{context}

问题：{question}
"""
        return PromptTemplate.from_template(template).format(
            question=question,
            context=context,
            source_rule=source_rule,
            shared_rules=shared_rules,
        )

    @staticmethod
    def format_context(documents: list, max_chars: int = 2500) -> str:
        blocks = []
        for index, doc in enumerate(documents, start=1):
            filename = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page") or "?"
            content = clean_text(doc.page_content)[:max_chars]
            blocks.append(
                f"[{index}] 文件：{filename}\n页码：P{page}\n原文片段：{content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def extract_sources(documents: list) -> list[dict]:
        sources = []
        for doc in documents:
            filename = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page") or "?"
            snippet = " ".join(clean_text(doc.page_content).split())[:400]
            sources.append({
                "filename": filename,
                "page": page,
                "snippet": snippet,
            })
        return sources

    @staticmethod
    def format_sources(documents: list) -> str:
        lines = ["来源："]
        for index, doc in enumerate(documents, start=1):
            filename = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page") or "?"
            snippet = " ".join(clean_text(doc.page_content).split())[:600]
            lines.append(
                f"{index}. 文件：{filename}\n"
                f"   页码：P{page}\n"
                f"   原文片段：{snippet}"
            )
        return "\n".join(lines)

    @staticmethod
    def is_compare_question(question):
        keywords = [
            "比较",
            "对比",
            "差异",
            "相同点",
            "不同点",
            "方法差异",
            "数据来源差异",
        ]
        return any(keyword in question for keyword in keywords)


def clean_text(text):
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        char
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
        else " "
        for char in normalized
    )


def wants_source_details(question):
    keywords = [
        "来源",
        "文件",
        "pdf",
        "PDF",
        "列出",
        "展示",
        "哪些论文",
        "哪几篇论文",
        "哪些文章",
        "哪几篇文章",
        "文章有哪些",
        "论文有哪些",
        "找到哪些",
    ]
    return any(keyword in question for keyword in keywords)
