"""
RAG (Retrieval-Augmented Generation) 问答链模块

流程：
1. 使用向量检索器从知识库中召回相关论文片段
2. 格式化这些片段成提示词上下文
3. 构建对应的系统提示（区分对比和普通问答）
4. 调用本地 Ollama + Qwen 大模型进行回答
5. 在回答后自动附加来源信息（文件名、页码、原文）

设计原则：
- 所有回答必须基于论文片段，不编造
- 答案附带完整的来源追踪
- 对比和普通问答使用不同的 prompt 模板
"""

import time
import unicodedata

from langchain.prompts import PromptTemplate

from paperagent.ollama_client import OllamaClient
from paperagent.performance import AnswerResult, QueryTiming


class RAGChain:
    """
    RAG 问答链 - 负责向量检索、提示词构建和大模型调用

    关键特性：
    - 自动来源追踪（每个回答片段都标注来源）
    - 区分对比和普通问答的不同 prompt 模板
    - 本地化部署（使用 Ollama API 而非 OpenAI）
    """
    def __init__(self, retriever, model_name: str = "qwen2.5:7b-instruct", ollama_client=None):
        """
        初始化 RAG 链

        Args:
            retriever: LangChain 向量检索器（FAISS 封装）
            model_name (str): Ollama 中的模型名称（如 qwen2.5:7b-instruct）
        """
        self.retriever = retriever
        self.model_name = model_name
        self.ollama_client = ollama_client or OllamaClient(model_name)
        self.last_result = None

    def answer(self, question: str) -> str:
        """
        完整的 RAG 问答流程

        流程：
        1. 向量检索 - 从知识库中找到相关的论文片段
        2. 格式化上下文 - 组织片段为结构化的输入
        3. 构建提示词 - 根据问题类型选择对应的 system prompt
        4. 调用大模型 - 通过 Ollama API 获取回答
        5. 格式化输出 - 确保回答包含来源信息和统一格式

        Args:
            question (str): 用户问题

        Returns:
            str: 格式化的完整回答（包含"回答："前缀和来源部分）
        """
        return self.answer_with_trace(question).text

    def answer_with_trace(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        documents = self.retriever.invoke(question)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        context = self.format_context(documents)
        include_sources = wants_source_details(question)
        prompt = self.build_prompt(question, context, include_sources=include_sources)

        llm_started = time.perf_counter()
        response = self.call_ollama(prompt)
        llm_ms = (time.perf_counter() - llm_started) * 1000

        text = self.finalize_response(response, documents, include_sources)
        result = AnswerResult(
            text=text,
            route="rag",
            model_name=self.model_name,
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
        context = self.format_context(documents)
        include_sources = wants_source_details(question)
        prompt = self.build_prompt(question, context, include_sources=include_sources)

        chunks = []
        llm_started = time.perf_counter()
        for chunk in self.ollama_client.generate_stream(prompt):
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
        self.last_result = AnswerResult(
            text=text,
            route="rag",
            model_name=self.model_name,
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
    def format_context(documents: list) -> str:
        """
        将检索到的论文片段格式化为结构化的上下文

        格式示例：
        [1] 文件：2021_AJ_某论文.pdf
        页码：P12
        原文片段：...

        Args:
            documents: LangChain Document 列表

        Returns:
            str: 格式化的上下文字符串
        """
        blocks = []
        for index, doc in enumerate(documents, start=1):
            filename = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page") or "?"
            content = clean_text(doc.page_content)[:800]
            blocks.append(
                f"[{index}] 文件：{filename}\n页码：P{page}\n原文片段：{content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def format_sources(documents: list) -> str:
        """
        将检索结果格式化为带来源的引用信息

        在回答后自动附加，便于追踪论文来源

        Args:
            documents: LangChain Document 列表

        Returns:
            str: 格式化的来源信息（每条包含文件名、页码、片段）
        """
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

    def call_ollama(self, prompt):
        return self.ollama_client.generate(prompt)

    def finalize_response(self, response, documents, include_sources):
        if include_sources and "来源：" not in response and "相关文件：" not in response:
            response = f"{response.rstrip()}\n\n{self.format_sources(documents)}"
        if not response.lstrip().startswith("总结："):
            response = f"总结：基于当前检索到的论文片段生成回答。\n\n{response}"
        return response

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
    """清掉 PDF 解析中常见的不可见控制字符，避免污染 prompt 和来源片段。"""

    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        char
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
        else " "
        for char in normalized
    )


def wants_source_details(question):
    """只有用户明确要论文/文件/来源时，才在界面上展开相关文件列表。"""

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
