"""
PaperAgent 核心引擎模块

这个模块实现了问题理解和路由的核心逻辑：
1. 低信息量问题拦截 - 避免无意义查询
2. 作者论文查询 - 使用结构化索引快速响应
3. 论文对比 - 点名两篇论文时走对比逻辑
4. 开放式问答 - 其他问题走 RAG 检索链

关键设计：不是所有问题都丢给 RAG，而是通过智能路由选择最合适的答案方式。
"""

import re
import time
from dataclasses import dataclass
from typing import Optional

from paperagent.config import DEFAULT_MODEL, PAPER_DIR, VECTORSTORE_DIR
from paperagent.ingestion import PaperIngestor
from paperagent.paper_index import (
    PaperIndex,
    find_target_compare_terms,
    format_author_answer,
    format_keyword_paper_answer,
    format_paper_compare_answer,
    format_target_compare_answer,
    is_author_paper_question,
    is_keyword_paper_list_question,
)
from paperagent.performance import AnswerResult, QueryTiming
from paperagent.rag_chain import RAGChain
from paperagent.ollama_client import OllamaClient


@dataclass
class AgentStatus:
    """知识库状态信息数据类"""
    paper_count: int              # 已加载的论文 PDF 数量
    has_vectorstore: bool         # 向量库是否就绪
    has_paper_index: bool         # 论文索引是否就绪
    model_name: str               # 当前使用的 LLM 模型名


class PaperAgentEngine:
    """
    PaperAgent 核心智能体类

    职责：
    - 初始化知识库（导入论文、构建向量库、建立索引）
    - 路由用户问题到不同的处理模块
    - 聚合不同模块的回答结果
    """
    def __init__(self, model_name=DEFAULT_MODEL):
        """初始化智能体，设置使用的 LLM 模型"""
        self.model_name = model_name
        self.rag_chain = None           # RAG 问答链（延迟加载）
        self.paper_index = None         # 论文索引（延迟加载）

    def initialize(self, rebuild: bool = False, pdf_files: Optional[list] = None):
        """
        初始化知识库

        Args:
            rebuild (bool): 是否重建向量库（True=重建，False=使用现有）
            pdf_files (list): 不支持在线上传（设计决策：论文需要先用 MinerU 预处理）

        流程：
        1. 从 data/mineru 加载 MinerU 提取的 markdown 文本
        2. 文本分片、Embedding、构建 FAISS 向量库
        3. 从 data/papers 构建论文作者/标题索引
        """
        if pdf_files:
            raise ValueError("Online PDF upload is disabled. Add PDFs to data/papers and parse them with MinerU first.")

        retriever = PaperIngestor(
            paper_dir=PAPER_DIR,
            vectorstore_dir=VECTORSTORE_DIR,
            rebuild=rebuild,
        ).get_retriever()
        self.rag_chain = RAGChain(retriever, model_name=self.model_name)
        self.paper_index = PaperIndex(PAPER_DIR)

    def set_model(self, model_name: str):
        """Switch the LLM used by the already-loaded engine without rebuilding FAISS."""

        self.model_name = model_name
        if self.rag_chain is not None:
            self.rag_chain.model_name = model_name
            self.rag_chain.ollama_client = OllamaClient(model_name)

    def answer(self, question: str) -> str:
        """
        核心问答入口 - 智能路由用户问题

        路由策略（优先级从高到低）：
        1. 低信息量过滤 - 拒绝信息不足的问题，避免无效查询
        2. 作者论文查询 - 使用索引快速检索（"张会彦是谁？" → 结构化答案）
        3. 论文对比 - 用户点名两篇论文时走对比模板（不再随机召回第三篇）
        4. 开放式问答 - 其他问题走向量检索 + LLM 回答（会附带来源）

        Args:
            question (str): 用户输入的问题

        Returns:
            str: 完整的回答（包含原文片段和来源）
        """
        return self.answer_with_trace(question).text

    def answer_with_trace(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        question = question.strip()

        # 策略 1: 拦截低信息量问题，避免无意义检索
        if is_low_information_question(question):
            return self._result(low_information_answer(), "low_information", started)

        if self.paper_index is None:
            self.paper_index = PaperIndex(PAPER_DIR)

        # 策略 2: 检测作者相关问题（例如："张会彦是谁" "张会彦的论文" 等）
        author = is_author_paper_question(question)
        if author:
            records = self.paper_index.search_author(author)
            return self._result(format_author_answer(author, records), "author", started)

        # 策略 3: 检测论文对比（例如："比较 XXX 和 YYY" "XXX 和 YYY 的异同"）
        mentioned_papers = self.paper_index.find_mentioned_papers(question)
        if len(mentioned_papers) >= 2 and is_compare_question(question):
            answer = format_paper_compare_answer(question, mentioned_papers)
            if answer:
                return self._result(answer, "compare", started)

        # 策略 4: 明确要求列论文/文章时，先走轻量结构化索引，避免为清单问题启动生成模型。
        if is_keyword_paper_list_question(question):
            matches = self.paper_index.search_by_keywords(question)
            if matches:
                return self._result(format_keyword_paper_answer(question, matches), "keyword_index", started)

        # 策略 5: 天体对象数据来源对比，按对象拆分检索，避免泛 RAG 只召回一侧。
        target_terms = find_target_compare_terms(question)
        if target_terms:
            target_records = {
                target: self.paper_index.search_target(target, question=question)
                for target in target_terms
            }
            answer = format_target_compare_answer(question, target_records)
            if answer:
                return self._result(answer, "target_compare", started)

        # 策略 6: 开放式问题 → RAG 检索 + LLM 生成
        if self.rag_chain is None:
            return self._result("请先加载论文知识库。", "not_loaded", started)
        return self.rag_chain.answer_with_trace(question)

    def answer_stream(self, question: str):
        result = self.answer_with_trace_for_streamable_routes(question)
        if result is not None:
            yield result.text
            return

        if self.rag_chain is None:
            yield "请先加载论文知识库。"
            return
        yield from self.rag_chain.answer_stream(question.strip())
        self.last_result = self.rag_chain.last_result

    def answer_with_trace_for_streamable_routes(self, question: str):
        question = question.strip()
        if is_low_information_question(question):
            return self._result(low_information_answer(), "low_information", time.perf_counter())

        if self.paper_index is None:
            self.paper_index = PaperIndex(PAPER_DIR)

        author = is_author_paper_question(question)
        if author:
            return self._result(
                format_author_answer(author, self.paper_index.search_author(author)),
                "author",
                time.perf_counter(),
            )

        mentioned_papers = self.paper_index.find_mentioned_papers(question)
        if len(mentioned_papers) >= 2 and is_compare_question(question):
            answer = format_paper_compare_answer(question, mentioned_papers)
            if answer:
                return self._result(answer, "compare", time.perf_counter())

        if is_keyword_paper_list_question(question):
            matches = self.paper_index.search_by_keywords(question)
            if matches:
                return self._result(format_keyword_paper_answer(question, matches), "keyword_index", time.perf_counter())

        target_terms = find_target_compare_terms(question)
        if target_terms:
            target_records = {
                target: self.paper_index.search_target(target, question=question)
                for target in target_terms
            }
            answer = format_target_compare_answer(question, target_records)
            if answer:
                return self._result(answer, "target_compare", time.perf_counter())
        return None

    def _result(self, text: str, route: str, started: float) -> AnswerResult:
        self.last_result = AnswerResult(
            text=text,
            route=route,
            model_name=self.model_name,
            timing=QueryTiming(total_ms=(time.perf_counter() - started) * 1000),
        )
        return self.last_result

    @staticmethod
    def status(model_name=DEFAULT_MODEL) -> AgentStatus:
        """
        检查知识库加载状态

        Returns:
            AgentStatus: 包含论文数、向量库状态、索引状态、模型名的状态对象
        """
        return AgentStatus(
            paper_count=len(list(PAPER_DIR.glob("*.pdf"))),
            has_vectorstore=(VECTORSTORE_DIR / "index.faiss").exists()
            and (VECTORSTORE_DIR / "index.pkl").exists(),
            has_paper_index=(VECTORSTORE_DIR / "paper_index.json").exists(),
            model_name=model_name,
        )


# ============ 问题分类和过滤函数 ============

def is_low_information_question(question: str) -> bool:
    """
    检测问题信息量是否过低

    规则：
    - 少于 4 个字符 → 直接拒绝
    - 中英文字符少于 3 个 → 拒绝（例如："？" "嗯" "啊"）

    目的：避免对 "？" 这样的无意义输入进行 RAG 检索
    """
    normalized = question.strip()
    if len(normalized) < 4:
        return True

    # 统计中文和英文字符数
    letters_or_cjk = re.findall(r"[A-Za-z\u4e00-\u9fff]", normalized)
    return len(letters_or_cjk) < 3


def is_compare_question(question: str) -> bool:
    """
    检测问题是否是论文对比问题

    关键词列表：
    - 比较、对比、差异、相同点、不同点

    用途：当用户问两篇论文的异同时，触发对比模板而不是通用 RAG
    """
    keywords = ["异同", "比较", "对比", "差异", "相同点", "不同点", "方法差异", "数据来源差异"]
    return any(keyword in question for keyword in keywords)


def low_information_answer() -> str:
    """
    对低信息量问题的友好提示

    返回：使用示例，引导用户提出有效问题
    """
    return (
        "这个问题信息量太少，我不会强行检索论文。\n\n"
        "可以试试：\n"
        "- 张会彦是谁，文章有哪些？\n"
        "- 哪些论文使用了 Gaia DR2？\n"
        "- 对比两篇指定论文有什么异同？\n"
        "- 总结 2021_AJ 这篇论文的观测方法。"
    )
