"""Fast in-memory BM25 + dense reciprocal-rank fusion retriever."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    latin = re.findall(r"[a-z0-9]+(?:[.+-][a-z0-9]+)*", normalized)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    cjk = []
    for run in cjk_runs:
        if len(run) <= 8:
            cjk.append(run)
        cjk.extend(run[index : index + 2] for index in range(len(run) - 1))
    return latin + cjk


ASTRO_DOMAIN_MAPPINGS = {
    "海卫一": ["triton", "neptune i"],
    "木卫六": ["himalia", "jupiter vi"],
    "木卫七": ["elara", "jupiter vii"],
    "木卫八": ["pasiphae"],
    "天王星卫星": ["uranian satellites", "miranda", "ariel", "umbriel", "titania", "oberon"],
    "海王星卫星": ["neptunian satellites", "triton", "nereid", "proteus"],
    "木星卫星": ["jovian satellites", "jupiter", "galilean"],
    "不规则卫星": ["irregular satellites"],
    "佘山": ["sheshan", "1.56m", "1.56-m", "shao"],
    "星表": ["gaia dr2", "gaia edr3", "gaia dr3", "ucac4"],
    "残差": ["residuals", "o-c", "rms"],
    "历表": ["ephemeris", "ephemerides", "de430", "de440"],
    "测角精度": ["astrometric precision", "accuracy", "positional residuals"],
    "张会彦": ["zhang", "huiyan", "hui-yan"],
    "乔荣川": ["qiao", "rongchuan", "rong-chuan"],
}


def expand_query_domain_terms(query: str) -> str:
    expanded = [query]
    query_lower = query.lower()
    for zh_term, en_terms in ASTRO_DOMAIN_MAPPINGS.items():
        if zh_term.lower() in query_lower:
            expanded.extend(en_terms)
    return " ".join(expanded)


class BM25Index:
    def __init__(self, documents, k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(doc.page_content)) for doc in self.documents]
        self.lengths = [sum(freq.values()) for freq in self.term_frequencies]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequency.update(frequencies.keys())

    def search(self, query: str, k: int) -> list:
        query_terms = Counter(tokenize(query))
        scored = []
        total = len(self.documents)
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length_norm = 1 - self.b + self.b * self.lengths[index] / max(self.average_length, 1)
            for term, query_count in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                frequency_docs = self.document_frequency[term]
                inverse_frequency = math.log(1 + (total - frequency_docs + 0.5) / (frequency_docs + 0.5))
                score += query_count * inverse_frequency * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
                )
            if score:
                scored.append((score, index))
        scored.sort(reverse=True)
        return [self.documents[index] for _, index in scored[:k]]


class HybridRetriever:
    """Fuse semantic and exact-term retrieval while limiting duplicate passages."""

    def __init__(self, vectorstore, documents, k: int = 7, fetch_k: int = 24):
        self.vectorstore = vectorstore
        self.documents = list(documents)
        self.k = k
        self.fetch_k = fetch_k
        self.lexical = BM25Index(self.documents)

    def invoke(self, query: str):
        explicit_source = find_explicit_source(query, self.documents)
        if explicit_source:
            dense = self.vectorstore.similarity_search(
                query,
                k=self.fetch_k,
                filter={"filename": explicit_source},
                fetch_k=len(self.documents),
            )
            lexical = [
                document
                for document in self.lexical.search(query, k=len(self.documents))
                if document.metadata.get("filename") == explicit_source
            ][: self.fetch_k]
        else:
            expanded_query = expand_query_domain_terms(query)
            dense = self.vectorstore.similarity_search(query, k=self.fetch_k)
            lexical = self.lexical.search(expanded_query, k=self.fetch_k)
        fused = defaultdict(float)
        by_key = {}
        # Dense retrieval remains primary for cross-lingual questions. BM25 is a
        # precision boost for exact paper names, catalog names, years, and acronyms.
        for weight, results in ((1.0, dense), (0.25, lexical)):
            for rank, document in enumerate(results, start=1):
                key = document_key(document)
                by_key[key] = document
                fused[key] += weight / (10 + rank)

        if explicit_source:
            for key, document in by_key.items():
                fused[key] += section_intent_boost(query, document.metadata.get("section_title", ""))
                fused[key] += content_quality_adjustment(query, document)

        ordered = sorted(fused, key=lambda key: fused[key], reverse=True)
        # Preserve the strongest multilingual semantic hit. Lexical fusion then
        # improves the remaining recall without letting partial CJK bigram
        # matches displace a high-confidence cross-language result.
        selected = [] if explicit_source else dense[:1]
        if explicit_source:
            section_groups = requested_section_groups(query)
            for group_index, section_names in enumerate(section_groups):
                quota = 1 if group_index == 0 and "abstract" in section_names else 2
                focused = self.vectorstore.similarity_search(
                    section_query_hint(section_names),
                    k=self.fetch_k,
                    filter={"filename": explicit_source},
                    fetch_k=len(self.documents),
                )
                candidates = (
                    document
                    for document in focused
                    if section_matches(document, section_names)
                    and document not in selected
                )
                for candidate in candidates:
                    selected.append(candidate)
                    if sum(section_matches(item, section_names) for item in selected) >= quota:
                        break
                if len(selected) >= self.k:
                    break
        per_source = Counter()
        per_section = Counter()
        for document in selected:
            per_source[document.metadata.get("filename", "unknown")] += 1
            per_section[document.metadata.get("section_title", "全文")] += 1
        for key in ordered:
            document = by_key[key]
            if any(document_key(item) == key for item in selected):
                continue
            source = document.metadata.get("filename", "unknown")
            source_limit = self.k if explicit_source else 2
            if per_source[source] >= source_limit:
                continue
            section = document.metadata.get("section_title", "全文")
            if explicit_source and len(requested_section_groups(query)) > 1 and per_section[section] >= 2:
                continue
            selected.append(document)
            per_source[source] += 1
            per_section[section] += 1
            if len(selected) == self.k:
                break
        return selected


def document_key(document) -> tuple:
    return (
        document.metadata.get("filename"),
        document.metadata.get("chunk_id"),
        document.page_content[:80],
    )


def normalize_match(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", unicodedata.normalize("NFKC", text).lower())


def find_explicit_source(query: str, documents) -> str | None:
    """Detect an explicitly named PDF and prevent cross-paper contamination."""

    normalized_query = normalize_match(query)
    filenames = {doc.metadata.get("filename") for doc in documents}
    query_terms = meaningful_latin_terms(query)
    query_years = set(re.findall(r"(?:19|20)\d{2}", query))
    matches = []
    for filename in filenames:
        if not filename:
            continue
        normalized_stem = normalize_match(Path(filename).stem)
        if len(normalized_stem) >= 16 and normalized_stem in normalized_query:
            matches.append((10.0 + len(normalized_stem) / 1000, filename))
            continue

        source_terms = meaningful_latin_terms(Path(filename).stem)
        overlap = len(query_terms & source_terms)
        coverage = overlap / max(len(query_terms), 1)
        source_years = set(re.findall(r"(?:19|20)\d{2}", filename))
        year_bonus = 0.4 if query_years and query_years & source_years else 0.0
        year_penalty = 0.5 if query_years and source_years and not query_years & source_years else 0.0
        if overlap >= 4 and coverage >= 0.65:
            matches.append((coverage + overlap / 20 + year_bonus - year_penalty, filename))
    if not matches:
        return None
    matches.sort(reverse=True)
    if len(matches) > 1 and matches[0][0] - matches[1][0] < 0.15:
        return None
    return matches[0][1]


def meaningful_latin_terms(text: str) -> set[str]:
    stop_words = {
        "the", "of", "and", "in", "on", "based", "during", "with", "from",
        "paper", "article", "aj", "apjs", "mnras", "icarus", "raa", "aas",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())
        if len(token) >= 3 and token not in stop_words
    }


def section_intent_boost(query: str, section_title: str) -> float:
    normalized_section = normalize_match(section_leaf(section_title))
    boost = 0.0
    for names in requested_section_groups(query):
        if any(normalize_match(name) in normalized_section for name in names):
            boost = max(boost, 2.0)
    return boost


def requested_section_groups(query: str) -> list[tuple[str, ...]]:
    groups = []
    if any(marker in query for marker in ("总结", "摘要", "概述")):
        groups.append(("abstract", "摘要"))
    if any(marker in query for marker in ("背景", "动机", "为什么", "研究意义")):
        groups.append(("abstract", "摘要", "introduction", "引言"))
    if any(marker in query for marker in ("观测数据", "观测资料", "观测", "数据来源")):
        groups.append(("observation", "observations", "观测"))
    if any(marker in query for marker in ("处理方法", "方法", "流程", "怎么")):
        groups.append(("method", "measurement", "reduction", "preprocessing", "方法", "处理", "归算"))
    if "结论" in query:
        groups.append(("conclusion", "结论"))
    elif any(marker in query for marker in ("结果", "发现")):
        groups.append(("results", "comparison", "conclusion", "结果", "对比", "结论"))
    return groups


def section_matches(document, section_names: tuple[str, ...]) -> bool:
    section = normalize_match(section_leaf(document.metadata.get("section_title", "")))
    return any(normalize_match(name) in section for name in section_names)


def section_query_hint(section_names: tuple[str, ...]) -> str:
    names = set(section_names)
    if "abstract" in names:
        return "摘要 核心内容 abstract"
    if "observation" in names:
        return "观测数据 望远镜 时间 数量 星表 observation telescope positions catalog"
    if "method" in names:
        return "处理方法 图像预处理 中值滤波 质心测定 天体测量归算 image preprocessing centroiding reduction"
    if "conclusion" in names:
        return "主要结果 数值 结论 意义 conclusion results findings"
    return " ".join(section_names)


def section_leaf(section_title: str) -> str:
    if "›" not in section_title:
        return ""
    return section_title.rsplit("›", 1)[-1].strip()


def content_quality_adjustment(query: str, document) -> float:
    """Down-rank boilerplate and raw table fragments for narrative questions."""

    content = document.metadata.get("content") or document.page_content
    normalized = " ".join(content.lower().split())
    penalty = 0.0
    if "content of this paper is arranged" in normalized or "paper is organized as follows" in normalized:
        penalty -= 1.0
    narrative_query = any(marker in query for marker in ("背景", "动机", "概述", "为什么", "研究意义"))
    if narrative_query and content.count("|") >= 8:
        penalty -= 0.75
    return penalty
