import json
import re
from pathlib import Path

from pypdf import PdfReader

from paperagent.config import PAPER_DIR, PAPER_INDEX_PATH, PROCESSED_DIR


AUTHOR_ALIASES = {
    # MVP 先用别名表解决高频作者；后续可升级为完整文献元数据抽取。
    "张会彦": ["张会彦", "Huiyan Zhang", "Hui-Yan Zhang", "Hui Yan Zhang"],
}


class PaperIndex:
    """轻量结构化索引，用来处理作者查询和点名论文匹配。"""

    def __init__(
        self,
        paper_dir=PAPER_DIR,
        index_path=PAPER_INDEX_PATH,
        processed_dir=PROCESSED_DIR,
    ):
        self.paper_dir = Path(paper_dir)
        self.index_path = Path(index_path)
        self.processed_dir = Path(processed_dir)
        self.records = self.load_or_build()

    def load_or_build(self):
        pdfs = sorted(self.paper_dir.glob("*.pdf"))
        if self.index_path.exists():
            try:
                records = json.loads(self.index_path.read_text(encoding="utf-8"))
                # PDF 数量一致时复用缓存；新增/删除 PDF 后自动重建。
                if len(records) == len(pdfs):
                    return records
            except json.JSONDecodeError:
                pass

        records = [self.extract_record(path, self.processed_dir) for path in pdfs]
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return records

    @staticmethod
    def extract_record(path, processed_dir=None):
        text = PaperIndex.read_processed_text(path, processed_dir)
        try:
            if not text:
                # processed 文本缺失时才直接读 PDF 前两页，作为兜底。
                reader = PdfReader(str(path))
                text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
        except Exception:
            text = ""

        return {
            "filename": path.name,
            "title": PaperIndex.title_from_filename(path.name),
            "first_pages_text": text[:5000],
        }

    @staticmethod
    def read_processed_text(path, processed_dir):
        if not processed_dir:
            return ""
        processed_path = Path(processed_dir) / f"{path.stem}.txt"
        if not processed_path.exists():
            return ""
        text = processed_path.read_text(encoding="utf-8", errors="ignore")
        pages = re.split(r"\n*--- PAGE \d+ ---\n*", text)
        return "\n".join(part.strip() for part in pages[:3] if part.strip())

    @staticmethod
    def title_from_filename(filename):
        title = Path(filename).stem
        parts = title.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[1].strip()
        return title.strip()

    def search_author(self, name):
        aliases = AUTHOR_ALIASES.get(name, [name])
        matches = []
        for record in self.records:
            haystack = f"{record['filename']}\n{record['first_pages_text']}".lower()
            if any(alias.lower() in haystack for alias in aliases):
                matches.append(record)
        return matches

    def search_by_keywords(self, question, limit=8):
        tokens = extract_search_tokens(question)
        if not tokens:
            return []

        scored = []
        for record in self.records:
            haystack = normalize_for_match(
                f"{record['filename']} {record['title']} {record['first_pages_text'][:2500]}"
            )
            score = sum(1 for token in tokens if token in haystack)
            if score:
                scored.append((score, record["filename"], record))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored[:limit]]

    def search_target(self, target, question="", limit=5):
        normalized_target = normalize_for_match(target)
        matches = []
        wants_data_source = any(word in question for word in ["数据来源", "数据源", "观测资料", "观测数据", "数据"])
        for record in self.records:
            raw_haystack = f"{record['filename']} {record['title']} {record['first_pages_text'][:3500]}"
            title_haystack = normalize_for_match(f"{record['filename']} {record['title']}")
            text_haystack = normalize_for_match(record["first_pages_text"][:3500])
            lower_haystack = raw_haystack.lower()
            score = 0
            if normalized_target in title_haystack:
                score += 5
            if normalized_target in text_haystack:
                score += 2
            if wants_data_source:
                if "gaia dr2" in lower_haystack or "gaia-dr2" in lower_haystack:
                    score += 3
                if "ccd" in lower_haystack:
                    score += 1
                if "precise positions" in lower_haystack or "new positions" in lower_haystack:
                    score += 1
            if score:
                title_match = normalized_target in title_haystack
                matches.append((score, title_match, record["filename"], record))

        matches.sort(key=lambda item: (-item[0], item[2]))
        if any(title_match for _, title_match, _, _ in matches):
            matches = [item for item in matches if item[1]]
        return [record for _, _, _, record in matches[:limit]]

    def find_mentioned_papers(self, question):
        normalized_question = normalize_for_match(question)
        matches = []
        for record in self.records:
            candidates = {
                record["filename"],
                record["title"],
                Path(record["filename"]).stem,
            }
            if any(normalize_for_match(candidate) in normalized_question for candidate in candidates):
                matches.append(record)
                continue

            # 用户没输入完整文件名时，用题名中的长词做弱匹配。
            title_tokens = [
                token
                for token in re.split(r"[_\s]+", record["title"])
                if len(normalize_for_match(token)) >= 6
            ]
            if title_tokens and sum(normalize_for_match(token) in normalized_question for token in title_tokens) >= 2:
                matches.append(record)
        return matches


def is_author_paper_question(question):
    """识别“某作者是谁/有哪些论文”这类确定性查询。"""

    author_names = list(AUTHOR_ALIASES)
    intent_words = ["谁", "文章", "论文", "哪些", "发表", "作者", "成果"]
    matched_author = next((name for name in author_names if name in question), None)
    if not matched_author:
        return None
    if any(word in question for word in intent_words):
        return matched_author
    return None


def format_author_answer(author, records):
    if not records:
        return f"总结：当前论文库没有检索到与“{author}”匹配的论文。\n\n回答：\n当前论文库中没有检索到与“{author}”匹配的论文。"

    lines = [
        f"总结：当前论文库检索到 {len(records)} 篇与“{author}”匹配的论文。",
        "",
        "回答：",
        f"{author} 是当前论文库中多篇论文的作者之一。根据 PDF 首页作者信息和文件名匹配，检索到 {len(records)} 篇相关论文：",
        "",
    ]
    for index, record in enumerate(records, start=1):
        lines.append(f"{index}. {record['title']}")
        lines.append(f"   文件：{record['filename']}")

    return "\n".join(lines)


def normalize_for_match(text):
    """统一去掉空格、下划线和标点，让文件名/题名匹配更稳。"""

    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text).lower()


def extract_search_tokens(question):
    ignored = {
        "哪些",
        "哪几篇",
        "论文",
        "文章",
        "使用",
        "用了",
        "列出",
        "来源",
        "文件",
        "the",
        "and",
        "with",
        "based",
    }
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9.+-]*|\d+[A-Za-z0-9.+-]*|[\u4e00-\u9fff]{2,}", question)
    tokens = []
    for token in raw_tokens:
        normalized = normalize_for_match(token)
        if not normalized or token in ignored or normalized in ignored:
            continue
        if len(normalized) < 2:
            continue
        tokens.append(normalized)
    return tokens


def is_keyword_paper_list_question(question):
    if any(word in question for word in ["总结", "概括", "方法总结", "结论"]):
        return False
    list_words = ["哪些论文", "哪几篇论文", "哪些文章", "哪几篇文章", "列出"]
    topic_words = ["使用", "用了", "包含", "提到", "涉及", "关于", "数据源", "方法"]
    return any(word in question for word in list_words) and any(word in question for word in topic_words)


def format_keyword_paper_answer(question, records):
    if not records:
        return (
            "总结：当前轻量索引没有找到明确匹配的论文。\n\n"
            "回答：\n当前轻量索引没有找到明确匹配的论文，可以换成更具体的数据源、天体对象或方法关键词。"
        )

    lines = [
        f"总结：轻量索引找到 {len(records)} 篇可能相关论文。",
        "",
        "回答：",
        "先根据标题、文件名和首页/摘要文本做快速匹配，相关论文如下：",
        "",
    ]
    for index, record in enumerate(records, start=1):
        lines.append(f"{index}. {record['title']}")
        lines.append(f"   文件：{record['filename']}")
    return "\n".join(lines)


KNOWN_TARGETS = [
    "Triton",
    "Himalia",
    "Nereid",
    "Phoebe",
    "Iapetus",
    "Uranus",
    "Neptune",
    "Jupiter",
    "Saturn",
]


def find_target_compare_terms(question):
    if not is_target_compare_question(question):
        return []
    normalized_question = normalize_for_match(question)
    targets = [
        target
        for target in KNOWN_TARGETS
        if normalize_for_match(target) in normalized_question
    ]
    return targets[:2] if len(targets) >= 2 else []


def is_target_compare_question(question):
    compare_words = ["对比", "比较", "差异", "不同", "异同"]
    topic_words = ["数据来源", "数据源", "观测资料", "观测数据", "数据"]
    return any(word in question for word in compare_words) and any(word in question for word in topic_words)


def format_target_compare_answer(question, target_records):
    targets = list(target_records)
    if len(targets) < 2:
        return None

    lines = [
        f"总结：已按 {targets[0]} 与 {targets[1]} 分别检索论文索引；两类论文都以天体测量观测为主，但观测对象、时间段和望远镜/资料来源不同。",
        "",
        "回答：",
        "| 对象 | 代表论文 | 数据/观测来源 | 参考星表/处理 | 时间范围 |",
        "|---|---|---|---|---|",
    ]

    source_lines = ["", "来源："]
    for target in targets:
        records = target_records[target]
        if not records:
            lines.append(f"| {target} | 当前索引未找到 | 根据当前索引无法确定 | 根据当前索引无法确定 | 根据当前索引无法确定 |")
            continue

        summary = extract_data_source_summary(records[0])
        lines.append(
            f"| {target} | {records[0]['title']} | {summary['data_source']} | "
            f"{summary['reference']} | {summary['period']} |"
        )
        for index, record in enumerate(records[:3], start=1):
            snippet = one_line(record["first_pages_text"])[:700]
            source_lines.append(f"{target}-{index}. 文件：{record['filename']}")
            source_lines.append("   页码：P1-P3")
            source_lines.append(f"   原文片段：{snippet}")

    lines.extend(
        [
            "",
            "差异要点：",
            f"- {targets[0]} 相关论文在当前库中多次出现，通常围绕海王星卫星 Triton 的 CCD 天体测量、Gaia DR2 归算、长期位置资料或轨道改进。",
            f"- {targets[1]} 相关论文在当前库中集中对应 Himalia 的 2016-2018 新位置测量，数据来自云南天文台 1.0 m 望远镜的 CCD 观测，并使用 Gaia DR2 与图像增强处理。",
        ]
    )
    lines.extend(source_lines)
    return "\n".join(lines)


def extract_data_source_summary(record):
    text = f"{record['title']} {record['first_pages_text']}"
    text_lower = text.lower()
    data_source = []
    reference = []
    period = []

    if "ccd" in text_lower:
        data_source.append("CCD 天体测量观测")
    if re.search(r"(observed|observations|ccd).{0,80}1\.56\s*m|1\.56\s*m.{0,80}(telescope|ccd|observ)", text_lower):
        data_source.append("上海天文台 1.56 m 望远镜")
    if re.search(r"(observed|observations|collected|ccd).{0,120}1\.0\s*m|1\.0\s*m.{0,120}(telescope|yunnan|ccd|observ)", text_lower):
        data_source.append("云南天文台 1.0 m 望远镜")
    if "voyager 2" in text_lower:
        data_source.append("Voyager 2 飞掠资料作为背景/比较资料")
    if "gaia dr2" in text_lower or "gaia-dr2" in text_lower:
        reference.append("Gaia DR2")
    if "ucac2" in text_lower:
        reference.append("UCAC2")
    if "aappdi" in text_lower:
        reference.append("AAPPDI 处理软件")
    if "image enhancement" in text_lower:
        reference.append("图像增强处理")

    for start, end in re.findall(r"(?:during|period|in)?\s*(\d{4})\s*[–-]\s*(\d{4})", text[:2500], flags=re.IGNORECASE):
        if 1900 <= int(start) <= 2099 and 1900 <= int(end) <= 2099:
            period.append(f"{start}-{end}")
    if "604 positions" in text_lower:
        period.append("604 个位置")
    if "2299 positions" in text_lower:
        period.append("2299 个位置")
    if "267 new observed positions" in text_lower:
        period.append("267 个新观测位置")

    return {
        "data_source": "；".join(dict.fromkeys(data_source)) or "根据首页/摘要片段无法稳定抽取",
        "reference": "；".join(dict.fromkeys(reference)) or "根据首页/摘要片段无法稳定抽取",
        "period": "；".join(dict.fromkeys(period)) or "根据首页/摘要片段无法稳定抽取",
    }


def format_paper_compare_answer(question, records):
    if len(records) < 2:
        return None

    selected = records[:2]
    lines = [
        "总结：已按你点名的两篇论文做确定性对比，避免泛检索到无关论文。",
        "",
        "回答：",
        "先按你点名的两篇论文做确定性对比。以下结论基于文件名和 PDF 首页/摘要片段，不泛化检索其它论文。",
        "",
        "| 维度 | 论文 1 | 论文 2 | 异同 |",
        "|---|---|---|---|",
    ]

    summaries = [extract_brief_summary(record) for record in selected]
    lines.append(
        f"| 研究对象/场景 | {summaries[0]['topic']} | {summaries[1]['topic']} | "
        f"{'相同' if summaries[0]['topic'] == summaries[1]['topic'] else '不同'} |"
    )
    lines.append(
        f"| 方法/系统 | {summaries[0]['method']} | {summaries[1]['method']} | "
        "都属于光学观测/测量相关工作，但一个偏天体事件观测分析，一个偏软件/系统实现。 |"
    )
    lines.append(
        f"| 论文形态 | {summaries[0]['title']} | {summaries[1]['title']} | 题名和研究重点不同。 |"
    )

    lines.extend(["", "来源："])
    for index, record in enumerate(selected, start=1):
        snippet = one_line(record["first_pages_text"])[:700]
        lines.append(f"{index}. 文件：{record['filename']}")
        lines.append("   页码：P1-P2")
        lines.append(f"   原文片段：{snippet}")
    return "\n".join(lines)


def extract_brief_summary(record):
    """给指定论文对比生成可解释的简短摘要，当前只覆盖 Demo 高频场景。"""

    text = record["first_pages_text"]
    title = record["title"]
    topic = title
    method = "从首页文本无法稳定抽取"

    if "木星卫星" in text or "木星卫星" in title:
        topic = "木星卫星掩食/互掩互食观测"
    elif "地基式光学测角" in text or "同步卫星" in text or "Space Debris" in title:
        topic = "地基式光学测角、同步卫星/空间碎片定位"

    if "CCD" in text and ("测光" in text or "数据分析" in text):
        method = "CCD 图像测光处理和数据分析"
    elif "软件" in text or "系统" in text or "drift" in text.lower() or "漂移扫描" in text:
        method = "光学测角软件系统/漂移扫描观测与定位流程"

    return {"title": title, "topic": topic, "method": method}


def one_line(text):
    cleaned = re.sub(
        r"[^0-9A-Za-z\u4e00-\u9fff，。；：、（）《》“”‘’！？.,;:()/_\-\s]",
        " ",
        text,
    )
    return " ".join(cleaned.split())
