"""MinerU-aware Markdown cleanup and structure-preserving chunking."""

import html
import re
import unicodedata


DEFAULT_CHUNK_SIZE = 1600
DEFAULT_CHUNK_OVERLAP = 200
SKIPPED_SECTIONS = {
    "references",
    "reference",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "orcid ids",
}


def clean_markdown(text: str) -> str:
    """Remove extraction noise while retaining prose, tables, and equations."""

    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"!\[[^\]]*\]\([^\n)]+\)", "", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:td|th)>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    cleaned = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if re.fullmatch(r"\d{1,4}", line) or re.fullmatch(r"[-_=| ]{3,}", line):
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def split_markdown(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Split by heading and paragraph, avoiding arbitrary mid-sentence cuts."""

    chunks = []
    for section_title, section_text in split_by_heading(text):
        if should_skip_section(section_title):
            continue
        section_chunks = pack_blocks(split_blocks(section_text, chunk_size), chunk_size)
        for index, body in enumerate(section_chunks):
            if index and chunk_overlap:
                overlap = tail_at_boundary(section_chunks[index - 1], chunk_overlap)
                if overlap:
                    body = f"{overlap}\n\n{body}"
            body = body.strip()
            if body:
                chunks.append({"section_title": section_title, "text": body, "page": None})
    return chunks


def split_by_heading(text: str) -> list[tuple[str, str]]:
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [("全文", text)]

    sections = []
    if preface := text[: matches[0].start()].strip():
        sections.append(("题名前内容", preface))
    hierarchy: dict[int, str] = {}
    for index, match in enumerate(matches):
        level = len(match.group(1))
        hierarchy[level] = match.group(2).strip()
        hierarchy = {key: value for key, value in hierarchy.items() if key <= level}
        title = " › ".join(hierarchy[key] for key in sorted(hierarchy))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            sections.append((title, body))
    return sections


def should_skip_section(section_title: str) -> bool:
    leaf = section_title.rsplit("›", 1)[-1].strip().lower()
    leaf = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", leaf)
    return leaf in SKIPPED_SECTIONS


def split_blocks(text: str, chunk_size: int) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    result = []
    for block in blocks:
        if len(block) <= chunk_size * 1.35:
            result.append(block)
        else:
            result.extend(split_at_boundaries(block, chunk_size))
    return result


def split_at_boundaries(text: str, target: int) -> list[str]:
    pieces = []
    remaining = text.strip()
    while len(remaining) > target:
        window = remaining[:target]
        candidates = [match.end() for match in re.finditer(r"[。！？.!?;；]\s*|\n", window)]
        cut = max((pos for pos in candidates if pos >= target * 0.55), default=target)
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def pack_blocks(blocks: list[str], chunk_size: int) -> list[str]:
    chunks = []
    current = ""
    for block in blocks:
        parts = split_at_boundaries(block, chunk_size) if len(block) > chunk_size else [block]
        for part in parts:
            candidate = f"{current}\n\n{part}".strip() if current else part
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = part
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def tail_at_boundary(text: str, limit: int) -> str:
    tail = text[-limit:]
    boundary = re.search(r"[。！？.!?;；]\s*", tail)
    return tail[boundary.end() :].strip() if boundary else tail.strip()
