# PaperAgent

[简体中文](README.md) | **English**

An intelligent research paper agent tailored for celestial astrometry and planetary ephemerides, featuring hybrid routing, deep reasoning chains, paper-level citations, and a responsive web interface.

[Live Demo](https://paper.sues.fun) · Streamlit · FAISS · BGE-M3 · MinerU · Frontier Reasoning Engine

**Author**: [Zhe Yang (杨哲)](https://zheoi.com)

---

## Interface Preview

![PaperAgent Desktop UI](assets/screenshots/desktop-chat.png)

<p align="center">
  <img src="assets/screenshots/mobile-chat.png" alt="PaperAgent Mobile UI" width="390">
</p>

---

## Overview

**PaperAgent** is designed for deep-space astrometry, irregular satellite orbit ephemerides, and Gaia stellar catalog systematic error reductions.

It leverages a **Hybrid Query Router**:
1. **Deterministic Fast Path**: Structured queries regarding paper catalogs, dataset origins, and satellite comparisons hit deterministic metadata indexes directly with sub-10ms latency.
2. **Academic RAG Chain**: In-depth mechanism queries, astrometric residuals, and observation methods route to the enhanced retrieval-augmented generation pipeline.

---

## Key Features

- **Hybrid Routing**: Fast deterministic answers for metadata queries; contextual RAG for open academic questions.
- **High-Precision Knowledge Base**: MinerU formula/table extraction, 1,600-character hierarchical chunking, BGE-M3 embeddings, and FAISS indexing.
- **Frontier Reasoning Kernels**:
  - ⚡ **Flash**: Ultra-long context window and rapid response.
  - 🧠 **Thinking**: Deep chain-of-thought derivation for astrometric residuals and observation models.
  - 🔬 **Pro**: Long-document synthesis and cross-paper literature reviews.
- **Transparent Citations**: Every claim is mapped to paper titles, page numbers, and original snippets.
- **Modern Responsive Design**: Fully responsive across desktop widescreen and mobile devices (iPhone 390px viewport).

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py --server.port 8000
```

---

## Testing

```bash
python -m unittest discover -s tests
```

---

## Author & Contact

- **Author**: Zhe Yang (杨哲)
- **Website**: [zheoi.com](https://zheoi.com)

---

## License

[Apache License 2.0](LICENSE)
