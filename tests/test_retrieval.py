import unittest

from langchain_core.documents import Document

from paperagent.retrieval.chunking import clean_markdown, split_markdown
from paperagent.retrieval.hybrid import (
    BM25Index,
    HybridRetriever,
    content_quality_adjustment,
    find_explicit_source,
    requested_section_groups,
    section_query_hint,
    section_matches,
    section_intent_boost,
)


class ChunkingTest(unittest.TestCase):
    def test_clean_markdown_removes_images_and_flattens_tables(self):
        cleaned = clean_markdown(
            "# Title\n\n![](images/noise.jpg)\n\n"
            "<table><tr><td>Gaia DR3</td><td>2024</td></tr></table>"
        )

        self.assertNotIn("images/noise.jpg", cleaned)
        self.assertNotIn("<td>", cleaned)
        self.assertIn("Gaia DR3", cleaned)

    def test_split_markdown_keeps_section_hierarchy_and_skips_references(self):
        text = "# Demo\n\nAbstract text.\n\n## 1. Method\n\nCCD method details.\n\n## References\n\nIgnored citation."

        chunks = split_markdown(text, chunk_size=80, chunk_overlap=10)

        self.assertTrue(any("Demo › 1. Method" == chunk["section_title"] for chunk in chunks))
        self.assertFalse(any("Ignored citation" in chunk["text"] for chunk in chunks))


class HybridRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.target = Document(
            page_content="论文：Uranian satellites\nGaia DR3 observations during 2013-2020",
            metadata={"filename": "target.pdf", "chunk_id": 1},
        )
        self.other = Document(
            page_content="论文：Saturn satellites\nCCD observations during 2004-2014",
            metadata={"filename": "other.pdf", "chunk_id": 1},
        )

    def test_bm25_recovers_exact_scientific_terms(self):
        index = BM25Index([self.other, self.target])

        self.assertEqual(index.search("Uranian Gaia DR3", 1)[0], self.target)

    def test_hybrid_fusion_preserves_dense_anchor_and_adds_exact_match(self):
        class DenseStore:
            def __init__(self, documents):
                self.documents = documents

            def similarity_search(self, query, k):
                return self.documents[:k]

        retriever = HybridRetriever(DenseStore([self.other, self.target]), [self.other, self.target], k=2)

        results = retriever.invoke("Uranian Gaia DR3")

        self.assertEqual(results[0], self.other)
        self.assertIn(self.target, results)

    def test_explicit_filename_is_detected(self):
        target = Document(
            page_content="paper",
            metadata={"filename": "2022_Precise_Positions_of_Uranian_Satellites.pdf"},
        )
        query = "2022 Precise Positions of Uranian Satellites 背景信息"

        self.assertEqual(
            find_explicit_source(query, [self.other, target]),
            "2022_Precise_Positions_of_Uranian_Satellites.pdf",
        )

    def test_background_intent_prioritizes_abstract_and_introduction(self):
        self.assertGreater(section_intent_boost("背景信息", "Paper › Abstract"), 0)
        self.assertGreater(section_intent_boost("研究背景", "Paper › 1. Introduction"), 0)
        self.assertEqual(section_intent_boost("研究背景", "Paper › 3. Results"), 0)

    def test_background_intent_downranks_outline_boilerplate(self):
        document = Document(
            page_content="",
            metadata={"content": "The content of this paper is arranged as follows."},
        )

        self.assertLess(content_quality_adjustment("背景信息", document), 0)

    def test_partial_english_title_resolves_unique_paper(self):
        target = Document(
            page_content="paper",
            metadata={
                "filename": "2021_AJ_New Positions of Triton Based on Gaia DR2 and New Image-processing Methods AJ_161_237.pdf"
            },
        )
        distractor = Document(
            page_content="paper",
            metadata={"filename": "2022_RAA_Precise Positions of Triton based on Gaia DR2.pdf"},
        )
        query = "总结 2021_AJ_New Positions of Triton Based on Gaia DR2 这篇论文"

        self.assertEqual(find_explicit_source(query, [target, distractor]), target.metadata["filename"])

    def test_multi_part_summary_requests_section_coverage(self):
        groups = requested_section_groups("总结论文的观测数据、处理方法和结论")

        self.assertEqual(len(groups), 4)

    def test_paper_title_does_not_masquerade_as_method_section(self):
        title_page = Document(
            page_content="authors",
            metadata={"section_title": "A Paper with New Processing Methods"},
        )
        method = Document(
            page_content="method",
            metadata={"section_title": "A Paper with New Processing Methods › 3. Method"},
        )

        self.assertFalse(section_matches(title_page, ("method",)))
        self.assertTrue(section_matches(method, ("method",)))

    def test_method_group_uses_focused_evidence_query(self):
        hint = section_query_hint(("method", "measurement", "reduction"))

        self.assertIn("质心测定", hint)
        self.assertIn("centroiding", hint)


if __name__ == "__main__":
    unittest.main()
