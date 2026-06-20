import unittest

from paperagent.engine import PaperAgentEngine
from paperagent.paper_index import PaperIndex
from paperagent.rag_chain import RAGChain


class FakeRAGChain:
    def __init__(self):
        self.called = False

    def answer_with_trace(self, question):
        self.called = True
        raise AssertionError("RAG should not be called for deterministic fast paths")


class EngineFastPathsTest(unittest.TestCase):
    def test_author_question_returns_answer_result_without_rag(self):
        engine = PaperAgentEngine()
        engine.rag_chain = FakeRAGChain()
        engine.paper_index = PaperIndex.__new__(PaperIndex)
        engine.paper_index.records = [
            {
                "filename": "demo.pdf",
                "title": "Demo Paper",
                "first_pages_text": "张会彦 Huiyan Zhang",
            }
        ]

        result = engine.answer_with_trace("张会彦是谁，文章有哪些？")

        self.assertEqual(result.route, "author")
        self.assertIn("Demo Paper", result.text)
        self.assertFalse(engine.rag_chain.called)

    def test_keyword_paper_list_question_uses_paper_index(self):
        index = PaperIndex.__new__(PaperIndex)
        index.records = [
            {
                "filename": "gaia.pdf",
                "title": "Gaia DR2 astrometry",
                "first_pages_text": "This paper uses Gaia DR2 reference catalog.",
            },
            {
                "filename": "other.pdf",
                "title": "Unrelated",
                "first_pages_text": "No matching data source.",
            },
        ]

        matches = index.search_by_keywords("哪些论文使用了 Gaia DR2？", limit=5)

        self.assertEqual([record["filename"] for record in matches], ["gaia.pdf"])

    def test_summary_question_with_source_request_does_not_use_keyword_list_route(self):
        engine = PaperAgentEngine()
        engine.rag_chain = object()
        engine.paper_index = PaperIndex.__new__(PaperIndex)
        engine.paper_index.records = [
            {
                "filename": "gaia.pdf",
                "title": "Gaia DR2 astrometry",
                "first_pages_text": "This paper uses Gaia DR2 reference catalog.",
            }
        ]

        result = engine.answer_with_trace_for_streamable_routes(
            "总结 2021_AJ_New Positions of Triton Based on Gaia DR2 这篇论文的观测数据、处理方法和结论，请列出来源。"
        )

        self.assertIsNone(result)

    def test_set_model_updates_existing_rag_chain_without_reinitializing_index(self):
        engine = PaperAgentEngine("qwen2.5:7b-instruct")
        engine.rag_chain = RAGChain(retriever=object(), model_name="qwen2.5:7b-instruct")

        engine.set_model("qwen3:30b")

        self.assertEqual(engine.model_name, "qwen3:30b")
        self.assertEqual(engine.rag_chain.model_name, "qwen3:30b")
        self.assertEqual(engine.rag_chain.ollama_client.model_name, "qwen3:30b")

    def test_target_data_source_compare_uses_split_index_route(self):
        engine = PaperAgentEngine()
        engine.rag_chain = FakeRAGChain()
        engine.paper_index = PaperIndex.__new__(PaperIndex)
        engine.paper_index.records = [
            {
                "filename": "2022_RAA_Precise Positions of Triton in 2010-2014 based on Gaia-DR2.pdf",
                "title": "Precise Positions of Triton in 2010-2014 based on Gaia-DR2",
                "first_pages_text": (
                    "Triton was observed by the 1.56 m telescope of Shanghai Astronomical Observatory. "
                    "604 positions of Triton during 2010-2014 are calculated with Gaia DR2."
                ),
            },
            {
                "filename": "2019_PSS New precise astrometric positions of Himalia in 2016-2018 based on.pdf",
                "title": "New precise astrometric positions of Himalia in 2016-2018 based on Gaia DR2",
                "first_pages_text": (
                    "267 new observed positions of Himalia in the period 2016-2018 were collected "
                    "by the 1.0 m telescopes at Yunnan Astronomical Observatory. Gaia DR2 and image enhancement."
                ),
            },
        ]

        result = engine.answer_with_trace("对比 Triton 和 Himalia 相关论文的数据来源差异。")

        self.assertEqual(result.route, "target_compare")
        self.assertIn("Triton", result.text)
        self.assertIn("Himalia", result.text)
        self.assertIn("Gaia DR2", result.text)
        self.assertFalse(engine.rag_chain.called)


if __name__ == "__main__":
    unittest.main()
