import unittest

from paperagent.generation.rag import RAGChain, build_retrieval_query


class RAGPromptTest(unittest.TestCase):
    def test_prompt_contains_output_and_hidden_reasoning_rules(self):
        chain = RAGChain(retriever=object())

        prompt = chain.build_prompt(
            "总结这篇论文的观测方法。",
            "[1] 文件：demo.pdf\n页码：P1\n原文片段：CCD observations with Gaia DR2.",
            include_sources=True,
        )

        self.assertIn("只根据“论文片段”回答", prompt)
        self.assertIn("不输出思考过程", prompt)
        self.assertNotIn("/no_think", prompt)
        self.assertIn("topocentric", prompt)
        self.assertIn("禁止猜测修复", prompt)
        self.assertIn("观测数据、处理方法、主要结果、可复核来源", prompt)
        self.assertIn("来源：", prompt)
        self.assertIn("PDF 文件名、页码和一句依据", prompt)

    def test_short_follow_up_uses_previous_user_question_for_retrieval(self):
        query = build_retrieval_query(
            "它具体用了什么方法？",
            [
                {"role": "user", "content": "总结 Triton 2021 年这篇论文。"},
                {"role": "assistant", "content": "这篇论文重新处理了观测数据。"},
            ],
        )

        self.assertIn("Triton 2021", query)
        self.assertIn("它具体用了什么方法", query)


if __name__ == "__main__":
    unittest.main()
