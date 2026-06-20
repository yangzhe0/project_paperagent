import unittest

from paperagent.rag_chain import RAGChain


class RAGPromptTest(unittest.TestCase):
    def test_prompt_contains_output_and_no_thinking_rules(self):
        chain = RAGChain(retriever=object())

        prompt = chain.build_prompt(
            "总结这篇论文的观测方法。",
            "[1] 文件：demo.pdf\n页码：P1\n原文片段：CCD observations with Gaia DR2.",
            include_sources=True,
        )

        self.assertIn("只根据“论文片段”回答", prompt)
        self.assertIn("不输出思考过程", prompt)
        self.assertIn("/no_think", prompt)
        self.assertIn("观测数据、处理方法、主要结果、可复核来源", prompt)
        self.assertIn("来源：", prompt)
        self.assertIn("PDF 文件名、页码和一句依据", prompt)


if __name__ == "__main__":
    unittest.main()
