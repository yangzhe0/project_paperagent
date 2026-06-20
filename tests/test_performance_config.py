import unittest

from paperagent import config
from paperagent.performance import AnswerResult, QueryTiming


class PerformanceConfigTest(unittest.TestCase):
    def test_fast_profile_is_default_model(self):
        self.assertEqual(config.DEFAULT_MODEL_PROFILE, "fast")
        self.assertEqual(config.DEFAULT_MODEL, config.MODEL_PROFILES["fast"])
        self.assertEqual(config.get_model_name("deep"), config.MODEL_PROFILES["deep"])

    def test_unknown_profile_falls_back_to_default(self):
        self.assertEqual(config.get_model_name("missing"), config.DEFAULT_MODEL)

    def test_answer_result_formats_compact_diagnostics(self):
        result = AnswerResult(
            text="总结：ok",
            route="rag",
            model_name="qwen2.5:7b-instruct",
            timing=QueryTiming(retrieval_ms=12.4, llm_ms=250.2, total_ms=300.8),
        )

        self.assertEqual(
            result.diagnostics(),
            "route=rag · model=qwen2.5:7b-instruct · total=301ms · retrieval=12ms · llm=250ms",
        )


if __name__ == "__main__":
    unittest.main()
