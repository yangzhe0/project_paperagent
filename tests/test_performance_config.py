import unittest

from paperagent import config
from paperagent.core.types import AnswerResult, QueryTiming


class PerformanceConfigTest(unittest.TestCase):
    def test_flash_profile_is_default_model(self):
        self.assertEqual(config.DEFAULT_MODEL_PROFILE, "flash")
        self.assertEqual(config.DEFAULT_MODEL, config.MODEL_PROFILES["flash"])
        self.assertEqual(config.get_model_name("flash"), config.MODEL_PROFILES["flash"])

    def test_profiles_match_academic_models(self):
        self.assertEqual(config.MODEL_PROFILES["flash"], "gemini-3.8-flash-high")
        self.assertEqual(config.MODEL_PROFILES["thinking"], "claude-sonnet-4-6-thinking")
        self.assertEqual(config.MODEL_PROFILES["pro"], "gemini-3.1-pro")
        self.assertEqual(config.get_model_profile("flash").num_ctx, 128000)
        self.assertEqual(config.get_model_profile("thinking").num_ctx, 128000)
        self.assertEqual(config.get_model_profile("thinking").think, "high")

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
