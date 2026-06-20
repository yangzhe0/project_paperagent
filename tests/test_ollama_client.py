import unittest

from paperagent.ollama_client import OllamaClient, parse_ollama_stream_lines


class OllamaClientTest(unittest.TestCase):
    def test_parse_stream_lines_yields_response_chunks(self):
        lines = [
            '{"response":"你","done":false}\n'.encode("utf-8"),
            '{"response":"好","done":false}\n'.encode("utf-8"),
            b'{"done":true}\n',
        ]

        self.assertEqual(list(parse_ollama_stream_lines(lines)), ["你", "好"])

    def test_parse_stream_lines_raises_on_error_payload(self):
        with self.assertRaisesRegex(RuntimeError, "model not found"):
            list(parse_ollama_stream_lines([b'{"error":"model not found"}\n']))

    def test_build_payload_sets_stream_flag(self):
        client = OllamaClient(model_name="qwen2.5:7b-instruct", num_ctx=2048, timeout=30)

        payload = client.build_payload("hello", stream=True)

        self.assertEqual(payload["model"], "qwen2.5:7b-instruct")
        self.assertEqual(payload["prompt"], "hello")
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 2048)

    def test_post_uses_injected_opener(self):
        class FakeResponse:
            def read(self):
                return b'{"response":"ok"}'

        class FakeOpener:
            def __init__(self):
                self.called = False

            def open(self, req, timeout):
                self.called = True
                self.timeout = timeout
                return FakeResponse()

        opener = FakeOpener()
        client = OllamaClient(model_name="qwen2.5:7b-instruct", timeout=12, opener=opener)

        self.assertEqual(client.generate("hello"), "ok")
        self.assertTrue(opener.called)
        self.assertEqual(opener.timeout, 12)


if __name__ == "__main__":
    unittest.main()
