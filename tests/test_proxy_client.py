import json
import unittest

from paperagent.proxy_client import ProxyClient


class ProxyClientTest(unittest.TestCase):
    def test_build_payload_includes_model_and_messages(self):
        client = ProxyClient(model_name="gemini-3.8-flash-high", api_key="test-key")
        payload = client.build_payload(
            prompt="Hello", system_prompt="System instruction", stream=True
        )

        self.assertEqual(payload["model"], "gemini-3.8-flash-high")
        self.assertTrue(payload["stream"])
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "System instruction")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "Hello")

    def test_generate_parses_content_and_reasoning(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self):
                data = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "2",
                                "reasoning_content": "1+1 equals 2",
                            }
                        }
                    ],
                    "usage": {"total_tokens": 15},
                }
                return json.dumps(data).encode("utf-8")

        class FakeOpener:
            def open(self, req, timeout):
                return FakeResponse()

        client = ProxyClient(model_name="gemini-3.8-flash-high", opener=FakeOpener())
        content = client.generate("1+1=?")

        self.assertEqual(content, "2")
        self.assertEqual(client.last_reasoning, "1+1 equals 2")
        self.assertEqual(client.last_usage.get("total_tokens"), 15)

    def test_generate_stream_yields_chunks_and_collects_reasoning(self):
        class FakeStreamResponse:
            def __init__(self):
                self.lines = [
                    b': ping\n',
                    b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":" World"}}]}\n',
                    b'data: [DONE]\n',
                ]
                self.idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.idx < len(self.lines):
                    line = self.lines[self.idx]
                    self.idx += 1
                    return line
                raise StopIteration

            def close(self):
                pass

        class FakeOpener:
            def open(self, req, timeout):
                return FakeStreamResponse()

        client = ProxyClient(model_name="gemini-3.8-flash-high", opener=FakeOpener())
        chunks = list(client.generate_stream("Hi"))

        self.assertEqual(chunks, ["Hello", " World"])
        self.assertEqual(client.last_reasoning, "thinking...")


if __name__ == "__main__":
    unittest.main()
