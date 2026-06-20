import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptEntryPointTest(unittest.TestCase):
    def test_benchmark_query_help_runs_when_called_by_path(self):
        result = subprocess.run(
            [sys.executable, "scripts/benchmark_query.py", "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Benchmark PaperAgent query latency", result.stdout)


if __name__ == "__main__":
    unittest.main()
