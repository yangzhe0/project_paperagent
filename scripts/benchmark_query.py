#!/usr/bin/env python
import argparse
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperagent.config import DEFAULT_MODEL, MODEL_PROFILES, get_model_name
from paperagent.engine import PaperAgentEngine


DEFAULT_QUESTIONS = [
    "张会彦是谁，文章有哪些？",
    "哪些论文使用了 Gaia DR2？请列出来源。",
    "总结 2021_AJ_New Positions of Triton 这篇论文的观测方法。",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PaperAgent query latency.")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default=None)
    parser.add_argument("--model", default=None, help="Override Ollama model name.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("questions", nargs="*", default=DEFAULT_QUESTIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model or get_model_name(args.profile) or DEFAULT_MODEL

    engine = PaperAgentEngine(model_name)
    load_started = time.perf_counter()
    engine.initialize(rebuild=False)
    load_ms = (time.perf_counter() - load_started) * 1000
    print(f"load_ms={load_ms:.0f} model={model_name}")

    for question in args.questions:
        totals = []
        print(f"\nQ: {question}")
        for run in range(1, args.repeat + 1):
            result = engine.answer_with_trace(question)
            totals.append(result.timing.total_ms)
            print(f"  run={run} {result.diagnostics()}")
        if len(totals) > 1:
            print(f"  mean={statistics.mean(totals):.0f}ms median={statistics.median(totals):.0f}ms")


if __name__ == "__main__":
    main()
