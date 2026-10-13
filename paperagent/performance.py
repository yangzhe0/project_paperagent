from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryTiming:
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class AnswerResult:
    text: str
    route: str
    model_name: str
    timing: QueryTiming = field(default_factory=QueryTiming)
    reasoning: Optional[str] = None
    sources: list[dict] = field(default_factory=list)

    def diagnostics(self) -> str:
        return (
            f"route={self.route} · model={self.model_name} · total={self.timing.total_ms:.0f}ms"
            f" · retrieval={self.timing.retrieval_ms:.0f}ms · llm={self.timing.llm_ms:.0f}ms"
        )
