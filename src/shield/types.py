from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Stage = Literal["S0", "S1", "S2", "S3"]
Action = Literal["pass", "sanitize", "block"]


@dataclass(frozen=True)
class Span:
    """Intervallo di caratteri riferito al testo canonico."""

    start: int
    end: int

    def overlap(self, other: Span) -> int:
        """Numero di caratteri condivisi con un altro intervallo, 0 se disgiunti."""
        return max(0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    position: int  # indice del chunk all'interno del documento
    text: str


@dataclass(frozen=True)
class StageResult:
    stage: Stage
    score: float  # in [0, 1]
    spans: tuple[Span, ...]
    latency_ms: float
    detail: str = ""  # es. nome della regola che ha scattato


@dataclass(frozen=True)
class Verdict:
    action: Action
    score: float
    decided_by: Stage
    canonical_text: str  # testo dopo S0
    clean_text: str  # testo dopo sanitizzazione
    spans: tuple[Span, ...]
    trace: tuple[StageResult, ...]
    total_latency_ms: float

    @property
    def escalated(self) -> bool:
        return any(step.stage == "S3" for step in self.trace)


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    meta: dict[str, str]


@dataclass(frozen=True)
class RagResult:
    query: str
    answer: str
    retrieved: tuple[Document, ...]
    verdicts: tuple[Verdict, ...]
    context_used: str
    retrieval_ms: float
    defense_ms: float
    generation_ms: float


@runtime_checkable
class Classifier(Protocol):
    """Interfaccia di S2: qualunque detector che assegni uno score in [0, 1] a un chunk."""

    def score_one(self, text: str, query: str | None = None) -> float: ...

    def score(self, texts: list[str], queries: list[str] | None = None) -> list[float]: ...


@runtime_checkable
class Judge(Protocol):
    """Interfaccia di S3: verdetto binario su un chunk, True se ritiene sia un'iniezione."""

    def judge(self, chunk_text: str) -> bool: ...
