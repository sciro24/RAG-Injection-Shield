from __future__ import annotations

import re
import time

from .config import CascadeConfig
from .normalize import canonicalize
from .rules import score_rules
from .types import Action, Classifier, Judge, Span, StageResult, Verdict

# Segmentazione grossolana per la localizzazione: righe e frasi. Serve solo a dare
# uno span da rimuovere quando a decidere sono S2 o S3, che uno span non lo producono.
_SEGMENT_RE = re.compile(r"[^\n.!?]+[\n.!?]*")
_MIN_SEGMENT_CHARS = 12


def merge_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    """Fonde gli intervalli sovrapposti o adiacenti, in ordine crescente."""
    if not spans:
        return ()
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end:
            merged[-1] = Span(last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def _boundary_left(text: str, index: int, terminators: str) -> int:
    return max((text.rfind(char, 0, index) for char in terminators), default=-1) + 1


def _boundary_right(text: str, index: int, terminators: str) -> int:
    found = [p for p in (text.find(char, index) for char in terminators) if p != -1]
    return min(found) + 1 if found else len(text)


def _anchor(text: str, span: Span) -> int:
    """Indice dell'ultimo carattere non spazio dello span.

    Cercare il confine destro a partire da `span.end` sbaglia quando lo span finisce
    gia' su un terminatore: la ricerca lo scavalca e aggancia la frase successiva."""
    end = span.end
    while end > span.start and text[end - 1].isspace():
        end -= 1
    return max(span.start, end - 1)


def expand_spans(
    text: str, spans: tuple[Span, ...], max_line_chars: int = 400, max_fraction: float = 0.8
) -> tuple[Span, ...]:
    """Estende ogni span ai confini della riga che lo contiene.

    Un regex aggancia la frase-esca ("ignore all previous instructions"), ma l'istruzione
    iniettata e' l'intera riga: rimuovere la sola esca lascia il payload nel contesto e
    l'LLM lo esegue comunque.

    Si ripiega sui confini di frase quando la riga e' molto lunga, oppure quando coincide
    di fatto con l'intero documento: li' l'estensione alla riga cancellerebbe tutto e
    farebbe degradare a `block` anche un testo in cui l'iniezione e' una frase sola."""
    expanded: list[Span] = []
    for span in spans:
        anchor = _anchor(text, span)
        start = _boundary_left(text, span.start, "\n")
        end = _boundary_right(text, anchor, "\n")
        too_long = end - start > max_line_chars
        eats_everything = end - start > max_fraction * max(1, len(text))
        if too_long or eats_everything:
            start = _boundary_left(text, span.start, ".!?\n")
            end = _boundary_right(text, anchor, ".!?\n")
        expanded.append(Span(start, min(max(end, span.end), len(text))))
    return merge_spans(tuple(expanded))


def sanitize(
    text: str, spans: tuple[Span, ...], marker: str, min_remaining_ratio: float
) -> tuple[str, Action]:
    """Sostituisce gli span col marcatore; degrada a block se resta troppo poco testo."""
    if not spans:
        return "", "block"
    pieces: list[str] = []
    cursor = 0
    for span in merge_spans(spans):
        pieces.append(text[cursor : span.start])
        pieces.append(marker)
        cursor = max(cursor, span.end)
    pieces.append(text[cursor:])
    cleaned = "".join(pieces)
    kept = len(cleaned) - marker.__len__() * len(merge_spans(spans))
    if not text or kept / max(1, len(text)) < min_remaining_ratio:
        return "", "block"
    return cleaned, "sanitize"


class Cascade:
    """Orchestrazione S0-S3 e sanitizzazione. Con `classifier`/`judge` a None si
    ottengono le configurazioni sole-regole e senza-escalation dell'ablation."""

    def __init__(
        self, config: CascadeConfig, classifier: Classifier | None, judge: Judge | None
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.judge = judge

    def inspect(self, chunk_text: str, query: str | None = None) -> Verdict:
        started = time.perf_counter()
        trace: list[StageResult] = []

        canonical = self._run_s0(chunk_text, trace)
        rules = self._run_s1(canonical, trace)
        if rules.score >= self.config.rules_block_threshold:
            return self._decide(canonical, rules.score, "S1", rules.spans, trace, started)

        if self.classifier is None:
            return self._pass(canonical, rules.score, "S1", trace, started)

        model = self._run_s2(canonical, query, trace)
        if model.score < self.config.tau_lo:
            return self._pass(canonical, model.score, "S2", trace, started)
        if model.score > self.config.tau_hi:
            spans = rules.spans or self._localize(canonical, query)
            return self._decide(canonical, model.score, "S2", spans, trace, started)

        # Banda grigia. Senza giudice il sistema non ha nulla di calibrato con cui
        # decidere qui: lascia passare, cosi' l'ablation misura la recall persa
        # dall'assenza di escalation e non un FPR gonfiato.
        if self.judge is None:
            return self._pass(canonical, model.score, "S2", trace, started)

        verdict = self._run_s3(canonical, trace)
        if not verdict.score:
            return self._pass(canonical, model.score, "S3", trace, started)
        spans = rules.spans or self._localize(canonical, query)
        return self._decide(canonical, model.score, "S3", spans, trace, started)

    def _run_s0(self, text: str, trace: list[StageResult]) -> str:
        started = time.perf_counter()
        canonical, flags = canonicalize(text)
        detail = ",".join(name for name, fired in flags.items() if fired)
        trace.append(StageResult("S0", 0.0, (), (time.perf_counter() - started) * 1000.0, detail))
        return canonical

    def _run_s1(self, canonical: str, trace: list[StageResult]) -> StageResult:
        result = score_rules(canonical)
        trace.append(result)
        return result

    def _run_s2(self, canonical: str, query: str | None, trace: list[StageResult]) -> StageResult:
        started = time.perf_counter()
        context = query if self.config.use_query_context else None
        score = self.classifier.score_one(canonical, context)  # type: ignore[union-attr]
        result = StageResult("S2", score, (), (time.perf_counter() - started) * 1000.0)
        trace.append(result)
        return result

    def _run_s3(self, canonical: str, trace: list[StageResult]) -> StageResult:
        started = time.perf_counter()
        flagged = self.judge.judge(canonical)  # type: ignore[union-attr]
        result = StageResult(
            "S3",
            1.0 if flagged else 0.0,
            (),
            (time.perf_counter() - started) * 1000.0,
            "INJECTION" if flagged else "SAFE",
        )
        trace.append(result)
        return result

    def _localize(self, canonical: str, query: str | None) -> tuple[Span, ...]:
        """Ricerca dello span sospetto per segmenti, quando a decidere e' stato S2 o S3."""
        if self.classifier is None:
            return ()
        segments = [
            (m.start(), m.end())
            for m in _SEGMENT_RE.finditer(canonical)
            if m.end() - m.start() >= _MIN_SEGMENT_CHARS
        ]
        if len(segments) < 2:
            return (Span(0, len(canonical)),)
        context = query if self.config.use_query_context else None
        texts = [canonical[a:b] for a, b in segments]
        scores = self.classifier.score(texts, [context or ""] * len(texts) if context else None)
        flagged = [
            Span(a, b) for (a, b), s in zip(segments, scores, strict=True) if s > self.config.tau_hi
        ]
        return tuple(flagged) if flagged else (Span(0, len(canonical)),)

    def _pass(
        self, canonical: str, score: float, stage: str, trace: list[StageResult], started: float
    ) -> Verdict:
        return Verdict(
            action="pass",
            score=score,
            decided_by=stage,  # type: ignore[arg-type]
            canonical_text=canonical,
            clean_text=canonical,
            spans=(),
            trace=tuple(trace),
            total_latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _decide(
        self,
        canonical: str,
        score: float,
        stage: str,
        spans: tuple[Span, ...],
        trace: list[StageResult],
        started: float,
    ) -> Verdict:
        spans = expand_spans(canonical, merge_spans(spans))
        if self.config.high_score_action == "block":
            clean, action = "", "block"
        else:
            clean, action = sanitize(
                canonical, spans, self.config.redaction_marker, self.config.min_remaining_ratio
            )
        return Verdict(
            action=action,
            score=score,
            decided_by=stage,  # type: ignore[arg-type]
            canonical_text=canonical,
            clean_text=clean,
            spans=merge_spans(spans),
            trace=tuple(trace),
            total_latency_ms=(time.perf_counter() - started) * 1000.0,
        )
