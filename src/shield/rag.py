from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .cascade import Cascade
from .config import RagConfig
from .llm import RAG_PROMPT, LocalLLM
from .types import Document, RagResult, Verdict

_PASSAGE_SEPARATOR = "\n\n---\n\n"


def load_corpus(path: str | Path) -> list[Document]:
    """Legge un corpus jsonl con campi `id`, `text` e metadati liberi."""
    documents: list[Document] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = {k: str(v) for k, v in row.items() if k not in {"id", "text"}}
        documents.append(Document(id=str(row["id"]), text=row["text"], meta=meta))
    return documents


def split_documents(docs: list[Document], size: int, overlap: int) -> list[Document]:
    """Spezza i documenti in passaggi a caratteri: il retrieval non e' oggetto di studio."""
    passages: list[Document] = []
    stride = max(1, size - overlap)
    for doc in docs:
        if len(doc.text) <= size:
            passages.append(doc)
            continue
        for index, start in enumerate(range(0, len(doc.text), stride)):
            piece = doc.text[start : start + size]
            if not piece.strip():
                continue
            meta = dict(doc.meta, parent_id=doc.id, passage_index=str(index))
            passages.append(Document(id=f"{doc.id}#{index}", text=piece, meta=meta))
    return passages


class DenseRetriever:
    """Retrieval denso su matrice numpy: con ~300 documenti un vector db sarebbe zavorra."""

    def __init__(self, docs: list[Document], model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.docs = docs
        self._encoder = SentenceTransformer(model_name)
        self._matrix: NDArray[np.float32] = self._encoder.encode(
            [d.text for d in docs], normalize_embeddings=True, show_progress_bar=False
        )

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        vector = self.encode([query])[0]
        scores = self._matrix @ vector
        top = np.argsort(scores)[::-1][:k]
        return [self.docs[int(i)] for i in top]

    def replace(self, index: int, doc: Document) -> Document:
        """Sostituisce un documento ricodificando solo la sua riga: serve all'esperimento
        ASR, che avvelena un documento diverso a ogni campione."""
        previous = self.docs[index]
        self.docs[index] = doc
        self._matrix[index] = self.encode([doc.text])[0]
        return previous

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        return self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def apply_defense(
    docs: list[Document], query: str, defense: Cascade | None
) -> tuple[list[str], tuple[Verdict, ...]]:
    """Passa ogni documento recuperato nella cascata e restituisce i testi da inoltrare."""
    if defense is None:
        return [d.text for d in docs], ()
    kept: list[str] = []
    verdicts: list[Verdict] = []
    for doc in docs:
        verdict = defense.inspect(doc.text, query)
        verdicts.append(verdict)
        if verdict.action != "block" and verdict.clean_text.strip():
            kept.append(verdict.clean_text)
    return kept, tuple(verdicts)


class RagPipeline:
    """Retrieval, difesa opzionale, generazione. Il punto di innesto della difesa e'
    esplicito: `apply_defense` sta fra `retrieve` e il prompt del generatore."""

    def __init__(self, retriever: DenseRetriever, llm: LocalLLM, config: RagConfig) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config

    def build_context(
        self,
        query: str,
        defense: Cascade | None,
        k: int | None = None,
        pin: Document | None = None,
    ) -> tuple[list[Document], list[str], tuple[Verdict, ...], float, float]:
        """`pin` e' un documento che deve comunque stare nel contesto: se il retrieval non
        lo recupera prende il posto dell'ultimo. Serve all'esperimento ASR, che misura la
        difesa e non il retrieval: un'iniezione sposta l'embedding e senza questo vincolo
        tre campioni su quattro non arriverebbero nemmeno alla cascata."""
        started = time.perf_counter()
        retrieved = self.retriever.retrieve(query, k or self.config.top_k)
        if pin is not None and all(d.id != pin.id for d in retrieved):
            retrieved = [*retrieved[:-1], pin]
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        passages, verdicts = apply_defense(retrieved, query, defense)
        defense_ms = (time.perf_counter() - started) * 1000.0
        return retrieved, passages, verdicts, retrieval_ms, defense_ms

    def answer(
        self,
        query: str,
        defense: Cascade | None,
        max_tokens: int | None = None,
        pin: Document | None = None,
    ) -> RagResult:
        retrieved, passages, verdicts, retrieval_ms, defense_ms = self.build_context(
            query, defense, pin=pin
        )
        context = _PASSAGE_SEPARATOR.join(passages) if passages else "(no usable context)"
        prompt = RAG_PROMPT.format(context=context, question=query)
        started = time.perf_counter()
        answer = self.llm.generate(prompt, max_tokens or self.llm.config.max_new_tokens)
        generation_ms = (time.perf_counter() - started) * 1000.0
        return RagResult(
            query=query,
            answer=answer.strip(),
            retrieved=tuple(retrieved),
            verdicts=verdicts,
            context_used=context,
            retrieval_ms=retrieval_ms,
            defense_ms=defense_ms,
            generation_ms=generation_ms,
        )

    def stream_answer(
        self, query: str, defense: Cascade | None, max_tokens: int | None = None
    ) -> tuple[list[Document], tuple[Verdict, ...], Iterator[str]]:
        """Come `answer` ma restituisce i token man mano: la demo non deve restare ferma."""
        retrieved, passages, verdicts, _, _ = self.build_context(query, defense)
        context = _PASSAGE_SEPARATOR.join(passages) if passages else "(no usable context)"
        prompt = RAG_PROMPT.format(context=context, question=query)
        stream = self.llm.stream(prompt, max_tokens or self.llm.config.max_new_tokens)
        return retrieved, verdicts, stream
