from __future__ import annotations

from shield.config import RagConfig
from shield.rag import RagPipeline
from shield.types import Document


class FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        return self.docs[:k]


def make_pipeline(docs: list[Document]) -> RagPipeline:
    config = RagConfig(encoder_name="x", top_k=2, chunk_chars=900, chunk_overlap_chars=0)
    return RagPipeline(FakeRetriever(docs), llm=None, config=config)  # type: ignore[arg-type]


def test_pin_prende_il_posto_dell_ultimo_se_il_retrieval_non_lo_recupera():
    docs = [Document(f"d{i}", f"testo {i}", {}) for i in range(4)]
    pipeline = make_pipeline(docs)
    retrieved, passages, _, _, _ = pipeline.build_context("q", None, pin=docs[3])
    assert [d.id for d in retrieved] == ["d0", "d3"]
    assert passages == ["testo 0", "testo 3"]


def test_pin_resta_al_suo_rango_se_recuperato():
    docs = [Document(f"d{i}", f"testo {i}", {}) for i in range(4)]
    pipeline = make_pipeline(docs)
    retrieved, _, _, _, _ = pipeline.build_context("q", None, pin=docs[0])
    assert [d.id for d in retrieved] == ["d0", "d1"]


def test_senza_pin_il_contesto_e_il_retrieval():
    docs = [Document(f"d{i}", f"testo {i}", {}) for i in range(4)]
    retrieved, _, _, _, _ = make_pipeline(docs).build_context("q", None)
    assert [d.id for d in retrieved] == ["d0", "d1"]
