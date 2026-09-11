from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .rules import score_rules

# Separatore fra query e chunk. Il modello vede `[query] [SEP] [chunk]` quando la
# query e' disponibile: la stessa istruzione e' benigna in una domanda dell'utente
# e malevola dentro un documento, e senza il contesto la distinzione si perde.
_SEP = " [SEP] "


def vram_report() -> dict[str, float]:
    """VRAM allocata e picco in GiB; valori a zero se non c'e' una GPU."""
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "peak_gib": 0.0, "total_gib": 0.0}
    gib = float(2**30)
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "peak_gib": torch.cuda.max_memory_allocated() / gib,
        "total_gib": torch.cuda.get_device_properties(0).total_memory / gib,
    }


def build_input(text: str, query: str | None) -> str:
    return f"{query}{_SEP}{text}" if query else text


class TransformerClassifier:
    """S2: inferenza batch di un encoder addestrato, in bf16 su GPU quando c'e'."""

    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = 256,
        batch_size: int = 64,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        default = "bfloat16" if self.device == "cuda" else "float32"
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), dtype=getattr(torch, dtype or default)
        )
        self._model.to(self.device).eval()
        self._positive_index = _positive_index(self._model.config.id2label)

    def score(self, texts: list[str], queries: list[str] | None = None) -> list[float]:
        return [float(v) for v in self.score_array(texts, queries)]

    def score_array(
        self, texts: list[str], queries: list[str] | None = None
    ) -> NDArray[np.float64]:
        if not texts:
            return np.zeros(0, dtype=np.float64)
        pairs = [build_input(t, queries[i] if queries else None) for i, t in enumerate(texts)]
        out = np.empty(len(pairs), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**encoded).logits.float()
                probs = torch.softmax(logits, dim=-1)[:, self._positive_index]
                out[start : start + len(batch)] = probs.cpu().numpy()
        return out

    def score_one(self, text: str, query: str | None = None) -> float:
        return float(self.score_array([text], [query] if query else None)[0])


def _positive_index(id2label: dict[int, str]) -> int:
    """Individua l'indice della classe 'iniezione' dalle etichette del checkpoint."""
    for index, label in id2label.items():
        if str(label).upper() in {"INJECTION", "LABEL_1", "1", "UNSAFE", "MALICIOUS", "JAILBREAK"}:
            return int(index)
    return len(id2label) - 1


class RuleDetector:
    """Baseline sole regole S1, esposta con la stessa interfaccia del classificatore."""

    def score(self, texts: list[str], queries: list[str] | None = None) -> list[float]:
        return [score_rules(t).score for t in texts]

    def score_one(self, text: str, query: str | None = None) -> float:
        return score_rules(text).score


class TfidfDetector:
    """Baseline TF-IDF piu' regressione logistica, caricata da un artefatto picklato."""

    def __init__(self, model_path: str | Path) -> None:
        with Path(model_path).open("rb") as handle:
            self._pipeline = pickle.load(handle)  # noqa: S301

    def score(self, texts: list[str], queries: list[str] | None = None) -> list[float]:
        pairs = [build_input(t, queries[i] if queries else None) for i, t in enumerate(texts)]
        return [float(p) for p in self._pipeline.predict_proba(pairs)[:, 1]]

    def score_one(self, text: str, query: str | None = None) -> float:
        return self.score([text], [query] if query else None)[0]
