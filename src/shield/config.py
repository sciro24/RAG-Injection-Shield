from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import Action


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed: Path
    models: Path
    figures: Path
    results: Path
    demo_corpus: Path
    demo_calibration: Path


@dataclass(frozen=True)
class DataConfig:
    promptshield_repo: str
    bipia_repo: str
    bipia_ref: str
    bipia_domains: tuple[str, ...]
    xsum_repo: str
    max_docs_per_domain: int
    variants_per_doc: int
    synthetic_variants_per_doc: int
    chunk_tokens: int
    chunk_overlap: int
    min_overlap_chars: int
    positive_ratio: float
    calibration_benign_min: int
    train_fraction: float
    positions: tuple[str, ...]


@dataclass(frozen=True)
class ClassifierConfig:
    model_name: str
    fallback_model_name: str
    max_length: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    epochs: int
    warmup_ratio: float
    eval_steps: int
    weight_decay: float
    bf16: bool
    optimizer: str
    seed: int
    checkpoint_dir: Path
    external_baseline: str


@dataclass(frozen=True)
class CascadeConfig:
    rules_block_threshold: float
    tau_lo: float
    tau_hi: float
    high_score_action: Action
    redaction_marker: str
    min_remaining_ratio: float
    use_query_context: bool


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model_name: str
    lmstudio_base_url: str
    extra_base_urls: tuple[str, ...]
    lmstudio_model: str
    reasoning_effort: str
    max_new_tokens: int
    judge_max_new_tokens: int
    temperature: float
    dtype: str


@dataclass(frozen=True)
class RagConfig:
    encoder_name: str
    top_k: int
    chunk_chars: int
    chunk_overlap_chars: int


@dataclass(frozen=True)
class EvaluationConfig:
    target_fprs: tuple[float, ...]
    bootstrap_n: int
    asr_sample_size: int
    utility_queries: int
    configurations: tuple[str, ...]
    attack_families: tuple[str, ...]
    victims: tuple[str, ...]


@dataclass(frozen=True)
class DemoConfig:
    n_reviews: int
    n_poisoned: int
    n_calibration: int


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    data: DataConfig
    classifier: ClassifierConfig
    cascade: CascadeConfig
    llm: LLMConfig
    rag: RagConfig
    evaluation: EvaluationConfig
    demo: DemoConfig
    raw: dict[str, Any] = field(default_factory=dict)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value)


def load_config(path: str | Path) -> Config:
    """Legge il file yaml e lo converte nelle dataclass tipizzate del progetto."""
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    paths = Paths(**{k: Path(v) for k, v in raw["paths"].items()})
    data = DataConfig(
        **{
            k: (_as_tuple(v) if k in {"bipia_domains", "positions"} else v)
            for k, v in raw["data"].items()
        }
    )
    classifier = ClassifierConfig(
        **{k: (Path(v) if k == "checkpoint_dir" else v) for k, v in raw["classifier"].items()}
    )
    evaluation = EvaluationConfig(
        **{
            k: (
                _as_tuple(v)
                if k in {"target_fprs", "configurations", "attack_families", "victims"}
                else v
            )
            for k, v in raw["evaluation"].items()
        }
    )
    return Config(
        seed=raw["seed"],
        paths=paths,
        data=data,
        classifier=classifier,
        cascade=CascadeConfig(**raw["cascade"]),
        llm=LLMConfig(
            **{k: (_as_tuple(v) if k == "extra_base_urls" else v) for k, v in raw["llm"].items()}
        ),
        rag=RagConfig(**raw["rag"]),
        evaluation=evaluation,
        demo=DemoConfig(**raw["demo"]),
        raw=raw,
    )
