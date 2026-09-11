"""Addestra il classificatore S2 e la baseline TF-IDF sugli stessi split."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shield.classifier import build_input, vram_report  # noqa: E402
from shield.config import Config, load_config  # noqa: E402

# ln(2): la loss di un classificatore binario che predice sempre la stessa classe.
# Restarci incollati significa che il modello e' collassato e non si riprendera'.
DEGENERATE_LOSS = 0.6931
DEGENERATE_TOLERANCE = 5e-4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info("seed=%d", seed)


class DivergenceGuard(TrainerCallback):
    """Interrompe il training se la loss diventa NaN o si blocca sulla soluzione costante.
    Senza questa guardia un collasso si scopre solo alla fine, a GPU gia' occupata a vuoto."""

    def __init__(self, patience: int = 6) -> None:
        self.patience = patience
        self.degenerate_in_a_row = 0

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN003
        loss = (logs or {}).get("loss")
        if loss is None:
            return control
        if not np.isfinite(loss):
            logger.error(
                "loss non finita (%s) al passo %d: training interrotto", loss, state.global_step
            )
            control.should_training_stop = True
            return control
        if abs(loss - DEGENERATE_LOSS) < DEGENERATE_TOLERANCE:
            self.degenerate_in_a_row += 1
            if self.degenerate_in_a_row >= self.patience:
                logger.error(
                    "loss ferma su ln2 da %d rilevazioni al passo %d: il modello e' collassato, "
                    "training interrotto. Abbassa learning_rate o cambia model_name.",
                    self.degenerate_in_a_row,
                    state.global_step,
                )
                control.should_training_stop = True
        else:
            self.degenerate_in_a_row = 0
        return control


def read_split(config: Config, name: str) -> pl.DataFrame:
    return pl.read_parquet(config.paths.processed / f"{name}.parquet")


def to_inputs(frame: pl.DataFrame) -> tuple[list[str], list[int]]:
    queries = frame["query"].to_list() if "query" in frame.columns else [""] * frame.height
    texts = [build_input(t, q or None) for t, q in zip(frame["text"], queries, strict=True)]
    return texts, frame["label"].to_list()


def load_tokenizer(config: Config) -> tuple[object, str]:
    """DeBERTa-v3 richiede sentencepiece: se la conversione fallisce si ripiega su roberta."""
    from transformers import AutoTokenizer

    name = config.classifier.model_name
    try:
        return AutoTokenizer.from_pretrained(name), name
    except Exception as error:  # noqa: BLE001
        fallback = config.classifier.fallback_model_name
        logger.warning("tokenizer di %s non caricabile (%s), ripiego su %s", name, error, fallback)
        return AutoTokenizer.from_pretrained(fallback), fallback


def _encode_split(dataset_cls: object, tokenizer: object, frame: pl.DataFrame, max_length: int):  # noqa: ANN202
    texts, labels = to_inputs(frame)
    dataset = dataset_cls.from_dict({"text": texts, "labels": labels})  # type: ignore[attr-defined]
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),  # type: ignore[operator]
        batched=True,
        remove_columns=["text"],
    )


def _persist(
    trainer: object, tokenizer: object, output_dir: Path, curves_to: Path | None
) -> dict[str, float]:
    """Salva checkpoint, curve e log, e segnala un eventuale collasso del modello."""
    final = trainer.evaluate()  # type: ignore[attr-defined]
    logger.info("eval_loss finale: %.4f", final["eval_loss"])
    if abs(final["eval_loss"] - DEGENERATE_LOSS) < 1e-2:
        logger.error("il modello e' degenere (eval_loss ~ ln2): il checkpoint NON e' utilizzabile")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))  # type: ignore[attr-defined]
    tokenizer.save_pretrained(str(output_dir))  # type: ignore[attr-defined]
    history = trainer.state.log_history  # type: ignore[attr-defined]
    if curves_to is not None:
        plot_curves(history, curves_to)
    (output_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    report = vram_report()
    report["eval_loss"] = float(final["eval_loss"])
    return report


def _training_args(
    config: Config, steps_per_epoch: int, output_dir: Path, max_steps: int
) -> object:
    from transformers import TrainingArguments

    cfg = config.classifier
    total = max_steps if max_steps > 0 else steps_per_epoch * cfg.epochs
    # Sui domini piccoli del leave-one-domain-out il run e' piu' corto dell'intervallo
    # di valutazione: senza almeno una eval, load_best_model_at_end non ha nulla da caricare.
    interval = max(1, min(cfg.eval_steps, total // 2))
    return TrainingArguments(
        output_dir=str(config.paths.models / "runs" / output_dir.name),
        num_train_epochs=cfg.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=int(cfg.warmup_ratio * steps_per_epoch * cfg.epochs),
        bf16=cfg.bf16 and torch.cuda.is_available(),
        optim=cfg.optimizer,
        seed=cfg.seed,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=interval,
        save_steps=interval,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
        report_to=[],
    )


def fit_detector(
    config: Config,
    train_frame: pl.DataFrame,
    val_frame: pl.DataFrame,
    output_dir: Path,
    max_steps: int = -1,
    curves_to: Path | None = None,
) -> dict[str, float]:
    """Addestra un encoder di sequenza sugli split passati e salva il checkpoint.
    Riusata dall'esperimento leave-one-domain-out, che cambia solo gli split."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
    )

    cfg = config.classifier
    tokenizer, model_name = load_tokenizer(config)
    logger.info("modello: %s", model_name)
    splits = {
        name: _encode_split(Dataset, tokenizer, frame, cfg.max_length)
        for name, frame in (("train", train_frame), ("validation", val_frame))
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "SAFE", 1: "INJECTION"},
        label2id={"SAFE": 0, "INJECTION": 1},
    )
    steps_per_epoch = max(1, len(splits["train"]) // cfg.batch_size)
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * cfg.epochs
    logger.info(
        "training: %d esempi, %d passi previsti, valutazione ogni %d passi",
        len(splits["train"]),
        total_steps,
        cfg.eval_steps,
    )
    args = _training_args(config, steps_per_epoch, output_dir, max_steps)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[DivergenceGuard()],
    )
    trainer.train()
    return _persist(trainer, tokenizer, output_dir, curves_to)


def train_transformer(config: Config, max_steps: int = -1, subset: int = 0) -> dict[str, float]:
    """Training del classificatore principale sugli split di PromptShield."""
    train_frame = read_split(config, "train")
    if subset:
        train_frame = train_frame.sample(
            n=min(subset, train_frame.height), shuffle=True, seed=config.seed
        )
    return fit_detector(
        config,
        train_frame,
        read_split(config, "validation"),
        config.classifier.checkpoint_dir,
        max_steps=max_steps,
        curves_to=config.paths.figures / "training_loss.png",
    )


def plot_curves(history: list[dict], target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    figure, axis = plt.subplots(figsize=(6, 4))
    if train:
        axis.plot(*zip(*train, strict=True), label="train")
    if evals:
        axis.plot(*zip(*evals, strict=True), marker="o", label="validation")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.set_title("Curve di loss del classificatore S2")
    axis.legend()
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)
    logger.info("scritto %s", target)


def train_tfidf(config: Config) -> Path:
    """Baseline: TF-IDF su n-grammi di parole e di caratteri, regressione logistica."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_union

    texts, labels = to_inputs(read_split(config, "train"))
    pipeline = Pipeline(
        [
            (
                "features",
                make_union(
                    TfidfVectorizer(
                        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200_000
                    ),
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000
                    ),
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=config.seed)),
        ]
    )
    pipeline.fit(texts, labels)
    target = config.paths.models / "tfidf_baseline.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(pipeline, handle)
    logger.info("scritto %s", target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="prova rapida: 250 passi su 4000 esempi, per verificare che la loss scenda",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seeds(config.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("VRAM all'avvio: %s", vram_report())

    vram = train_transformer(
        config, max_steps=250 if args.smoke else -1, subset=4000 if args.smoke else 0
    )
    logger.info("VRAM al termine: %s", vram)
    if args.smoke:
        logger.info("prova rapida conclusa: se eval_loss e' ben sotto 0.69 rilancia senza --smoke")
        return 0
    if not args.skip_tfidf:
        train_tfidf(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
