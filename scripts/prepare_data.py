"""Scarica le sorgenti, avvelena i documenti BIPIA e scrive gli split in parquet."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shield.attacks import HELDOUT_STYLES, TRAIN_STYLES, synth_attack  # noqa: E402
from shield.config import Config, load_config  # noqa: E402
from shield.data import (  # noqa: E402
    CANARY_FAMILIES,
    assert_disjoint_attacks,
    build_chunk_records,
    build_detector_splits,
    build_synthetic_records,
    load_bipia_attacks,
    load_bipia_documents,
    load_promptshield,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")


def describe(name: str, frame: pl.DataFrame) -> None:
    positives = int(frame.filter(pl.col("label") == 1).height)
    lengths = frame.select(pl.col("text").str.len_chars()).to_series()
    logger.info(
        "%-14s n=%6d positivi=%6d (%.1f%%) lunghezza mediana=%5d p95=%6d",
        name,
        frame.height,
        positives,
        100.0 * positives / max(1, frame.height),
        int(lengths.median() or 0),
        int(lengths.quantile(0.95) or 0),
    )
    if "domain" in frame.columns and frame["domain"].n_unique() > 1:
        by_domain = frame.group_by("domain").agg(
            pl.len().alias("n"), pl.col("label").mean().alias("quota_positivi")
        )
        for row in by_domain.sort("domain").iter_rows(named=True):
            logger.info(
                "    %-10s n=%5d positivi=%.2f", row["domain"], row["n"], row["quota_positivi"]
            )


COLUMNS = ["text", "query", "label", "domain"]


def build(config: Config) -> dict[str, pl.DataFrame]:
    """Split del detector.

    train        BIPIA train (iniezione indiretta su documenti reali) + attacchi sintetici
                 negli stili di addestramento + PromptShield (iniezione diretta)
    validation   stessa composizione, su documenti BIPIA tenuti fuori dal training
    calibration  soli benigni di PromptShield, per il confronto fra soglie
    test         BIPIA test: attacchi e documenti disgiunti dal training
    stealth      documenti BIPIA test avvelenati con gli stili tenuti fuori: pattern non noti
    bipia_train  tutto BIPIA train, per il leave-one-domain-out"""
    from transformers import AutoTokenizer

    rng = random.Random(config.seed)
    raw, cfg = config.paths.raw, config.data
    logger.info("seed=%d", config.seed)

    promptshield = build_detector_splits(
        load_promptshield(raw, cfg), cfg.calibration_benign_min, rng
    )
    tokenizer = AutoTokenizer.from_pretrained(config.classifier.model_name)

    train_docs = load_bipia_documents(raw, cfg, "train")
    train_attacks = load_bipia_attacks(raw, cfg, "train")
    test_docs = load_bipia_documents(raw, cfg, "test")
    test_attacks = load_bipia_attacks(raw, cfg, "test")
    rng.shuffle(train_docs)
    cut = int(len(train_docs) * cfg.train_fraction)
    fit_docs, val_docs = train_docs[:cut], train_docs[cut:]
    logger.info(
        "BIPIA: %d documenti di training (%d fit, %d validation), %d attacchi; "
        "%d documenti di test, %d attacchi",
        len(train_docs),
        len(fit_docs),
        len(val_docs),
        len(train_attacks),
        len(test_docs),
        len(test_attacks),
    )

    def chunks(docs: list, attacks: list) -> pl.DataFrame:
        return build_chunk_records(docs, attacks, tokenizer, cfg, rng)

    def synthetic(docs: list, styles: tuple[str, ...], variants: int) -> pl.DataFrame:
        return build_synthetic_records(docs, styles, tokenizer, cfg, rng, variants)

    bipia_fit, bipia_val = chunks(fit_docs, train_attacks), chunks(val_docs, train_attacks)
    synth_fit = synthetic(fit_docs, TRAIN_STYLES, cfg.synthetic_variants_per_doc)
    synth_val = synthetic(val_docs, TRAIN_STYLES, 2)

    seed = rng.randint(0, 2**31 - 1)
    splits = {
        "train": pl.concat(
            [
                bipia_fit.select(COLUMNS),
                synth_fit.select(COLUMNS),
                promptshield["train"].select(COLUMNS),
            ]
        ).sample(fraction=1.0, shuffle=True, seed=seed),
        "validation": pl.concat(
            [
                bipia_val.select(COLUMNS),
                synth_val.select(COLUMNS),
                promptshield["validation"].select(COLUMNS),
            ]
        ),
        "calibration": promptshield["calibration"],
        "test": chunks(test_docs, test_attacks),
        "stealth": synthetic(test_docs, HELDOUT_STYLES, 2),
        "bipia_train": pl.concat([bipia_fit, bipia_val]),
    }

    heldout = [a.text for a in test_attacks]
    heldout += [synth_attack(HELDOUT_STYLES, rng) for _ in range(300)]
    heldout += [t.format(canary="CNRYTEST0000") for f in CANARY_FAMILIES.values() for t in f]
    train_texts = splits["train"]["text"].to_list()
    logger.info(
        "controllo di non sovrapposizione: %d attacchi tenuti fuori contro %d testi di training",
        len(heldout),
        len(train_texts),
    )
    assert_disjoint_attacks(train_texts, heldout)
    logger.info("nessun attacco tenuto fuori compare nel training")
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    config.paths.processed.mkdir(parents=True, exist_ok=True)
    splits = build(config)
    for name, frame in splits.items():
        target = config.paths.processed / f"{name}.parquet"
        frame.write_parquet(target)
        describe(name, frame)
        logger.info("scritto %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
