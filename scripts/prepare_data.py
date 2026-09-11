"""Scarica le sorgenti, avvelena i documenti BIPIA e scrive gli split in parquet."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shield.config import Config, load_config  # noqa: E402
from shield.data import (  # noqa: E402
    assert_disjoint_attacks,
    build_chunk_records,
    build_detector_splits,
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


def build(config: Config) -> dict[str, pl.DataFrame]:
    from transformers import AutoTokenizer

    rng = random.Random(config.seed)
    raw = config.paths.raw
    logger.info("seed=%d", config.seed)

    promptshield = load_promptshield(raw, config.data)
    splits = build_detector_splits(promptshield, config.data.calibration_benign_min, rng)

    tokenizer = AutoTokenizer.from_pretrained(config.classifier.model_name)
    for split, name in (("test", "test"), ("train", "bipia_train")):
        docs = load_bipia_documents(raw, config.data, split)
        attacks = load_bipia_attacks(raw, config.data, split)
        logger.info(
            "BIPIA %s: %d documenti, %d istruzioni d'attacco", split, len(docs), len(attacks)
        )
        splits[name] = build_chunk_records(docs, attacks, tokenizer, config.data, rng)

    test_attacks = [a.text for a in load_bipia_attacks(raw, config.data, "test")]
    train_texts = promptshield["train"]["text"].to_list() + [
        a.text for a in load_bipia_attacks(raw, config.data, "train")
    ]
    logger.info(
        "controllo di non sovrapposizione: %d attacchi di test contro %d testi di training",
        len(test_attacks),
        len(train_texts),
    )
    assert_disjoint_attacks(train_texts, test_attacks)
    logger.info("nessuna istruzione d'attacco del test compare nel training")
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
