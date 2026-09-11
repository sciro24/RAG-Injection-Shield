"""Esegue gli esperimenti del progetto e scrive tabelle e figure in reports/."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shield.cascade import Cascade  # noqa: E402
from shield.classifier import (  # noqa: E402
    RuleDetector,
    TfidfDetector,
    TransformerClassifier,
    vram_report,
)
from shield.config import Config, load_config  # noqa: E402
from shield.evaluation import (  # noqa: E402
    DEMO_QUERIES,
    compare_detectors,
    hard_cases,
    run_asr,
    run_utility,
    sample_canary_cases,
    summarise_asr,
    transfer_matrix,
)
from shield.metrics import false_positive_rate, threshold_at_fpr  # noqa: E402
from shield.rag import DenseRetriever, RagPipeline, load_corpus, split_documents  # noqa: E402
from shield.types import Classifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


def write(frame: pl.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(target)
    logger.info("scritto %s (%d righe)", target, frame.height)


def read(config: Config, name: str) -> pl.DataFrame:
    return pl.read_parquet(config.paths.processed / f"{name}.parquet")


def checkpoint_is_degenerate(checkpoint: Path) -> bool:
    """Un checkpoint la cui eval_loss e' rimasta a ln2 predice sempre la stessa classe:
    usarlo produrrebbe tabelle plausibili ma prive di significato."""
    log = checkpoint / "training_log.json"
    if not log.exists():
        return False
    import json

    losses = [x["eval_loss"] for x in json.loads(log.read_text()) if "eval_loss" in x]
    return bool(losses) and abs(min(losses) - 0.6931) < 1e-2


def load_detectors(config: Config, with_external: bool) -> dict[str, Classifier]:
    """Il detector del progetto e le tre baseline previste dalla fase 4."""
    detectors: dict[str, Classifier] = {"rules_s1": RuleDetector()}
    checkpoint = config.classifier.checkpoint_dir
    if checkpoint.exists() and checkpoint_is_degenerate(checkpoint):
        raise SystemExit(
            f"Il checkpoint in {checkpoint} e' degenere (eval_loss ~ ln2): il modello predice "
            "sempre la stessa classe e ogni risultato che ne deriva sarebbe privo di senso. "
            "Cancellalo e rilancia 'make train'."
        )
    if checkpoint.exists():
        detectors["shield_s2"] = TransformerClassifier(
            checkpoint, config.classifier.max_length, config.classifier.eval_batch_size
        )
    else:
        logger.warning("checkpoint assente in %s: esegui prima scripts/train.py", checkpoint)
    tfidf = config.paths.models / "tfidf_baseline.pkl"
    if tfidf.exists():
        detectors["tfidf_lr"] = TfidfDetector(tfidf)
    if with_external:
        # Il detector pubblico e' un deberta-v3: in bf16 produce NaN su questa GPU,
        # quindi l'inferenza esterna gira in fp32.
        detectors["external_hf"] = TransformerClassifier(
            config.classifier.external_baseline,
            config.classifier.max_length,
            config.classifier.eval_batch_size,
            dtype="float32",
        )
    return detectors


def experiment_detector(config: Config, args: argparse.Namespace) -> None:
    """Confronto dei detector su due test set: BIPIA (attacchi reali, disgiunti dal
    training) e `stealth` (stili di attacco mai visti in training: i pattern non noti)."""
    calibration = read(config, "calibration")
    detectors = load_detectors(config, not args.no_external)
    logger.info("detector confrontati: %s", ", ".join(detectors))
    tables = []
    for name in ("test", "stealth"):
        frame = read(config, name)
        table = compare_detectors(detectors, frame, calibration, config, use_query=True)
        tables.append(table.with_columns(pl.lit(name).alias("test_set")))
        for row in table.iter_rows(named=True):
            logger.info(
                "  %-8s %-12s auc=%.3f tpr@1%%=%.3f tpr@0.1%%=%.3f",
                name,
                row["detector"],
                row["auc"],
                row["tpr_at_fpr_0.01"],
                row["tpr_at_fpr_0.001"],
            )
    write(pl.concat(tables), config.paths.results / "detector_comparison.csv")
    if "shield_s2" in detectors:
        dump_hard_cases(config, detectors["shield_s2"], read(config, "test"), calibration)


def dump_hard_cases(
    config: Config, detector: Classifier, test: pl.DataFrame, calibration: pl.DataFrame
) -> None:
    """Salva gli errori piu' netti del modello: la demo li usa come casi difficili."""
    import json

    from shield.evaluation import score_frame

    tau = threshold_at_fpr(score_frame(detector, calibration, True), 0.01)
    cases = hard_cases(test, score_frame(detector, test, True), tau)
    target = config.paths.results / "hard_cases.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cases, indent=2, ensure_ascii=False))
    logger.info("scritto %s (%d casi)", target, len(cases))


def experiment_generalization(config: Config, args: argparse.Namespace) -> None:
    """Leave-one-domain-out dentro BIPIA, con il detector completo (`all`) come riferimento."""
    from train import fit_detector  # noqa: PLC0415

    test = read(config, "test")
    rows: list[pl.DataFrame] = []
    detectors = load_detectors(config, False)
    if "shield_s2" in detectors:
        rows.append(transfer_matrix(detectors["shield_s2"], test, "all", config))

    bipia = read(config, "bipia_train")
    domains = sorted(bipia["domain"].unique().to_list())
    for domain in domains:
        subset = bipia.filter(pl.col("domain") == domain)
        if subset["label"].n_unique() < 2:
            continue
        logger.info("leave-one-domain-out: training su %s (%d chunk)", domain, subset.height)
        if args.lodo_detector == "tfidf":
            detector: Classifier = _fit_tfidf_on(subset, config)
        else:
            output = config.paths.models / f"lodo_{domain}"
            split = int(subset.height * config.data.train_fraction)
            fit_detector(config, subset.head(split), subset.tail(subset.height - split), output)
            detector = TransformerClassifier(
                output, config.classifier.max_length, config.classifier.eval_batch_size
            )
        rows.append(transfer_matrix(detector, test, domain, config))

    matrix = pl.concat(rows, how="diagonal") if rows else pl.DataFrame()
    write(matrix, config.paths.results / "transfer_matrix.csv")
    plot_transfer(matrix, config.paths.figures / "transfer_matrix.png")


def _fit_tfidf_on(frame: pl.DataFrame, config: Config) -> Classifier:
    """Variante economica del leave-one-domain-out: stessa procedura, modello lineare."""
    import pickle
    import tempfile

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_union

    from shield.classifier import build_input

    texts = [build_input(t, q or None) for t, q in zip(frame["text"], frame["query"], strict=True)]
    pipeline = Pipeline(
        [
            (
                "features",
                make_union(
                    TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2),
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=config.seed)),
        ]
    )
    pipeline.fit(texts, frame["label"].to_list())
    handle = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)  # noqa: SIM115
    pickle.dump(pipeline, handle)
    handle.close()
    return TfidfDetector(handle.name)


def plot_transfer(matrix: pl.DataFrame, target: Path) -> None:
    if matrix.is_empty():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = matrix.pivot(values="tpr_at_fpr_0.01", index="trained_on", on="tested_on")
    rows = pivot["trained_on"].to_list()
    columns = [c for c in pivot.columns if c != "trained_on"]
    values = pivot.select(columns).to_numpy().astype(float)
    figure, axis = plt.subplots(figsize=(1.4 * len(columns) + 3, 1.0 * len(rows) + 2.5))
    image = axis.imshow(values, cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    axis.set_yticks(range(len(rows)), rows)
    axis.set_xlabel("dominio di test")
    axis.set_ylabel("dominio di training")
    axis.set_title("TPR a FPR 1% per trasferimento fra domini")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="w")
    figure.colorbar(image, ax=axis, label="TPR @ FPR 1%")
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)
    logger.info("scritto %s", target)


def build_retriever(config: Config) -> DenseRetriever:
    documents = split_documents(
        [d for d in load_corpus(config.paths.demo_corpus) if d.meta.get("poisoned") != "1"],
        config.rag.chunk_chars,
        config.rag.chunk_overlap_chars,
    )
    logger.info("corpus della demo: %d passaggi", len(documents))
    return DenseRetriever(documents, config.rag.encoder_name)


def build_pipeline(config: Config, victim: str | None = None) -> RagPipeline:
    """Pipeline RAG sul modello vittima indicato; senza `victim` usa quello del file yaml."""
    from dataclasses import replace

    from shield.llm import LocalLLM, lmstudio_load

    llm_config = replace(config.llm, lmstudio_model=victim) if victim else config.llm
    if config.llm.backend != "transformers":
        # Caricamento esplicito ed esclusivo: lasciare che il server carichi il modello
        # alla prima richiesta non scarica gli altri, e due modelli non stanno in VRAM.
        seconds = lmstudio_load(config.llm.lmstudio_base_url, llm_config.lmstudio_model)
        logger.info("modello %s in memoria (%.1f s)", llm_config.lmstudio_model, seconds)
    llm = LocalLLM(llm_config)
    if victim and llm.backend == "lmstudio" and llm._served_model.lower() != victim.lower():
        raise RuntimeError(
            f"modello vittima {victim!r} non servito da LM Studio: scaricalo o correggi "
            "evaluation.victims in configs/config.yaml"
        )
    llm.healthcheck()
    return RagPipeline(build_retriever(config), llm, config.rag)


def model_slug(model_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in model_id.lower())


def corpus_calibration(config: Config) -> pl.DataFrame:
    """Chunk benigni della distribuzione di esercizio (il corpus RAG), per stimare tau."""
    docs = load_corpus(config.paths.demo_calibration)
    passages = split_documents(docs, config.rag.chunk_chars, config.rag.chunk_overlap_chars)
    return pl.DataFrame(
        {
            "text": [d.text for d in passages],
            "query": [""] * len(passages),
            "label": [0] * len(passages),
        }
    )


def calibrated_cascade(config: Config, detectors: dict[str, Classifier]) -> Config:
    """Sostituisce tau_lo e tau_hi con le soglie stimate sul corpus di esercizio.

    Le soglie stimate su PromptShield non si trasferiscono: l'1% di coda dei suoi
    benigni satura a ~1.0, quindi tau finisce sopra lo score della maggior parte degli
    attacchi reali. Vengono comunque calcolate e registrate, perche' il confronto fra
    le due e' un risultato da riportare."""
    if "shield_s2" not in detectors:
        return config
    from shield.config import CascadeConfig
    from shield.evaluation import score_frame

    detector = detectors["shield_s2"]
    fprs = config.evaluation.target_fprs
    remote = score_frame(detector, read(config, "calibration"), True)
    logger.info(
        "soglie da PromptShield (fuori distribuzione): tau_lo=%.8g tau_hi=%.8g",
        threshold_at_fpr(remote, max(fprs)),
        threshold_at_fpr(remote, min(fprs)),
    )
    local = score_frame(detector, corpus_calibration(config), True)
    low, high = threshold_at_fpr(local, max(fprs)), threshold_at_fpr(local, min(fprs))
    logger.info("soglie di esercizio (corpus RAG): tau_lo=%.8g tau_hi=%.8g", low, high)
    updated = dict(config.cascade.__dict__, tau_lo=low, tau_hi=high)
    object.__setattr__(config, "cascade", CascadeConfig(**updated))
    return config


def defenses(config: Config, detectors: dict[str, Classifier]) -> dict[str, Cascade | None]:
    classifier = detectors.get("shield_s2")
    return {
        "none": None,
        "s1": Cascade(config.cascade, None, None),
        "s1_s2": Cascade(config.cascade, classifier, None),
        "full": Cascade(config.cascade, classifier, None),  # il giudice viene innestato dopo
    }


def experiment_asr(config: Config, args: argparse.Namespace) -> None:
    """ASR per modello vittima, famiglia di vettori e configurazione della difesa.

    Un file grezzo per modello (`asr_raw_<modello>.csv`), cosi' i modelli si aggiungono
    uno alla volta senza rifare gli altri; le tabelle riassuntive si ricostruiscono da
    tutti i grezzi presenti con `asr-report`."""
    detectors = load_detectors(config, False)
    config = calibrated_cascade(config, detectors)
    victims = tuple(args.victim) if args.victim else config.evaluation.victims
    n = args.asr_samples or config.evaluation.asr_sample_size
    families = config.evaluation.attack_families
    for victim in victims:
        logger.info("=== modello vittima: %s ===", victim)
        pipeline = build_pipeline(config, victim)
        variants = defenses(config, detectors)
        variants["full"] = Cascade(config.cascade, detectors.get("shield_s2"), pipeline.llm)
        cases = sample_canary_cases(
            pipeline, n, config.data.positions, random.Random(config.seed), families
        )
        logger.info(
            "ASR: %d campioni per configurazione (%d per famiglia x %s), %d configurazioni",
            len(cases),
            n,
            "/".join(families),
            len(config.evaluation.configurations),
        )
        frames = []
        for name in config.evaluation.configurations:
            logger.info("configurazione %s...", name)
            frames.append(
                run_asr(pipeline, cases, variants[name], name, random.Random(config.seed), victim)
            )
            for family, part in frames[-1].group_by("family", maintain_order=True):
                logger.info(
                    "  [%s] ASR %.3f su %d campioni | recuperato spontaneamente nel %.0f%%",
                    family[0],
                    float(part["success"].mean() or 0.0),
                    part.height,
                    100 * float(part["poisoned_retrieved"].mean() or 0.0),
                )
        write(pl.concat(frames), config.paths.results / f"asr_raw_{model_slug(victim)}.csv")
        pipeline.llm.close()
    experiment_asr_report(config, args)
    logger.info("VRAM: %s", vram_report())


def experiment_asr_report(config: Config, args: argparse.Namespace) -> None:
    """Riassume tutti i grezzi `asr_raw_*.csv` in tre tabelle: per modello e famiglia, per
    posizione, per vettore."""
    files = sorted(config.paths.results.glob("asr_raw_*.csv"))
    if not files:
        logger.warning(
            "nessun asr_raw_*.csv in %s: esegui prima l'esperimento asr", config.paths.results
        )
        return
    raw = pl.concat([pl.read_csv(f) for f in files], how="vertical_relaxed")
    logger.info("ASR: %d righe da %d modelli", raw.height, raw["victim"].n_unique())
    write(
        summarise_asr(raw, ["victim", "family", "configuration"]), config.paths.results / "asr.csv"
    )
    write(
        summarise_asr(raw, ["victim", "family", "configuration", "position"]),
        config.paths.results / "asr_by_position.csv",
    )
    write(
        summarise_asr(raw, ["victim", "family", "vector", "configuration"]),
        config.paths.results / "asr_by_vector.csv",
    )


def experiment_utility(config: Config, args: argparse.Namespace) -> None:
    pipeline = build_pipeline(config)
    detectors = load_detectors(config, False)
    config = calibrated_cascade(config, detectors)
    cascade = Cascade(config.cascade, detectors.get("shield_s2"), pipeline.llm)
    frame = run_utility(
        pipeline, cascade, DEMO_QUERIES, args.utility_queries or config.evaluation.utility_queries
    )
    write(frame, config.paths.results / "utility.csv")
    threshold_reality_check(config, detectors)
    logger.info(
        "chunk rimossi a torto: %.3f | somiglianza media delle risposte: %.3f",
        float(frame["false_removal_rate"].mean() or 0.0),
        float(frame["answer_similarity"].mean() or 0.0),
    )


def threshold_reality_check(config: Config, detectors: dict[str, Classifier]) -> None:
    """Quanto vale davvero la soglia di esercizio fuori dal corpus sintetico.

    tau e' calibrato su recensioni generate dagli stessi modelli di frase del corpus:
    l'FPR misurato li' e' circolare. Questa tabella lo confronta con l'FPR sugli stessi
    chunk benigni reali di BIPIA, e va letta insieme a utility.csv."""
    if "shield_s2" not in detectors:
        return
    from shield.evaluation import score_frame

    detector = detectors["shield_s2"]
    tau = config.cascade.tau_hi
    rows = [
        {
            "corpus": "sintetico (recensioni della demo)",
            "domain": "demo",
            "n": corpus_calibration(config).height,
            "fpr_at_operating_tau": false_positive_rate(
                score_frame(detector, corpus_calibration(config), True), tau
            ),
        }
    ]
    real = read(config, "test").filter(pl.col("label") == 0)
    for domain in ["tutti", *sorted(real["domain"].unique().to_list())]:
        subset = real if domain == "tutti" else real.filter(pl.col("domain") == domain)
        rows.append(
            {
                "corpus": "reale (BIPIA benigni)",
                "domain": domain,
                "n": subset.height,
                "fpr_at_operating_tau": false_positive_rate(
                    score_frame(detector, subset, True), tau
                ),
            }
        )
    table = pl.DataFrame(rows).with_columns(pl.lit(tau).alias("tau"))
    write(table, config.paths.results / "threshold_reality_check.csv")
    for row in table.iter_rows(named=True):
        logger.info(
            "  FPR a tau=%.3g | %-34s %-9s n=%5d  %.1f%%",
            row["tau"],
            row["corpus"],
            row["domain"],
            row["n"],
            100 * row["fpr_at_operating_tau"],
        )


EXPERIMENTS = {
    "detector": experiment_detector,
    "generalization": experiment_generalization,
    "asr": experiment_asr,
    "asr-report": experiment_asr_report,
    "utility": experiment_utility,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default="all", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--asr-samples", type=int, default=0)
    parser.add_argument(
        "--victim",
        action="append",
        help="id LM Studio del modello vittima; ripetibile. Default: evaluation.victims",
    )
    parser.add_argument("--utility-queries", type=int, default=0)
    parser.add_argument("--lodo-detector", default="transformer", choices=["transformer", "tfidf"])
    parser.add_argument("--no-external", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    logger.info("seed=%d", config.seed)

    names = (
        [n for n in EXPERIMENTS if n != "asr-report"]
        if args.experiment == "all"
        else [args.experiment]
    )
    for name in names:
        logger.info("=== esperimento: %s ===", name)
        EXPERIMENTS[name](config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
