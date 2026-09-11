"""Figure per la presentazione, generate dai CSV in reports/results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shield.config import load_config  # noqa: E402

BLUE, ORANGE, AQUA, TEXT, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#0b0b0b",
    "#52514e",
    "#e6e5e1",
)
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]
MODELS = {
    "google/gemma-4-e4b": "Gemma 4 E4B",
    "qwen3.8-4b-distill": "Qwen3.8 4B",
    "mistralai/ministral-3-3b": "Ministral 3 3B",
}
MODEL_TICKS = ["Gemma\n4 E4B", "Qwen3.8\n4B distill", "Ministral\n3 3B"]
FAMILIES = {"baseline": "espliciti", "adaptive": "adattivi", "steer": "di deviazione"}
CONFIGS = ["none", "s1", "s1_s2", "full"]
CONFIG_LABELS = {"none": "nessuna", "s1": "S1", "s1_s2": "S1+S2", "full": "S1+S2+S3"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "figure.dpi": 200,
    }
)


def tidy(ax: plt.Axes, ymax: float = 1.0) -> None:
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def label_bars(ax: plt.Axes, bars: object, fmt: str = "{:.2f}") -> None:
    for bar in bars:  # type: ignore[attr-defined]
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )


def fig_asr_none_vs_full(results: Path, out: Path) -> None:
    a = pl.read_csv(results / "asr.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.36
    for ax, (fam, title) in zip(axes, FAMILIES.items(), strict=True):
        none = [
            a.filter(
                (pl.col("victim") == v)
                & (pl.col("family") == fam)
                & (pl.col("configuration") == "none")
            )["asr"][0]
            for v in MODELS
        ]
        full = [
            a.filter(
                (pl.col("victim") == v)
                & (pl.col("family") == fam)
                & (pl.col("configuration") == "full")
            )["asr"][0]
            for v in MODELS
        ]
        b1 = ax.bar(x - width / 2, none, width, color=BLUE, label="senza difesa")
        b2 = ax.bar(x + width / 2, full, width, color=ORANGE, label="cascata completa")
        label_bars(ax, b1)
        label_bars(ax, b2)
        ax.set_xticks(x, MODEL_TICKS)
        ax.set_title(f"attacchi {title}")
        tidy(ax, 0.45)
    axes[0].set_ylabel("ASR (tasso di successo degli attacchi)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Senza difesa gli attacchi passano su ogni modello; con la cascata quasi mai",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "asr_none_vs_full.png")
    plt.close(fig)


def fig_asr_by_stage(results: Path, out: Path) -> None:
    a = pl.read_csv(results / "asr.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, (fam, title) in zip(axes, FAMILIES.items(), strict=True):
        means = [
            float(
                a.filter((pl.col("family") == fam) & (pl.col("configuration") == c))["asr"].mean()
            )
            for c in CONFIGS
        ]
        bars = ax.bar([CONFIG_LABELS[c] for c in CONFIGS], means, 0.6, color=BLUE_RAMP)
        label_bars(ax, bars)
        ax.set_title(f"attacchi {title}")
        tidy(ax, 0.35)
    axes[0].set_ylabel("ASR, media sui tre modelli")
    fig.suptitle(
        "Contributo di ogni stadio: le regole dimezzano, il classificatore azzera",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "asr_by_stage.png")
    plt.close(fig)


def fig_detector_tpr(results: Path, out: Path) -> None:
    d = pl.read_csv(results / "detector_comparison.csv")
    names = {
        "shield_s2": "S2 (questo progetto)",
        "tfidf_lr": "TF-IDF + LR",
        "external_hf": "riferimento pubblico",
        "rules_s1": "sole regole S1",
    }
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(names))
    width = 0.36
    for i, (test_set, colour, label) in enumerate(
        (("test", BLUE, "attacchi reali (BIPIA)"), ("stealth", ORANGE, "stili mai visti (stealth)"))
    ):
        vals = [
            d.filter((pl.col("test_set") == test_set) & (pl.col("detector") == k))[
                "tpr_at_fpr_0.01"
            ][0]
            for k in names
        ]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=colour, label=label)
        label_bars(ax, bars)
    ax.set_xticks(x, list(names.values()))
    ax.set_ylabel("TPR @ FPR 1%")
    tidy(ax, 1.05)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Nove iniezioni su dieci rilevate, anche su formulazioni mai viste")
    fig.tight_layout()
    fig.savefig(out / "detector_tpr.png")
    plt.close(fig)


def fig_threshold_fpr(results: Path, out: Path) -> None:
    t = pl.read_csv(results / "threshold_reality_check.csv")
    labels = [
        "recensioni demo\n(calibrazione)",
        "BIPIA tutti",
        "abstract",
        "table",
        "code",
        "email",
    ]
    order = ["demo", "tutti", "abstract", "table", "code", "email"]
    vals = [float(t.filter(pl.col("domain") == dom)["fpr_at_operating_tau"][0]) for dom in order]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(labels, vals, 0.6, color=[ORANGE] + [BLUE] * 5)
    label_bars(ax, bars, "{:.1%}")
    ax.set_ylabel("falsi positivi alla soglia della demo")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    tidy(ax, 1.12)
    ax.set_title("La soglia calibrata su un corpus non vale su un altro")
    fig.tight_layout()
    fig.savefig(out / "threshold_fpr.png")
    plt.close(fig)


def fig_vulnerability_by_vector(results: Path, out: Path) -> None:
    b = pl.read_csv(results / "asr_by_vector.csv").filter(pl.col("configuration") == "none")
    cols, ticks = [], []
    for fam, short in (("baseline", "E"), ("adaptive", "A"), ("steer", "D")):
        for vec in sorted(b.filter(pl.col("family") == fam)["vector"].unique().to_list()):
            cols.append((fam, vec))
            ticks.append(f"{short}{vec}")
    matrix = np.array(
        [
            [
                float(
                    b.filter(
                        (pl.col("victim") == v) & (pl.col("family") == f) & (pl.col("vector") == n)
                    )["asr"][0]
                )
                for f, n in cols
            ]
            for v in MODELS
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 2.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f5f8fd", "#cde2fb", "#5598e7", "#0d366b"]
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ticks)), ticks)
    ax.set_yticks(range(len(MODELS)), list(MODELS.values()))
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.5 else TEXT,
            )
    for boundary in (7.5, 13.5):
        ax.axvline(boundary, color="white", linewidth=3)
    ax.set_title("Quali attacchi funzionano senza difesa (E espliciti, A adattivi, D deviazione)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="ASR")
    fig.tight_layout()
    fig.savefig(out / "vulnerability_by_vector.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    results, out = config.paths.results, config.paths.figures
    out.mkdir(parents=True, exist_ok=True)
    for fn in (
        fig_asr_none_vs_full,
        fig_asr_by_stage,
        fig_detector_tpr,
        fig_threshold_fpr,
        fig_vulnerability_by_vector,
    ):
        fn(results, out)
        print("scritto", out / (fn.__name__.removeprefix("fig_") + ".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
