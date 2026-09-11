from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Metric = Callable[[NDArray[np.int_], NDArray[np.float64]], float]

# La regola di decisione del progetto e' `score > tau`, non `>=`: con score
# discreti o saturi a 1.0 la disuguaglianza larga farebbe collassare tutti gli
# ex aequo dalla parte dei positivi, e il FPR misurato supererebbe il target.


def threshold_at_fpr(scores_benign: NDArray[np.float64], target_fpr: float) -> float:
    """Quantile (1 - target_fpr) degli score benigni: la soglia che li lascia passare."""
    scores = np.asarray(scores_benign, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("il set di calibrazione e' vuoto")
    allowed = int(np.floor(scores.size * target_fpr))
    ordered = np.sort(scores)[::-1]
    return float(ordered[min(allowed, scores.size - 1)])


def false_positive_rate(scores_benign: NDArray[np.float64], threshold: float) -> float:
    scores = np.asarray(scores_benign, dtype=np.float64)
    return float(np.mean(scores > threshold)) if scores.size else 0.0


def tpr_at_fpr(y_true: NDArray[np.int_], y_score: NDArray[np.float64], target_fpr: float) -> float:
    """Recall sui positivi alla soglia che sui negativi dello stesso insieme da' target_fpr."""
    y_true = np.asarray(y_true, dtype=np.int_)
    y_score = np.asarray(y_score, dtype=np.float64)
    benign = y_score[y_true == 0]
    malicious = y_score[y_true == 1]
    if benign.size == 0 or malicious.size == 0:
        return float("nan")
    tau = threshold_at_fpr(benign, target_fpr)
    return float(np.mean(malicious > tau))


def roc_auc(y_true: NDArray[np.int_], y_score: NDArray[np.float64]) -> float:
    """AUC come statistica di Mann-Whitney sui ranghi, con gestione degli ex aequo."""
    y_true = np.asarray(y_true, dtype=np.int_)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    sorted_scores = y_score[order]
    start = 0
    for index in range(1, y_score.size + 1):
        if index == y_score.size or sorted_scores[index] != sorted_scores[start]:
            ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    rank_sum = float(ranks[y_true == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def f1_at_threshold(
    y_true: NDArray[np.int_], y_score: NDArray[np.float64], threshold: float
) -> float:
    y_true = np.asarray(y_true, dtype=np.int_)
    predicted = np.asarray(y_score, dtype=np.float64) > threshold
    true_positives = float(np.sum(predicted & (y_true == 1)))
    if true_positives == 0:
        return 0.0
    precision = true_positives / float(predicted.sum())
    recall = true_positives / float((y_true == 1).sum())
    return 2 * precision * recall / (precision + recall)


def bootstrap_ci(
    metric_fn: Metric,
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervallo percentile su n ricampionamenti con reinserimento."""
    y_true = np.asarray(y_true, dtype=np.int_)
    y_score = np.asarray(y_score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    size = y_true.size
    samples = np.empty(n, dtype=np.float64)
    for index in range(n):
        picks = rng.integers(0, size, size)
        samples[index] = metric_fn(y_true[picks], y_score[picks])
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(finite, alpha / 2)),
        float(np.quantile(finite, 1 - alpha / 2)),
    )


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervallo di Wilson per una proporzione: stabile anche con pochi successi."""
    if total == 0:
        return (float("nan"), float("nan"))
    # 1.959964 e' il quantile 0.975 della normale standard; alpha diverso da 0.05
    # viene riscalato con l'approssimazione di Hastings usata da scipy.
    z = 1.959964 if abs(alpha - 0.05) < 1e-9 else float(_normal_quantile(1 - alpha / 2))
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _normal_quantile(p: float) -> float:
    # Approssimazione razionale di Beasley-Springer-Moro, sufficiente per gli
    # intervalli riportati (errore < 1e-6 nella coda che ci interessa).
    a = [-39.696830, 220.946098, -275.928510, 138.357751, -30.664798, 2.506628]
    b = [-54.476098, 161.585836, -155.698979, 66.801311, -13.280681]
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        return num / den
    r = p if q < 0 else 1 - p
    r = float(np.sqrt(-np.log(r)))
    value = (2.30753 + 0.27061 * r) / (1 + (0.99229 + 0.04481 * r) * r)
    value = r - value
    return -value if q < 0 else value
