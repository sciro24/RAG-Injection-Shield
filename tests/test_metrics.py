from __future__ import annotations

import numpy as np

from shield.metrics import (
    bootstrap_ci,
    f1_at_threshold,
    false_positive_rate,
    roc_auc,
    threshold_at_fpr,
    tpr_at_fpr,
    wilson_ci,
)

RNG = np.random.default_rng(0)


def test_threshold_rispetta_il_fpr_target():
    benign = RNG.uniform(0, 1, 20_000)
    for target in (0.001, 0.01, 0.05):
        tau = threshold_at_fpr(benign, target)
        empirico = false_positive_rate(benign, tau)
        assert empirico <= target


def test_fpr_empirico_su_un_test_indipendente_resta_vicino_al_target():
    calibrazione = RNG.normal(0, 1, 50_000)
    test = RNG.normal(0, 1, 50_000)
    target = 0.01
    tau = threshold_at_fpr(calibrazione, target)
    empirico = false_positive_rate(test, tau)
    # errore di campionamento atteso: 3 deviazioni standard binomiali
    tolleranza = 3 * np.sqrt(target * (1 - target) / test.size)
    assert abs(empirico - target) <= tolleranza


def test_auc_su_separazione_perfetta_e_su_caso_casuale():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5] * 6)) == 0.5


def test_auc_coincide_con_scikit_learn():
    from sklearn.metrics import roc_auc_score

    y = RNG.integers(0, 2, 500)
    scores = RNG.uniform(0, 1, 500) + y * 0.5
    assert abs(roc_auc(y, scores) - roc_auc_score(y, scores)) < 1e-9


def test_tpr_at_fpr_su_separazione_perfetta():
    y = np.concatenate([np.zeros(1000), np.ones(1000)]).astype(int)
    scores = np.concatenate([RNG.uniform(0, 0.4, 1000), RNG.uniform(0.6, 1, 1000)])
    assert tpr_at_fpr(y, scores, 0.001) == 1.0


def test_tpr_at_fpr_cresce_col_budget_di_falsi_positivi():
    y = np.concatenate([np.zeros(5000), np.ones(5000)]).astype(int)
    scores = np.concatenate([RNG.normal(0, 1, 5000), RNG.normal(1.5, 1, 5000)])
    assert tpr_at_fpr(y, scores, 0.001) <= tpr_at_fpr(y, scores, 0.01)
    assert tpr_at_fpr(y, scores, 0.01) <= tpr_at_fpr(y, scores, 0.1)


def test_f1_at_threshold():
    y = np.array([0, 0, 1, 1])
    assert f1_at_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5) == 1.0
    assert f1_at_threshold(y, np.array([0.1, 0.2, 0.3, 0.4]), 0.5) == 0.0


def test_bootstrap_ci_contiene_la_stima_puntuale_ed_e_riproducibile():
    y = np.concatenate([np.zeros(500), np.ones(500)]).astype(int)
    scores = np.concatenate([RNG.normal(0, 1, 500), RNG.normal(2, 1, 500)])
    punto = roc_auc(y, scores)
    basso, alto = bootstrap_ci(roc_auc, y, scores, n=200, seed=1)
    assert basso <= punto <= alto
    assert bootstrap_ci(roc_auc, y, scores, n=200, seed=1) == (basso, alto)


def test_wilson_ci():
    basso, alto = wilson_ci(50, 100)
    assert basso < 0.5 < alto
    assert wilson_ci(0, 100)[0] == 0.0
    assert wilson_ci(100, 100)[1] == 1.0
    stretto = wilson_ci(500, 1000)
    largo = wilson_ci(5, 10)
    assert (stretto[1] - stretto[0]) < (largo[1] - largo[0])
