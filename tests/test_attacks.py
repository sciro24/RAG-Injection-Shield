from __future__ import annotations

import random

from shield.attacks import DECOY_HOTELS, HELDOUT_STYLES, TRAIN_STYLES, synth_attack
from shield.data import CANARY_FAMILIES, new_marker


def test_gli_stili_di_training_e_quelli_tenuti_fuori_sono_disgiunti():
    assert not set(TRAIN_STYLES) & set(HELDOUT_STYLES)


def test_ogni_attacco_sintetico_ha_il_segnaposto_riempito():
    rng = random.Random(0)
    for styles in (TRAIN_STYLES, HELDOUT_STYLES):
        for _ in range(200):
            attack = synth_attack(styles, rng)
            assert "{" not in attack and "}" not in attack
            assert len(attack) > 20


def test_il_marcatore_della_famiglia_steer_e_un_hotel_inventato():
    rng = random.Random(0)
    assert new_marker("steer", rng) in DECOY_HOTELS
    assert new_marker("baseline", rng).startswith("CNRY")


def test_ogni_vettore_ha_il_segnaposto_del_marcatore():
    for carriers in CANARY_FAMILIES.values():
        for template in carriers:
            assert "{canary}" in template
