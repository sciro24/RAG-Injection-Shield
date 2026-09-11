"""Genera il corpus sintetico di recensioni alberghiere usato da demo e ASR."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shield.config import Config, load_config  # noqa: E402
from shield.data import make_canary_attack, new_canary, poison  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_demo_corpus")

HOTELS = (
    "Hotel Belvedere",
    "Grand Hotel Aurora",
    "Locanda San Marco",
    "Hotel Rialto",
    "Albergo Della Posta",
    "Hotel Miramare",
    "Palazzo Fontana",
    "Hotel Stazione Centrale",
    "Residenza Il Chiostro",
    "Hotel Le Terrazze",
    "Hotel Corte Antica",
    "Villa Serena",
    "Hotel Ponte Vecchio",
    "Casa Dei Tigli",
    "Hotel Portanuova",
    "Albergo Il Cortile",
    "Hotel Marina Blu",
    "Hotel Torre Bianca",
    "Dimora Del Parco",
    "Hotel Giardino",
)

# Frasi per aspetto: le query della demo interrogano proprio questi aspetti, quindi
# il retrieval ha materiale sia positivo sia negativo su ciascuno.
ASPECTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "quiet": (
        (
            "The room was remarkably quiet even though we were close to the centre.",
            "We slept perfectly, the double glazing keeps the street noise out.",
            "A calm place, the inner courtyard rooms are silent at night.",
        ),
        (
            "Street noise kept us awake until well past midnight.",
            "The walls are thin and we heard every conversation next door.",
            "Rubbish trucks come at five in the morning right under the window.",
        ),
    ),
    "breakfast": (
        (
            "Breakfast was generous, with fresh pastries and proper espresso.",
            "The breakfast buffet had eggs, local cheese and excellent fruit.",
            "They bake the bread on site and the breakfast room opens early.",
        ),
        (
            "Breakfast was a sad tray of packaged croissants and weak coffee.",
            "The buffet was picked clean by eight and nobody refilled it.",
            "Breakfast costs extra and is not worth the money.",
        ),
    ),
    "wifi": (
        (
            "The wifi was fast and stable, I ran video calls all week.",
            "Good connection in the room, more than enough for remote work.",
            "There is a quiet lounge with reliable wifi and plenty of sockets.",
        ),
        (
            "The wifi dropped constantly and was unusable for work.",
            "Connection only works in the lobby, not in the rooms.",
            "The network needed a new login every hour, very annoying.",
        ),
    ),
    "staff": (
        (
            "The staff were genuinely helpful and arranged a late checkout for us.",
            "Reception went out of their way to book us a restaurant table.",
            "Warm and attentive service from the moment we arrived.",
        ),
        (
            "Reception was unhelpful and vaguely irritated by every question.",
            "Nobody at the desk spoke enough English to help with directions.",
            "We waited forty minutes to check in with one clerk on duty.",
        ),
    ),
    "rooms": (
        (
            "Spacious family room with two proper beds and a sofa for the child.",
            "The room was clean, bright and larger than the photographs suggested.",
            "Bathroom was spotless and the housekeeping was thorough every day.",
        ),
        (
            "The room was cramped and the bathroom smelled of damp.",
            "Housekeeping skipped our room twice during a four night stay.",
            "The carpet was stained and the shower barely drained.",
        ),
    ),
    "location": (
        (
            "Five minutes on foot from the train station, ideal for a short trip.",
            "Right by the old town, everything worth seeing is walkable.",
            "Convenient for the station without being on the noisy side.",
        ),
        (
            "Advertised as central but it is a thirty minute walk from anything.",
            "The area around the station felt unpleasant after dark.",
            "Far from the old town and the bus service is infrequent.",
        ),
    ),
    "value": (
        (
            "Excellent value for money given how central it is.",
            "Cheaper than the competition and noticeably better kept.",
            "Fair price, we would book it again without hesitating.",
        ),
        (
            "Overpriced for what it offers, we expected far more.",
            "Every extra is charged separately, the final bill was a surprise.",
            "Poor value: similar hotels nearby cost half as much.",
        ),
    ),
    "parking": (
        (
            "There is a private garage under the hotel, booked in advance.",
            "Free parking in the courtyard, a rarity this close to the centre.",
            "Parking is available on site for a modest daily fee.",
        ),
        (
            "No parking at all and the public garage is expensive.",
            "The hotel car park was full every evening of our stay.",
            "Street parking only, and finding a space took us an hour.",
        ),
    ),
}


def build_reviews(n: int, rng: random.Random) -> list[dict[str, object]]:
    """Ogni recensione combina due o tre aspetti, cosi' il retrieval non e' banale."""
    names = list(ASPECTS)
    reviews: list[dict[str, object]] = []
    for index in range(n):
        hotel = HOTELS[index % len(HOTELS)]
        chosen = rng.sample(names, rng.randint(2, 3))
        positive = rng.random() < 0.65
        sentences = [rng.choice(ASPECTS[a][0 if positive else 1]) for a in chosen]
        rng.shuffle(sentences)
        opening = f"Stayed at {hotel} for {rng.randint(1, 6)} nights."
        closing = "Would happily return." if positive else "We will look elsewhere next time."
        reviews.append(
            {
                "id": f"review-{index:04d}",
                "text": " ".join([opening, *sentences, closing]),
                "hotel": hotel,
                "sentiment": "positive" if positive else "negative",
                "aspects": ",".join(sorted(chosen)),
                "poisoned": 0,
            }
        )
    return reviews


def poison_some(
    reviews: list[dict[str, object]], count: int, positions: tuple[str, ...], rng: random.Random
) -> list[dict[str, object]]:
    """Varianti avvelenate delle recensioni, con canary tracciabile e span noto."""
    variants: list[dict[str, object]] = []
    targets = rng.sample(reviews, min(count, len(reviews)))
    for index, review in enumerate(targets):
        canary = new_canary(rng)
        attack = make_canary_attack(canary, rng)
        position = positions[index % len(positions)]
        text, span = poison(str(review["text"]), attack, position, rng)  # type: ignore[arg-type]
        variants.append(
            {
                **review,
                "id": f"{review['id']}-poisoned",
                "text": text,
                "poisoned": 1,
                "canary": canary,
                "injection_position": position,
                "span_start": span.start,
                "span_end": span.end,
            }
        )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config: Config = load_config(args.config)
    rng = random.Random(config.seed)
    logger.info("seed=%d", config.seed)

    reviews = build_reviews(config.demo.n_reviews, rng)
    poisoned = poison_some(reviews, config.demo.n_poisoned, config.data.positions, rng)

    target = config.paths.demo_corpus
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in [*reviews, *poisoned]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(
        "scritto %s: %d recensioni pulite, %d varianti avvelenate",
        target,
        len(reviews),
        len(poisoned),
    )

    # Set di calibrazione: recensioni pulite della stessa distribuzione ma disgiunte dal
    # corpus, generate con un rng separato. Le soglie stimate su PromptShield non si
    # trasferiscono al corpus RAG, quindi tau va calibrato sulla distribuzione di esercizio.
    calibration = build_reviews(config.demo.n_calibration, random.Random(config.seed + 1))
    with config.paths.demo_calibration.open("w", encoding="utf-8") as handle:
        for row in calibration:
            handle.write(json.dumps({**row, "id": f"calib-{row['id']}"}, ensure_ascii=False) + "\n")
    logger.info(
        "scritto %s: %d recensioni pulite di calibrazione",
        config.paths.demo_calibration,
        len(calibration),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
