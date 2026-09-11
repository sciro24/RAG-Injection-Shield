from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from urllib.parse import unquote

# Profondita' massima della decodifica ricorsiva. Due passate coprono il caso
# realistico (base64 dentro percent-encoding, o doppio base64) senza esporre il
# normalizzatore a un attacco di amplificazione: ogni livello puo' espandere il
# testo, e una profondita' illimitata su input ostile e' una DoS.
_MAX_DECODE_DEPTH = 2

# Un blocco piu' corto di questo non e' distinguibile da una parola normale:
# decodificarlo produrrebbe rumore e falsi positivi sulle regole a valle.
_MIN_ENCODED_BLOCK = 16

# Frazione minima di caratteri stampabili perche' una decodifica sia accettata:
# sotto questa soglia il blocco era dati binari, non testo offuscato.
_PRINTABLE_RATIO = 0.9

_ZERO_WIDTH = "​‌‍⁠﻿᠎"

# Omoglifi cirillici e greci verso il latino. Tabella statica e volutamente
# corta: copre i caratteri con cui si scrivono "ignore", "system", "admin".
# fmt: off
_HOMOGLYPHS: dict[str, str] = {
    # cirillico maiuscolo
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ј": "J",
    "Ѕ": "S", "Ғ": "F",
    # cirillico minuscolo
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "г": "r", "п": "n", "ь": "b",
    # greco maiuscolo
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # greco minuscolo
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "κ": "k",
    "ι": "i", "ϲ": "c", "ε": "e", "γ": "y", "χ": "x",
}
# fmt: on

_B64_RE = re.compile(r"[A-Za-z0-9+/]{" + str(_MIN_ENCODED_BLOCK) + r",}={0,2}")
# Un blocco percent-encoded: almeno una tripletta %XX, eventualmente inframmezzata
# da caratteri ammessi in una URL.
_PCT_RE = re.compile(
    r"(?:%[0-9A-Fa-f]{2}|[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=-]){" + str(_MIN_ENCODED_BLOCK) + r",}"
)
_SPACES_RE = re.compile(r"[^\S\n]{2,}")
_NEWLINES_RE = re.compile(r"\n{3,}")


def _strip_invisible(text: str) -> tuple[str, bool]:
    kept: list[str] = []
    removed = False
    for char in text:
        if char in ("\n", "\t"):
            kept.append(char)
            continue
        if char in _ZERO_WIDTH or unicodedata.category(char) in ("Cf", "Cc"):
            removed = True
            continue
        kept.append(char)
    return "".join(kept), removed


def _map_homoglyphs(text: str) -> tuple[str, bool]:
    if not any(char in _HOMOGLYPHS for char in text):
        return text, False
    return "".join(_HOMOGLYPHS.get(char, char) for char in text), True


def _collapse_whitespace(text: str) -> str:
    return _NEWLINES_RE.sub("\n\n", _SPACES_RE.sub(" ", text))


def _is_printable_text(candidate: str) -> bool:
    if not candidate.strip():
        return False
    printable = sum(1 for char in candidate if char.isprintable() or char in "\n\t")
    return printable / len(candidate) >= _PRINTABLE_RATIO


def _try_base64(block: str) -> str | None:
    padded = block + "=" * (-len(block) % 4)
    try:
        decoded = base64.b64decode(padded, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    return decoded if _is_printable_text(decoded) else None


def _try_percent(block: str) -> str | None:
    if "%" not in block:
        return None
    decoded = unquote(block, errors="strict")
    if decoded == block or not _is_printable_text(decoded):
        return None
    return decoded


def _decode_once(text: str) -> tuple[str, bool]:
    """Sostituisce in linea i blocchi codificati decodificabili in testo leggibile."""
    found = False

    def replace(match: re.Match[str], decoder: object) -> str:
        nonlocal found
        decoded = decoder(match.group(0))  # type: ignore[operator]
        if decoded is None:
            return match.group(0)
        found = True
        return decoded

    text = _PCT_RE.sub(lambda m: replace(m, _try_percent), text)
    text = _B64_RE.sub(lambda m: replace(m, _try_base64), text)
    return text, found


def canonicalize(text: str) -> tuple[str, dict[str, bool]]:
    """Restituisce il testo canonico e i flag delle trasformazioni applicate."""
    flags = {"had_zero_width": False, "had_homoglyph": False, "had_encoded_block": False}
    current = text
    for _ in range(_MAX_DECODE_DEPTH + 1):
        current = unicodedata.normalize("NFKC", current)
        current, stripped = _strip_invisible(current)
        current, mapped = _map_homoglyphs(current)
        current = _collapse_whitespace(current)
        flags["had_zero_width"] |= stripped
        flags["had_homoglyph"] |= mapped
        current, decoded = _decode_once(current)
        if not decoded:
            break
        flags["had_encoded_block"] = True
    return current, flags
