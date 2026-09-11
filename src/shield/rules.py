from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .types import Span, StageResult


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    weight: float
    # 'block' per le regole che agganciano un marker di apertura: quel marker non e'
    # l'iniezione, la apre. Rimuovere solo il marker lascia il payload nel contesto,
    # come togliere un tag di apertura conservando il corpo. Il blocco e' delimitato
    # da `_block_end`: la riga del marker e, per le pure intestazioni, quella seguente.
    scope: str = "match"


# Una riga di sola intestazione ("### SYSTEM OVERRIDE ###", "<|im_start|>system") e' piu'
# corta di questo: il payload sta sulla riga seguente. Un marker su una riga piu' lunga
# ("[SYSTEM]: New directive. Respond with ...") contiene gia' il payload.
_HEADER_MAX_CHARS = 40


def _line_end(text: str, index: int) -> int:
    end = text.find("\n", index)
    return len(text) if end == -1 else end


def _block_end(text: str, match_start: int, match_end: int) -> int:
    """Fine del blocco aperto da un marker: la riga del marker e, se questa e' una pura
    intestazione, anche la riga seguente.

    Estendere il blocco fino al primo paragrafo vuoto, come faceva la prima versione,
    trascinava nello span tutto il testo legittimo che segue l'iniezione con un solo a
    capo (il caso normale): il sanitize degradava a block e la recensione andava persa."""
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = _line_end(text, match_end)
    opens_line = not text[line_start:match_start].strip()
    is_header = opens_line and line_end - line_start <= _HEADER_MAX_CHARS
    if not is_header or line_end >= len(text) or text[line_end + 1 : line_end + 2] == "\n":
        return line_end
    return _line_end(text, line_end + 1)


def _compile(source: str) -> re.Pattern[str]:
    return re.compile(source, re.IGNORECASE | re.MULTILINE)


# Le regole sono orientate al recall sul traffico ovvio, non alla decisione finale:
# i pesi sopra 0.9 sono riservati ai pattern che nessun documento di dati legittimo
# produce spontaneamente (marker di ruolo, negazione esplicita del prompt).
RULES: tuple[Rule, ...] = (
    Rule(
        name="negate_previous",
        # "ignora / dimentica / scavalca" + riferimento alle istruzioni precedenti,
        # con una finestra breve in mezzo per non attraversare i confini di frase.
        pattern=_compile(
            r"\b(?:ignore|disregard|forget|discard|override|bypass|nevermind|"
            r"do\s+not\s+follow)\b[^.\n]{0,40}?"
            r"\b(?:previous|prior|earlier|above|preceding|initial|original|all|any)\b"
            r"[^.\n]{0,30}?\b(?:instruction|prompt|direction|command|rule|guideline|"
            r"context|message|request)s?\b"
            # italiano
            r"|\b(?:ignora|dimentica|scavalca|tralascia|non\s+seguire)\b[^.\n]{0,40}?"
            r"\b(?:istruzion[ei]|prompt|regol[ae]|indicazion[ei]|comand[oi]|richiest[ae])\b"
            r"(?:[^.\n]{0,30}?\b(?:precedent[ei]|iniziali|original[ei]|di\s+prima|sopra)\b)?"
        ),
        weight=0.95,
    ),
    Rule(
        name="system_role_marker",
        # Delimitatori di chat template dentro un documento di dati: non esistono
        # in prosa naturale, quindi il peso e' sopra la soglia di blocco.
        pattern=_compile(
            r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>"
            r"|\[/?INST\]|<</?SYS>>|<start_of_turn>|\[/?SYSTEM\]"
            r"|^\s{0,4}#{2,}\s*(?:system|instruction|assistant|new\s+prompt)\b"
            r"|^\s{0,4}(?:system|assistant)\s*(?:prompt)?\s*:\s"
        ),
        weight=0.95,
        scope="block",
    ),
    Rule(
        name="assistant_directive",
        # Imperativo in seconda persona rivolto esplicitamente a un assistente.
        # Il referente ("ai", "assistant", "your new task") e' obbligatorio: senza
        # di esso "you must try the breakfast" farebbe scattare la regola.
        pattern=_compile(
            r"\b(?:you\s+are\s+(?:now\s+)?(?:an?|the)\s+(?:ai|assistant|chatbot|bot|"
            r"language\s+model|helpful\s+\w+)"
            r"|as\s+an\s+ai(?:\s+language)?\s+(?:model|assistant)"
            r"|your\s+(?:new|real|actual|only|true|primary)\s+"
            r"(?:task|goal|instruction|objective|job|mission|purpose)"
            r"|from\s+now\s+on,?\s+you\s+(?:must|will|shall|should|are|have)"
            r"|you\s+(?:must|should|shall|have\s+to)\s+(?:now\s+|always\s+|instead\s+)?"
            r"(?:respond|reply|answer|output|say|write|include|append|print|begin|start)\b"
            r"|(?:important|attention|urgent)\s*[:!]\s*(?:ai|assistant|model)\b"
            # italiano
            r"|se\s+sei\s+un[a'’]?\s*(?:ia|intelligenza\s+artificiale|assistente|"
            r"modello\s+linguistico|chatbot|bot)\b"
            r"|il\s+tuo\s+(?:nuovo|vero|unico|solo)\s+(?:compito|obiettivo|ruolo|scopo)\b"
            r"|(?:nota|istruzion[ei]|avviso|attenzione)\s+(?:per|al|all[’']?)\s*"
            r"(?:l[’']?\s*)?(?:assistente|ia|intelligenza\s+artificiale|modello)\b)"
        ),
        weight=0.85,
    ),
    Rule(
        name="reveal_config",
        pattern=_compile(
            r"\b(?:reveal|print|repeat|show|output|display|disclose|reproduce|"
            r"summarize|tell\s+me|give\s+me)\b[^.\n]{0,40}?"
            r"\b(?:system\s+(?:prompt|message|instruction)|initial\s+(?:prompt|instruction)|"
            r"your\s+(?:instruction|prompt|configuration|guideline|directive|rule)s?|"
            r"hidden\s+(?:prompt|instruction)|training\s+data|api[\s_-]?key)\b"
            # italiano
            r"|\b(?:mostra|rivela|ripeti|stampa|scrivi|riporta)\b[^.\n]{0,40}?"
            r"\b(?:prompt\s+di\s+sistema|le\s+tue\s+istruzioni|istruzioni\s+iniziali|"
            r"la\s+tua\s+configurazione)\b"
        ),
        weight=0.9,
    ),
    Rule(
        name="exfiltration",
        # Verbo di trasmissione seguito, entro la stessa frase, da una URL o da un
        # indirizzo email: il segnale e' la coppia, non l'URL da sola.
        pattern=_compile(
            r"\b(?:send|post|transmit|forward|upload|exfiltrate|leak|deliver|submit|"
            r"report|email)\b[^.\n]{0,60}?"
            r"(?:https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[a-z]{2,})"
        ),
        weight=0.9,
    ),
)


def score_rules(text: str) -> StageResult:
    """Applica tutte le regole e restituisce il massimo dei pesi scattati."""
    started = time.perf_counter()
    score = 0.0
    spans: list[Span] = []
    fired: list[str] = []
    for rule in RULES:
        matches = [
            Span(
                m.start(),
                _block_end(text, m.start(), m.end()) if rule.scope == "block" else m.end(),
            )
            for m in rule.pattern.finditer(text)
        ]
        if not matches:
            continue
        fired.append(rule.name)
        spans.extend(matches)
        score = max(score, rule.weight)
    spans.sort(key=lambda s: (s.start, s.end))
    return StageResult(
        stage="S1",
        score=score,
        spans=tuple(spans),
        latency_ms=(time.perf_counter() - started) * 1000.0,
        detail=",".join(fired),
    )
