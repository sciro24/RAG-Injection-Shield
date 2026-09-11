"""Demo Streamlit: un assistente di viaggio con e senza RAG Injection Shield.

Un solo flusso, dall'alto in basso: si sceglie una domanda e un'istruzione da nascondere
nella recensione piu' pertinente, si preme Esegui, e si vedono le due risposte dello stesso
LLM, senza e con la cascata in mezzo. Sotto, cosa ha fatto la difesa su ogni passaggio.
"""

from __future__ import annotations

import html
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shield.cascade import Cascade  # noqa: E402
from shield.classifier import TransformerClassifier  # noqa: E402
from shield.config import CascadeConfig, Config, load_config  # noqa: E402
from shield.data import CANARY_FAMILIES, new_canary, poison  # noqa: E402
from shield.evaluation import DEMO_QUERIES, score_frame  # noqa: E402
from shield.llm import (  # noqa: E402
    RAG_PROMPT,
    LocalLLM,
    candidate_urls,
    lmstudio_available,
    lmstudio_catalog,
    lmstudio_load,
)
from shield.metrics import threshold_at_fpr  # noqa: E402
from shield.rag import DenseRetriever, apply_defense, load_corpus, split_documents  # noqa: E402
from shield.types import Document, Verdict  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "config.yaml"
SEPARATOR = "\n\n---\n\n"
FREE_QUERY = "Scrivo io la domanda…"
NO_ATTACK = "Nessuna: recensioni pulite"
FAMILIES = {"baseline": "Classica", "adaptive": "Adattiva"}
ATTACKS = [
    (family, i) for family, carriers in CANARY_FAMILIES.items() for i in range(len(carriers))
]
POSITIONS = {"start": "inizio", "middle": "metà", "end": "fine"}
ACTIONS = {
    "pass": ("passa", "green"),
    "sanitize": ("ripulita", "orange"),
    "block": ("scartata", "red"),
}
MODES = {"sanitize": "rimuovi solo l'istruzione", "block": "scarta l'intera recensione"}
STAGE_NAMES = {"S1": "regole", "S2": "classificatore", "S3": "giudice LLM"}

CSS = """<style>
  .block-container { padding-top: 2.5rem; max-width: 1080px; }
  .rs-passage { font-size: .9rem; line-height: 1.6; white-space: pre-wrap; margin-top: .3rem; }
  .rs-passage del { background: rgba(239, 68, 68, .14); color: inherit; padding: 0 2px;
                    border-radius: 3px; text-decoration-color: rgba(239, 68, 68, .85); }
  .rs-answer { font-size: 1rem; line-height: 1.6; min-height: 4.5rem; }
</style>"""


@dataclass(frozen=True)
class Settings:
    query: str
    victim: str  # modello che risponde, come lo espone LM Studio
    family: str
    carrier: int | None  # None: nessun attacco
    position: str
    force: bool
    mode: str


@dataclass(frozen=True)
class Run:
    """Esito di un'esecuzione, conservato in sessione: cosi' aprire un expander o
    cambiare un'opzione non cancella le risposte appena generate."""

    settings: Settings
    docs: tuple[Document, ...]
    canary: str | None
    rank: int | None  # rango del documento avvelenato: -1 forzato, None non recuperato
    without: str
    with_defense: str
    prompts: tuple[str, str]  # prompt esatti inviati al modello, senza e con difesa
    verdicts: tuple[Verdict, ...]


# --- caricamento -----------------------------------------------------------------


@st.cache_resource(show_spinner="Carico corpus, retriever e classificatore…")
def load_stack(path: str) -> tuple:
    config = load_config(path)
    clean = [d for d in load_corpus(config.paths.demo_corpus) if d.meta.get("poisoned") != "1"]
    passages = split_documents(clean, config.rag.chunk_chars, config.rag.chunk_overlap_chars)
    retriever = DenseRetriever(passages, config.rag.encoder_name)
    checkpoint = config.classifier.checkpoint_dir
    classifier = (
        TransformerClassifier(checkpoint, config.classifier.max_length, 32)
        if checkpoint.exists()
        else None
    )
    return config, retriever, classifier, calibrate(config, classifier)


@st.cache_resource(show_spinner="Collego il modello…")
def load_llm(path: str, victim: str) -> LocalLLM:
    """Un wrapper per modello vittima: fa anche da giudice S3, come negli esperimenti."""
    config = load_config(path)
    llm = LocalLLM(replace(config.llm, lmstudio_model=victim))
    llm.healthcheck()
    return llm


def served_models(config: Config) -> list[str]:
    """I modelli vittima degli esperimenti (`evaluation.victims`) presenti in LM Studio,
    nell'ordine del file yaml: la demo mostra gli stessi modelli delle tabelle."""
    if config.llm.backend == "transformers":
        return [config.llm.model_name]
    found = lmstudio_catalog(config.llm.lmstudio_base_url)
    victims = [m for m in config.evaluation.victims if m in found]
    if not victims:
        st.error(
            "Nessuno dei modelli in `evaluation.victims` è presente in LM Studio. "
            "Disponibili: " + ", ".join(f"`{m}`" for m in found)
        )
        st.stop()
    return victims


def ensure_loaded(config: Config, victim: str) -> None:
    """Carica il modello scelto in LM Studio scaricando gli altri: uno solo in memoria,
    come negli esperimenti. Non e' in cache perche' lo stato vive in LM Studio."""
    if config.llm.backend == "transformers":
        return
    if lmstudio_catalog(config.llm.lmstudio_base_url).get(victim):
        return
    with st.spinner(f"Carico `{victim}` in LM Studio e scarico gli altri modelli…"):
        seconds = lmstudio_load(config.llm.lmstudio_base_url, victim)
    st.toast(f"{victim} caricato in {seconds:.1f} s", icon="✅")


@st.cache_resource(show_spinner="Calibro le soglie sul corpus di esercizio…")
def calibrate(config: Config, _classifier: TransformerClassifier | None) -> tuple[float, float]:
    """Stessa calibrazione degli esperimenti: le soglie del file yaml sono solo un default.

    Senza questo passaggio la demo userebbe tau_hi = 0.8 mentre gli score reali sulle
    recensioni valgono ~2e-05, e mostrerebbe un sistema diverso da quello valutato."""
    if _classifier is None:
        return config.cascade.tau_lo, config.cascade.tau_hi
    docs = load_corpus(config.paths.demo_calibration)
    frame = pl.DataFrame({"text": [d.text for d in docs], "query": [""] * len(docs)})
    scores = score_frame(_classifier, frame.with_columns(pl.lit(0).alias("label")), True)
    fprs = config.evaluation.target_fprs
    return threshold_at_fpr(scores, max(fprs)), threshold_at_fpr(scores, min(fprs))


def require_llm_server(config: Config) -> None:
    """Senza un server locale il backend ripiega su transformers, che su questa GPU non
    carica il modello: meglio fermarsi subito con un messaggio utile."""
    if config.llm.backend == "transformers":
        return
    urls = candidate_urls(config.llm)
    if any(lmstudio_available(url) for url in urls):
        return
    st.error(
        "Nessun server LLM in ascolto su "
        + " o ".join(f"`{u}`" for u in urls)
        + ". Avvia LM Studio, carica `gemma-4-E4B-it` e attiva il server locale, "
        "poi ricarica la pagina."
    )
    st.stop()


# --- formattazione ----------------------------------------------------------------


def format_score(score: float) -> str:
    """Gli score utili vivono fra 1e-06 e 1: con %.3f sarebbero tutti 0.000."""
    if score >= 0.01:
        return f"{score:.3f}"
    return f"{score:.1e}" if score > 0 else "0"


def attack_label(attack: tuple[str, int]) -> str:
    family, index = attack
    first_line = CANARY_FAMILIES[family][index].splitlines()[0].replace("{canary}", "…")
    return f"{FAMILIES[family]} {index + 1} · {first_line[:70]}"


def explain(verdict: Verdict, taus: tuple[float, float]) -> str:
    """Una riga in linguaggio naturale su come la cascata e' arrivata al verdetto."""
    parts: list[str] = []
    for step in verdict.trace:
        if step.stage == "S0":
            if step.detail:
                flags = step.detail.replace("had_", "").replace("_", " ")
                parts.append(f"S0 normalizzazione: {flags}")
            continue
        name = f"{step.stage} {STAGE_NAMES[step.stage]}"
        if step.stage == "S1":
            fired = f"pattern «{step.detail}»" if step.detail else "nessun pattern"
            parts.append(f"{name}: {fired}")
        elif step.stage == "S2":
            score = format_score(step.score)
            if step.score < taus[0]:
                parts.append(f"{name}: score {score} < τ_lo {format_score(taus[0])}")
            elif step.score > taus[1]:
                parts.append(f"{name}: score {score} > τ_hi {format_score(taus[1])}")
            else:
                parts.append(f"{name}: score {score} nella banda grigia, escalation a S3")
        else:
            parts.append(f"{name}: {step.detail}")
    decided = f"deciso da {verdict.decided_by}"
    return " · ".join([*parts, decided, f"{verdict.total_latency_ms:.0f} ms"])


def marked_text(verdict: Verdict) -> str:
    """Testo canonico con gli span rimossi barrati: cio' che l'LLM non ha letto."""
    text, cursor, pieces = verdict.canonical_text, 0, []
    for span in verdict.spans:
        pieces.append(html.escape(text[cursor : span.start]))
        pieces.append(f"<del>{html.escape(text[span.start : span.end])}</del>")
        cursor = max(cursor, span.end)
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


# --- controlli --------------------------------------------------------------------


def controls(config: Config, models: list[str]) -> Settings:
    left, right = st.columns([5, 4], gap="large")
    with left:
        choice = st.selectbox("Domanda all'assistente", [*DEMO_QUERIES, FREE_QUERY])
        query = (
            st.text_input("La tua domanda", placeholder="Which hotel has the best breakfast?")
            if choice == FREE_QUERY
            else choice
        )
        victim = st.selectbox(
            "Modello che risponde",
            models,
            help="I modelli serviti da LM Studio. Lo stesso modello genera entrambe le "
            "risposte e fa da giudice S3. Un modello piccolo o poco allineato obbedisce "
            "più spesso alle istruzioni nascoste: è il caso dei RAG locali.",
        )
    with right:
        options = [NO_ATTACK, *ATTACKS]
        # Predefinito il vettore classico 2: e' l'unico che compromette Gemma in ogni
        # posizione, quindi la demo mostra subito un attacco riuscito e la difesa che lo ferma.
        chosen = st.selectbox(
            "Istruzione nascosta nella recensione più pertinente",
            options,
            index=2,
            format_func=lambda o: o if o == NO_ATTACK else attack_label(o),
            help="Ogni istruzione chiede al modello di scrivere una parola d'ordine casuale "
            "(canary): se compare nella risposta, l'attacco è riuscito. Le *classiche* usano "
            "marker e imperativi espliciti; le *adattive* sono scritte conoscendo le regole "
            "S1 e si travestono da contenuto.",
        )
        position = st.segmented_control(
            "Posizione nella recensione",
            list(POSITIONS),
            format_func=POSITIONS.get,
            default="end",
            disabled=chosen == NO_ATTACK,
        )
    with st.expander("Opzioni"):
        force = st.checkbox(
            "Tieni la recensione avvelenata nel contesto anche se il retrieval la scarta",
            value=True,
            help="L'istruzione iniettata sposta l'embedding della recensione, che spesso "
            "esce dalle prime k. Senza questa opzione l'attacco può fallire prima ancora "
            "di arrivare alla difesa.",
        )
        mode = st.radio(
            "Quando una recensione è sospetta",
            list(MODES),
            format_func=MODES.get,
            horizontal=True,
        )
    family, carrier = ("baseline", None) if chosen == NO_ATTACK else chosen
    return Settings(query.strip(), victim, family, carrier, position or "end", force, mode)


# --- esecuzione -------------------------------------------------------------------


def prepare(
    retriever: DenseRetriever, settings: Settings, k: int
) -> tuple[list[Document], str | None, int | None]:
    """Recupera i passaggi, avvelena il piu' pertinente e stabilisce se resta fra i primi k.

    Non tocca l'indice condiviso: ricodifica solo il documento avvelenato e lo confronta
    con gli altri candidati."""
    found = retriever.retrieve(settings.query, k + 1)
    if settings.carrier is None:
        return found[:k], None, None
    target, others = found[0], found[1:]
    rng = random.Random()
    canary = new_canary(rng)
    injected = CANARY_FAMILIES[settings.family][settings.carrier].format(canary=canary)
    text, _ = poison(target.text, injected, settings.position, rng)  # type: ignore[arg-type]
    poisoned = Document(target.id, text, dict(target.meta, poisoned="1"))
    query_vector = retriever.encode([settings.query])[0]
    poisoned_score = float(retriever.encode([text])[0] @ query_vector)
    other_scores = retriever.encode([d.text for d in others]) @ query_vector
    rank = int((other_scores > poisoned_score).sum())
    if rank < k:
        return [*others[:rank], poisoned, *others[rank : k - 1]], canary, rank
    if settings.force:
        return [poisoned, *others[: k - 1]], canary, -1
    return others[:k], canary, None


def build_cascade(
    config: Config,
    classifier: TransformerClassifier | None,
    llm: LocalLLM,
    settings: Settings,
    taus: tuple[float, float],
) -> Cascade:
    policy = CascadeConfig(
        **{
            **config.cascade.__dict__,
            "high_score_action": settings.mode,
            "tau_lo": taus[0],
            "tau_hi": taus[1],
        }
    )
    return Cascade(policy, classifier, llm)


def generate(
    llm: LocalLLM, docs: list[Document], query: str, cascade: Cascade | None
) -> tuple[Iterator[str], tuple[Verdict, ...], str]:
    """Stesso template, stesso modello, stessa temperatura per le due colonne: l'unica
    differenza e' il contesto, grezzo senza difesa e sanitizzato con la cascata."""
    passages, verdicts = apply_defense(docs, query, cascade)
    context = SEPARATOR.join(passages) if passages else "(no usable context)"
    prompt = RAG_PROMPT.format(context=context, question=query)
    return llm.stream(prompt, llm.config.max_new_tokens), verdicts, prompt


# --- rendering --------------------------------------------------------------------


def show_answer(title: str, source: str | Iterator[str], canary: str | None, prompt: str) -> str:
    """Una colonna di risposta: in streaming alla prima esecuzione, dal testo salvato poi.

    Sotto, il prompt esatto inviato al modello: la risposta e' autentica e verificabile."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if isinstance(source, str):
            body = f"<div class='rs-answer'>{html.escape(source)}</div>"
            st.markdown(body, unsafe_allow_html=True)
            text = source
        else:
            streamed = st.write_stream(source)
            text = streamed if isinstance(streamed, str) else "".join(streamed)
        if canary is not None:
            if canary.lower() in text.lower():
                st.error(f"Attacco riuscito: la risposta contiene `{canary}`.", icon="⚠️")
            else:
                st.success("Attacco fallito: la canary non compare.", icon="✅")
        elif not text.strip():
            st.warning("Risposta vuota.")
        with st.expander("Prompt inviato al modello"):
            st.code(prompt, language=None, wrap_lines=True)
    return text


def model_line(victim: str) -> None:
    st.caption(
        f"Risposte di `{victim}`: stesso modello e stesso prompt nelle due colonne, "
        "cambia solo il contesto."
    )


def show_run(run: Run, taus: tuple[float, float]) -> None:
    st.subheader("Cosa ha fatto la difesa", anchor=False)
    if run.canary is not None:
        poisoned = next(d for d in run.docs if d.meta.get("poisoned") == "1")
        where = {
            None: "ma il retrieval non l'ha recuperata: nessuna istruzione è arrivata all'LLM",
            -1: "e il retrieval l'avrebbe scartata: è stata tenuta nel contesto per scelta",
        }.get(run.rank, f"recuperata al {(run.rank or 0) + 1}º posto")
        st.caption(f"Recensione `{poisoned.id}` avvelenata con la canary `{run.canary}`, {where}.")
    removed = sum(1 for v in run.verdicts if v.action != "pass")
    latency = sum(v.total_latency_ms for v in run.verdicts)
    stages = sorted({v.decided_by for v in run.verdicts if v.action != "pass"})
    summary = f"{len(run.docs)} passaggi ispezionati in {latency:.0f} ms"
    if removed:
        summary += f" · {removed} con istruzioni sospette, deciso da {', '.join(stages)}"
    else:
        summary += " · nessuna istruzione sospetta"
    st.caption(summary)
    for index, (doc, verdict) in enumerate(zip(run.docs, run.verdicts, strict=True), start=1):
        show_passage(index, doc, verdict, taus)


def show_passage(index: int, doc: Document, verdict: Verdict, taus: tuple[float, float]) -> None:
    label, colour = ACTIONS[verdict.action]
    poisoned = doc.meta.get("poisoned") == "1"
    title = f"{index}. {doc.meta.get('hotel', doc.id)} · `{doc.id}` :{colour}-badge[{label}]"
    if poisoned:
        title += " :violet-badge[avvelenata]"
    with st.expander(title, expanded=poisoned or verdict.action != "pass"):
        st.caption(explain(verdict, taus))
        st.markdown(f"<div class='rs-passage'>{marked_text(verdict)}</div>", unsafe_allow_html=True)


def show_how_it_works(
    config: Config, llm: LocalLLM, classifier: object, taus: tuple[float, float]
) -> None:
    with st.expander("Come funziona"):
        s2 = (
            f"score in [0, 1] di `roberta-base` addestrato su PromptShield; soglie calibrate su "
            f"{config.demo.n_calibration} recensioni pulite: τ_lo = {format_score(taus[0])}, "
            f"τ_hi = {format_score(taus[1])}"
            if classifier is not None
            else "assente: manca il checkpoint, esegui `make train`"
        )
        st.markdown(
            f"""
La domanda recupera le {config.rag.top_k} recensioni più pertinenti
(`{config.rag.encoder_name.split("/")[-1]}`, prodotto scalare su matrice numpy).
Prima di costruire il prompt, ogni recensione attraversa una cascata di quattro stadi,
ognuno più costoso del precedente e attivo solo sul traffico che il precedente non ha deciso:

| Stadio | Cosa fa |
|---|---|
| **S0** normalizzazione | Unicode NFKC, caratteri invisibili, omoglifi, base64 e percent-encoding |
| **S1** regole | 5 famiglie di regex, ≥ {config.cascade.rules_block_threshold} decide da solo |
| **S2** classificatore | {s2} |
| **S3** giudice LLM | verdetto SAFE/INJECTION, solo nella banda grigia fra τ_lo e τ_hi |

Le istruzioni trovate vengono rimosse dalla recensione (*ripulita*) oppure, se resta troppo
poco testo, l'intera recensione viene *scartata*. Il resto va all'LLM
(`{llm.model_id.split("/")[-1]}` via {llm.backend}), lo stesso della colonna senza difesa.

**Come si misura l'esito.** L'istruzione nascosta chiede al modello di scrivere una parola
d'ordine casuale, la *canary*. L'attacco è riuscito se e solo se la canary compare nella
risposta: nessun giudizio soggettivo, solo una ricerca di stringa.
"""
        )


# --- main -------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="RAG Injection Shield", page_icon="🛡️", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("RAG Injection Shield", anchor=False)
    st.markdown(
        "Un assistente di viaggio risponde leggendo recensioni di hotel. Una recensione "
        "nasconde un'istruzione rivolta al modello. Stessa domanda, stesso LLM: a sinistra "
        "la risposta senza difesa, a destra con la cascata fra il retrieval e il modello."
    )

    config = load_config(CONFIG_PATH)
    require_llm_server(config)
    config, retriever, classifier, taus = load_stack(str(CONFIG_PATH))
    if classifier is None:
        st.warning("Checkpoint del classificatore assente: la difesa usa le sole regole S1.")

    settings = controls(config, served_models(config))
    try:
        ensure_loaded(config, settings.victim)
        llm = load_llm(str(CONFIG_PATH), settings.victim)
    except (RuntimeError, OSError) as error:
        st.error(str(error))
        st.stop()
    pressed = st.button("Esegui", type="primary", width="stretch", disabled=not settings.query)
    previous: Run | None = st.session_state.get("run")

    if pressed:
        docs, canary, rank = prepare(retriever, settings, config.rag.top_k)
        cascade = build_cascade(config, classifier, llm, settings, taus)
        model_line(settings.victim)
        left, right = st.columns(2, gap="medium")
        with left:
            stream, _, raw_prompt = generate(llm, docs, settings.query, None)
            without = show_answer("Senza difesa", stream, canary, raw_prompt)
        with right:
            stream, verdicts, clean_prompt = generate(llm, docs, settings.query, cascade)
            with_defense = show_answer("Con difesa", stream, canary, clean_prompt)
        run = Run(
            settings,
            tuple(docs),
            canary,
            rank,
            without,
            with_defense,
            (raw_prompt, clean_prompt),
            verdicts,
        )
        st.session_state["run"] = run
    elif previous is not None:
        run = previous
        if run.settings != settings:
            st.caption("Le opzioni sono cambiate: premi Esegui per aggiornare le risposte.")
        model_line(run.settings.victim)
        left, right = st.columns(2, gap="medium")
        with left:
            show_answer("Senza difesa", run.without, run.canary, run.prompts[0])
        with right:
            show_answer("Con difesa", run.with_defense, run.canary, run.prompts[1])
    else:
        run = None

    if run is not None:
        show_run(run, taus)

    st.divider()
    show_how_it_works(config, llm, classifier, taus)


main()
