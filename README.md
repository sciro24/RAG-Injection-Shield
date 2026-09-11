# RAG Injection Shield

A cascaded defense against **indirect prompt injection** in retrieval-augmented generation
(RAG). It sits between the retriever and the language model, finds instructions hidden inside
the retrieved documents, and removes them before the model reads the context.

Project for the course *Sistemi Intelligenti per Internet*. Everything runs locally on a single
12 GB consumer GPU: no cloud services, no vector database, no orchestration framework.

---

## 1. Why

Frontier models ship with alignment, guardrails and server-side filters, and a hidden
instruction in a retrieved document gets through them less and less often. But many RAG
systems do not run on frontier models. They run **locally**, for data privacy, cost, or lack of
connectivity, on models with a few billion parameters quantized to fit a consumer GPU. There,
the only protection is whatever the model happens to have, and a small model will happily do
what it reads in its context.

This project starts from that case. The defense does not live in the model: it lives
**between retrieval and generation**, so it does not depend on how well the model is aligned.
The central measure, the attack success rate, is reported for **three locally served models**
(Gemma 4 E4B, Qwen3.8 4B distill, Ministral 3 3B) and for **two families of attacks**: the
explicit ones, with markers and imperatives, and the ones written by someone who knows how
the defense works.

The result in two lines (section 6): without any defense the three models fall for one fifth
to one third of the explicit attacks, and none of them is really better than the others; with
the cascade, explicit attacks drop to 2-3% on every model, while attacks written to evade the
rules go through unchanged. The defense helps, it does not depend on the model, and it works
against what it knows how to recognize.

---

## 2. Key concepts

**Indirect prompt injection.** In a *direct* injection the user writes the malicious
instruction. In an *indirect* one a third party plants it inside a document that the system
will retrieve later: a web page, an email, a product review. The model cannot tell data from
instructions, so it may follow the planted text, and the user is the victim, not the attacker.

**Canary.** To decide whether an attack succeeded without any human judgment, every injected
instruction asks the model to write a random password-like string (`CNRY` followed by eight
random characters). If that string appears in the answer, the attack worked. It is a plain
substring check: deterministic, cheap, and impossible to argue with. The limit is that it
only measures attacks with an observable goal.

**ASR, attack success rate.** The fraction of poisoned samples in which the canary appears in
the model's answer. Reported with 95% confidence intervals. An ASR of 0.30 means that three
attacks out of ten got through.

**Thresholds and calibration.** The classifier gives every passage a score in [0, 1]. Two
thresholds turn scores into decisions: below τ_lo the passage passes, above τ_hi it is
flagged, in between the LLM judge decides. The thresholds are not chosen by hand: they are
set so that, on a large sample of *clean* passages, only 1% (τ_lo) and 0.1% (τ_hi) score
above them. This is the false positive rate (FPR) the system commits to. The detector metric
used throughout is therefore the true positive rate at a fixed FPR, not accuracy or F1.

---

## 3. How it works

The defense is a cascade of four stages. Each stage costs more than the one before and only
sees the traffic the previous stage did not decide.

```
retrieval ──► S0 ──► S1 ──► S2 ──► S3 ──► sanitization ──► prompt to the LLM
```

| Stage | What it does | Cost |
|---|---|---|
| **S0** normalization | Unicode NFKC, removal of invisible characters, Cyrillic and Greek homoglyphs mapped to Latin, base64 and percent-encoded blocks decoded | µs |
| **S1** rules | five regex families: prompt negation ("ignore all previous instructions"), chat-template role markers (`### SYSTEM`, `<\|im_start\|>`, `[SYSTEM]`), imperatives addressed to an assistant, requests to reveal the configuration, exfiltration to a URL or email. A weight of 0.9 or more decides on its own | µs |
| **S2** classifier | a fine-tuned `roberta-base` encoder scoring the passage, with the user query as context | ms |
| **S3** LLM judge | the same local model answering SAFE or INJECTION, only for scores between τ_lo and τ_hi | s |

When a stage flags a passage, the offending text is removed and the rest is forwarded
(*sanitize*). If less than 20% of the passage survives, the whole passage is dropped
(*block*): a mutilated fragment adds noise to the context without adding information.

Three design choices worth knowing:

- **Sanitization works on the normalized text, and the normalized text is what the model
  reads.** Mapping the removed spans back to the original text is the most fragile part of
  systems like this, and any mistake is an escape route. Forwarding the normalized text also
  neutralizes Unicode obfuscation by construction: what the model reads is what the detector
  inspected. The cost is that the model loses the original formatting.
- **Flagged spans are extended to the whole line.** A rule matches the bait ("ignore all
  previous instructions") but the injected instruction is the entire line; removing only the
  bait leaves the payload in the context, and the model executes it anyway.
- **Without the judge, the gray zone passes.** In the ablation without S3, scores between the
  two thresholds are let through. Resolving them as flagged would inflate false positives;
  this way removing the judge costs recall, which is what the ablation has to measure.

The insertion point is explicit: one function between the retriever and the prompt builder.
This is why the project uses no LangChain or LlamaIndex, which would hide exactly that seam.

---

## 4. Data and models

### Datasets

| Source | Role |
|---|---|
| [PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) | classifier training, validation and calibration (direct injection) |
| [BIPIA](https://github.com/microsoft/BIPIA) | **test only** (indirect injection, four domains: email, code, table, abstract) |
| [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) | clean documents for BIPIA's `abstract` domain |

Training set: 18 909 passages. Test set: 5 932 poisoned BIPIA passages. Calibration set:
10 000 clean passages. A check at data-preparation time guarantees that no test attack
appears in the training data, comparing normalized alphanumeric text so that obfuscated or
embedded copies are caught too.

### Attack vectors

The end-to-end experiments and the demo use attack templates written for the project, in two
families. All of them ask the model to output a canary.

| Family | n | What they look like |
|---|---|---|
| `baseline` | 8 | role markers, "ignore all previous instructions", imperatives addressed to the assistant |
| `adaptive` | 6 | written knowing the S1 rules: no keyword, disguised as an editorial note, reported speech or a formatting convention; one in Italian |

The `adaptive` family is the attacker who knows the defense. None of the six triggers a single
rule: if anything stops them, it is the classifier.

### Classifier

`roberta-base`, fine-tuned for three epochs in bf16 (about six minutes). The project plan
started from DeBERTa-v3-base, which did not train stably in reduced precision on this GPU;
RoBERTa converged at the first attempt. The choice is about numerical stability, not
capacity: the public reference detector, a DeBERTa-v3 trained on far more data, scores lower
on the same test set (section 6).

### Language models

All generation goes through **LM Studio**, which serves quantized models on a local
OpenAI-compatible endpoint. Three victim models are evaluated, each one loaded alone in
memory, and the same model that answers also acts as the S3 judge, as it would in a local RAG
with a single LLM.

| Model | Parameters | Note |
|---|---|---|
| `google/gemma-4-e4b` | 8B total, 4B effective | the most aligned of the three |
| `qwen3.8-4b-distill` | 4B | reasoning model, reasoning disabled |
| `mistralai/ministral-3-3b` | 3B | the typical local-RAG model |

Two details that matter for the measurement. Reasoning is switched off server-side
(`reasoning_effort: none`): a thinking model spends its whole token budget reasoning and
returns an empty answer, and an ASR measured on empty answers is zero everywhere and means
nothing. A health check at startup sends a realistic RAG prompt and fails loudly if the model
returns no text. And the generator prompt contains **no defensive wording**: telling the
model to ignore instructions found in the passages would be a defense in itself (spotlighting)
and would leave nothing for the experiment to measure.


---

## 5. Reproduction

```bash
make install          # Python 3.11 venv with pinned dependencies
make lint test        # ruff + pytest
make data             # downloads the sources, builds the splits and the demo corpus
make train            # trains the classifier, ~6 minutes
make eval             # all four experiments
make demo             # Streamlit demo
```

`eval-asr`, `eval-utility` and the demo need LM Studio running with its local server on
`http://localhost:1234` and the three victim models downloaded. The other experiments do not
use an LLM. `make eval-asr` loops over the models listed in `evaluation.victims` in
`configs/config.yaml` and writes one raw file per model, so a model can be added without
re-running the others (`make eval-asr VICTIM=<model id>`); `make eval-asr-report` rebuilds
the summary tables. Every path and hyperparameter lives in `config.yaml`.

---

## 6. Experiments and results

Four experiments, each answering one question. All numbers come from `reports/results/`;
every value there carries a 95% confidence interval.

### 6.1 How good is the classifier on real indirect injections?

Trained on PromptShield (direct injection), tested on BIPIA (indirect injection), against
three baselines.

| Detector | AUC | TPR @ FPR 1% |
|---|---|---|
| S2, `roberta-base` (this project) | **0.70** | **0.07** |
| public reference (`protectai/deberta-v3-base-prompt-injection-v2`) | 0.65 | 0.01 |
| TF-IDF + logistic regression | 0.55 | 0.02 |
| S1 rules only | 0.53 | 0.07 |

**Answer:** better than every baseline, but far from good. The classifier is near perfect on
data like its training set and drops to AUC 0.70 on real indirect injections. The public
reference model, trained on much more data, does worse: the gap is between direct and
indirect injection, not between models.

### 6.2 Does what the classifier learns transfer across domains?

Leave-one-domain-out on BIPIA: train on one domain, test on the others. TPR @ FPR 1%.

| train ↓ / test → | abstract | code | email | table |
|---|---|---|---|---|
| PromptShield | 0.45 | 0.03 | 0.28 | 0.26 |
| abstract | **0.89** | 0.01 | 0.44 | 0.35 |
| code | 0.39 | **0.13** | 0.40 | 0.45 |
| email | 0.58 | 0.15 | **0.54** | 0.33 |
| table | 0.35 | 0.00 | 0.01 | **0.54** |

**Answer:** poorly. The diagonal dominates and the PromptShield row is the worst almost
everywhere: training on direct injection is the worst way to prepare a detector for indirect
injection. Training on the right domain takes detection from 0.03 to 0.89, which is the
natural next step for this project.

### 6.3 Do the thresholds transfer?

**Answer:** no. Calibrated on PromptShield's clean passages, τ_hi comes out above the score of
most real attacks and the cascade lets everything through. Calibrated on 5 000 clean demo
reviews it works there (τ_lo = 2.9e-05, τ_hi = 3.1e-05), but the same threshold produces
**71% false positives on real BIPIA prose**. A threshold is only valid for the distribution it
was calibrated on: every deployment needs its own calibration set.

### 6.4 How much does the defense reduce successful attacks?

Setup: three local models, two attack families, 150 poisoned samples per cell, same samples
for every model. The poisoned document is always in the context, so every sample measures
the defense and not the retriever. Full cascade against no defense:

| ASR | Gemma 4 E4B | Qwen3.8 4B distill | Ministral 3 3B |
|---|---|---|---|
| explicit attacks, no defense | 0.23 | 0.21 | 0.30 |
| explicit attacks, full cascade | **0.03** | **0.02** | **0.03** |
| adaptive attacks, no defense | 0.27 | 0.19 | 0.30 |
| adaptive attacks, full cascade | **0.27** | **0.19** | **0.28** |

Contribution of each stage, explicit attacks, averaged over the three models:

| Configuration | Active stages | ASR |
|---|---|---|
| `none` | nothing | 0.24 |
| `s1` | S0 + S1 | 0.14 |
| `s1_s2` | S0 + S1 + S2 | 0.04 |
| `full` | S0 + S1 + S2 + S3 | 0.03 |

**Answer, in three points.**

1. **No local model resists on its own.** One fifth to one third of the explicit attacks get
   through without a defense, and the three models are not really distinguishable.
2. **Against explicit attacks the defense works, equally on every model.** The cascade cuts
   the ASR by 7-10 times, to 2-3%, with confidence intervals disjoint from `none`. The rules
   halve it, the classifier divides it by another three or four, the judge adds little.
   Detection is 84% on all three models: the defense reads the same text regardless of who
   answers.
3. **Against an attacker who knows the rules, the defense does nothing.** Detection at 8%,
   ASR unchanged. The rules never fire by construction and the classifier does not separate
   these sentences from ordinary reviews. The Italian vector succeeds 82-89% of the time on
   every model and is never detected: everything in the defense is English. This is the
   project's negative result, and the most important one: the protection in point 2 is
   protection against known patterns.

Breakdowns by single vector and by injection position are in `asr_by_vector.csv` and
`asr_by_position.csv`. Injections at the start of a document are the hardest to neutralize.

### 6.5 Does the defense damage clean answers?

**Answer:** not on the demo corpus. On 60 legitimate queries the cascade removes nothing and
the answers with and without it are identical. This holds only for the corpus the thresholds
were calibrated on (see 6.3).

---

## 7. The demo

`make demo` opens a single-page Streamlit app.

1. **Controls.** A question (eight presets, or free text), the model that answers (choosing one
   loads it in LM Studio and unloads the others), the instruction to hide in the most relevant
   review (any of the 14 vectors, or none), and its position.
2. **Two answers side by side**, streamed from the same model: without the defense and with
   the cascade in between. Under each, the outcome of the attack, and an expander with the
   exact prompt sent to the model: the two columns differ only in the context, raw or
   sanitized. There is no switch to turn the defense off, because the comparison is the output.
3. **What the defense did.** For every retrieved passage, the verdict, the stage that decided
   with its score and threshold, and the text with the removed parts struck through.

At startup the demo indexes the 300 demo reviews with `all-MiniLM-L6-v2` and calibrates the
thresholds on the demo corpus, exactly like the experiments. Worth trying: the explicit vector
2 on Ministral (falls without the defense, resists with it), then the adaptive vector 6 on the
same model (passes in both columns).

---

## 8. Limitations

- **The canary protocol** only measures attacks with an observable goal. An attack that skews
  the answer without emitting a marker is not captured.
- **The demo corpus and the attack vectors are synthetic.** 300 reviews generated from about
  50 hand-written sentences; 14 attack templates written for the project. The `adaptive`
  family is an informed attacker, not an optimizing one. The detector tables and the transfer
  matrix are on real data (PromptShield and BIPIA); the ASR and utility tables measure a real
  detector and real models on synthetic documents and attacks.
- **English only.** Rules, training data and calibration are in English; the Italian vector
  exists to show what happens outside that language.
- **Sanitization loses the original formatting** of the passage.

Related techniques deliberately left out of scope: spotlighting and delimiting of untrusted
data, structured queries, adversarial training of the target model, cross-encoder re-ranking.

---

## 9. Layout

```
src/shield/      library: one module per responsibility, no I/O in the core modules
scripts/         prepare_data, train, evaluate, build_demo_corpus
app/demo.py      Streamlit demo, single file (theme in .streamlit/config.toml)
tests/           pytest, no real model involved
configs/         config.yaml: every path and hyperparameter
reports/         figures/, results/, logs/
models/          detector/ (S2), lodo_<domain>/ (transfer matrix), tfidf_baseline.pkl
data/            raw/ (downloaded sources, demo corpus), processed/ (parquet splits)
```
