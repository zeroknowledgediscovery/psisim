# LSM Veritas: independent replication on public GSS data

This folder tests the claims of the *LSM Veritas* README (structural-coherence
scoring with native LSM persistence) using only what is publicly reachable:
the public GSS files (via `kjhealy/gssr`), the public DTAG native models
(`gss/gss_2016`, `gss/gss_2018`, release v0.2.0), and the native
`predict_distribution` runtime.

**Bottom line.** Every empirical claim in the README replicates
qualitatively. Two refinements are needed:

1. LSM-generated replacements are partly hard to detect because the scorer
   is the generator (*self-recognition*).
2. The native LSM is the strongest detector tested, but a one-partner pairwise
   model comes close.

The README's absolute AUCs are higher than ours. Most of the gap is
explained by our having fewer scored items per respondent.

---

## 1. What could and could not be reproduced

The README's canonical protocol retrains a native LSM on 80% of the MAGICS
`gss_2018.csv`. Two things block that here:

- The native **trainer** lives in the private `zeroknowledgediscovery/lsm`
  repository. The runtime modules available here can score and sample, but
  not train.
- The MAGICS file is not public. Its 1,784 rows are a **non-random** subset
  of the 2,348 public 2018 respondents: 55% married against 42.5% in the
  public file, and the rows are not in GSS id order.

So the claims were tested in three complementary setups:

| Setup | Model | Scored respondents | Held out? | Items per respondent |
|---|---|---|---|---:|
| **S**: surrogate holdout | surrogate LSM (below), trained on 80% of public 2018 | other 20% (470), 5 split seeds | yes, strict 80/20 | ~528 of 985 |
| **N16**: native cross-wave | native DTAG `gss_2016` | 1,000 public 2018 respondents | yes (different wave) | ~238 of 913 |
| **N18**: native same-wave | native DTAG `gss_2018` | 1,000 public 2018 respondents | no (unknown overlap) | ~401 of 1,034 |

**Surrogate LSM** (`surrogate_lsm.py`). It has the native layout:

- one categorical conditional tree φᵢ(x₋ᵢ) per item, trained only on
  training rows;
- missing inputs handled natively;
- leaves smoothed toward the item's training marginal;
- the target item can never be its own input.

Its hyperparameters (`min_samples_leaf=80, max_depth=8, alpha=10`) were
chosen on a validation slice of the *training* set only. On held-out rows it
reaches 0.739–0.757 nats/answer, against 1.054–1.065 for marginal-only.

**Native label mapping** (`data_prep.py`). Public value labels were matched
to the model's category strings by string similarity plus marginal
agreement with the model's empty-state Ψ₀. Each item was accepted only if:

- ≥97% of its observed values map;
- the total variation distance from Ψ₀ is ≤0.05 (2018) or ≤0.10 (2016).

That accepted 744 and 389 items respectively. Items binned into letters
inside the model (175 and 110) were left missing, because their bin edges
are unknown. A per-item audit is written next to the data.

The paired design (each corrupted copy keeps its source row's missingness
mask) is identical to the README's. AUC is computed from U = −ln 2 · P(x)
with corrupted rows as the positive class.

---

## 2. Claim-by-claim verdict

### 2.1 Veritas detects marginal-preserving permutation (README §12): **replicates**

| Requested | S realized / AUC | N16 realized / AUC | N18 realized / AUC | README realized / AUC |
|---:|---:|---:|---:|---:|
| 0.5% | 0.14% / 0.511 | 0.23% / 0.516 | 0.23% / 0.525 | 0.13% / 0.529 |
| 1% | 0.54% / 0.532 | 0.55% / 0.541 | 0.58% / 0.561 | 0.55% / 0.585 |
| 2% | 1.20% / 0.567 | 1.11% / 0.579 | 1.18% / 0.616 | 1.19% / 0.658 |
| 5% | 3.14% / 0.660 | 2.74% / 0.674 | 2.90% / 0.734 | 3.17% / 0.837 |
| 10% | 6.19% / 0.776 | 5.31% / 0.782 | 5.69% / 0.849 | 6.25% / 0.954 |
| 20% | 12.16% / 0.908 | 10.43% / 0.902 | 11.10% / 0.939 | 12.06% / 0.997 |

- The dose-response is monotone in all three setups, and every row has
  essentially all ΔU > 0 at ≥5%.
- AUC is lower than in the README. AUC rises with the *number* of changed
  answers per respondent, and our respondents carry 238–528 scored items
  against the README's ~1,000. N18 (401 items) sits between N16 (238) and the
  README, as that explanation predicts.
- Split-to-split SD of AUC (S, 5 seeds) is ≤0.006. One split is not driving
  the result.

### 2.2 The marginal-only control stays at chance (§3, §19): **replicates**

- Permutation: AUC(U_marg) = 0.500–0.504 at every level, in every setup
  (README 0.500–0.506).
- Conditional and sequential replacement: also 0.500–0.504.
- It leaves chance only for the positive control, uniform replacement, where
  answers become individually rare: 0.75–0.81 at 20%.

### 2.3 LSM-conditional replacement is much harder to detect (§13–14): **replicates**

| ~8% of answers changed | S | N16 | N18 | README (7.0%) |
|---|---:|---:|---:|---:|
| conditional (independent-masked) AUC | 0.617 | 0.615 | 0.758 | 0.691 |
| permutation AUC interpolated to the same realized rate | 0.829 | 0.848 | 0.897 | 0.960 |
| ΔU per changed answer: conditional / permutation (nats) | 0.64 / 2.3 | 1.0 / 4.5 | 1.9 / 4.7 | — |

- The gap holds at every realized rate.
- At ≤2% changed, conditional AUC is 0.500–0.526 everywhere, against the
  README's 0.498–0.504.

### 2.4 Sequential completion is a stronger adversary (§4.3, §24.1): **confirmed**

At ~8% changed, sequential completion is harder to detect than independent
draws in every setup:

| Setup | sequential AUC | independent AUC |
|---|---:|---:|
| S | 0.572 | 0.617 |
| N16 | 0.549 | 0.615 |
| N18 | 0.707 | 0.758 |

This is the README's §5 prediction: independent draws become mutually
incompatible as the selected set grows.

### 2.5 "Changes that violate the learned structure score higher than changes that follow it" (§15): **supported, with a qualification**

In the README design the generator and the scorer are the *same* model. A
low AUC could therefore reflect a model assigning high probability to its
own samples, not structural coherence. Two tests separate these.

**Cross-model test** (`run_crossmodel.py`, setup S, 3 splits × 3 repeats).
Two surrogates, A and B, are trained on disjoint halves of the training set
(939 respondents each, equal held-out fit 0.80 nats/answer). All rows are
scored by B.

| ~9% changed | AUC (scored by B) | ΔU per changed answer |
|---|---:|---:|
| B's own conditional draws (README design) | 0.596 | 0.51 |
| A's conditional draws (independent model, same population) | 0.639 | 0.76 |
| B's own sequential completion | 0.560 | 0.31 |
| A's sequential completion | 0.609 | 0.59 |
| permutation (12% changed) | 0.862 | 1.90 |

- Draws from an *independent* model of the same population cost
  0.50–0.76 nats per changed answer at every level. That is about a third of
  what permutation costs.
- So the README's interpretation holds: answers consistent with the
  population's conditional structure are much harder to detect than ones
  that break it.
- But the same model's own draws are cheaper still (0.18–0.51 nats).
  Self-recognition is real and secondary. The README's control therefore
  *overstates* how undetectable structure-respecting changes are.

**Cross-family generators** (`run_baselines.py`, `run_native_baselines.py`).
Replacements drawn from a logistic-regression model and from a latent-class
model were scored by every detector. Every model is close to blind to its
own samples:

- logistic regression scores its own draws at 0.42–0.50 (*below* chance);
- the latent-class model scores its own at 0.49–0.50;
- the native LSM scores its own at 0.50–0.61 (N16).

Generators with better models of the data are harder for *other* models to
detect:

- latent-class draws (a weak model) are caught by the native LSM almost as
  readily as permutation;
- logistic-regression and LSM draws are not.

Both effects should be stated together.

### 2.6 Do non-LSM methods replicate Veritas? Mostly not; a pairwise model comes close

Every detector was fitted on training data only and scored on *identical*
corrupted rows. For N16 the non-LSM detectors were trained on the public
GSS-2016 respondents (2,866 people), coded identically to the native
`gss_2016` model.

| N16: AUC at ~5% changed (10% requested) | native LSM | pairwise (best partner) | logistic-reg. PL | latent-class | PCA | kNN | iso. forest | marginal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| permutation | **0.779** | 0.736 | 0.667 | 0.542 | 0.586 | 0.568 | 0.503 | 0.502 |
| splice (nearest-respondent answers) | **0.692** | 0.642 | 0.571 | 0.500 | 0.526 | 0.501 | 0.494 | 0.490 |
| logistic-reg. conditional draws | **0.594** | 0.565 | 0.466 | 0.503 | 0.502 | 0.506 | 0.500 | 0.499 |
| latent-class conditional draws | **0.740** | 0.693 | 0.620 | 0.497 | 0.554 | 0.534 | 0.504 | 0.503 |
| native LSM conditional draws | 0.540 | 0.572 | **0.574** | 0.520 | 0.537 | 0.526 | 0.503 | 0.503 |
| uniform | **0.920** | 0.900 | 0.841 | 0.719 | 0.711 | 0.731 | 0.593 | 0.672 |

- **Generic anomaly detectors do not reproduce Veritas.** Isolation forest
  stays at chance. PCA reconstruction, kNN distance and a latent-class joint
  likelihood are weak (permutation AUC 0.54–0.59 where the native LSM gets
  0.78). Veritas is not "any outlier score".
- **The native LSM is the best detector for every corruption it did not
  generate**, at every level, ahead of the next best by 0.02–0.06 AUC at
  moderate levels. The only exception is a tie (0.986 vs 0.987) at 20%
  uniform replacement.
- This holds even though the pairwise and logistic-regression models were
  trained on more respondents (2,866 vs ~1,950) and fit held-out answers
  *better* per answer (0.71–0.72 vs 0.78 nats). Detection power is not the
  same thing as pseudo-likelihood fit.
- **A pairwise model gets most of the way.** It predicts each answer from its
  single most informative partner question. With the tree surrogate
  (setup S) it matches the LSM within 0.01 AUC. With the native model it
  trails by 0.03–0.06.
- So much of the Veritas signal lives in strong pairwise links between
  questions. The native LSM's advantage over pairwise structure is real but
  modest.

Full tables: `findings/detector_tables.md`.

### 2.7 Other README statements

- **§11 exploratory in-sample result** (AUC ≈ 0.997–1.0 at 3–11% changed):
  the N18 setting is the closest analogue, and we get 0.734–0.939 at 2.9–11%.
  The difference is consistent with the 401 vs 1,034 items scored.
- **§7 OOV audit:** the held-out OOV fraction in setup S is
  0.018–0.028% (46–70 cells). The README reports 0.12% on its own split and
  coding.
- **§16–17 adversarial/private-model use:** consistent with the results.
  Evading Veritas requires generating from a good conditional model, and
  works best with access to the scoring model itself (self-recognition).
  That strengthens the private-model argument.
- **§18 "not a lie detector":** not tested; it is a framing statement.

---

## 3. Suggested changes to the Veritas README

1. Report the *number of scored items per respondent* next to every AUC.
   Detection power scales with it.
2. Add an independent-generator control next to the same-model conditional
   control (a cross-model or cross-family test). It separates structural
   coherence from self-recognition.
3. Add non-LSM detectors (at least a best-partner pairwise model) to show
   what the LSM adds beyond pairwise structure.
4. In §15, soften "changes generated in accordance with the learned structure
   are hard to detect". Independent structure-respecting draws are detected
   at roughly ⅓ of permutation's per-answer cost, and the scoring model's own
   draws at less.

---

## 4. Not tested

- the exact canonical protocol: native trainer and MAGICS data unavailable;
- other survey families (Afrobarometer, Eurobarometer, WVS);
- §24.4 partial-response calibration. It is feasible with the current code
  if wanted.

Hot-deck ("splice") replacement uses answers copied from the most similar
training respondent. It is **not** a structure-preserving control. Even the
closest donors share only ~57% of answers with the respondent (a random
respondent shares 43.5%), so it behaves as a milder form of structure
breaking.

---

## 5. Reproduce

```bash
git clone --depth 1 https://github.com/kjhealy/gssr        # public GSS cumulative file
python3 extract_gssr.py gssr/data/gss_all.rda WORK         # year-2018 rows + value labels
                                                           # (2016 rows: same, year==2016)
python3 applications/psisimulation/fetch_model.py gss/gss_2018   # and gss/gss_2016
export PYTHONPATH=<dir with predict_distribution*.so> LD_LIBRARY_PATH=<libomp dir>
python3 data_prep.py --work WORK
python3 run_experiments.py surrogate --raw WORK/surrogate_2018_raw.pkl --out results/surrogate_gss2018_holdout --split-seeds 20261002,1,2,3,4
python3 run_experiments.py native --model-dir ~/.cache/dtag/models/gss/gss_2016 --rows WORK/native_gss_gss_2016.pkl --out results/native_gss2016_on_2018 --n-rows 1000 --repeats 5 --seq-repeats 3
python3 run_experiments.py native --model-dir ~/.cache/dtag/models/gss/gss_2018 --rows WORK/native_gss_gss_2018.pkl --out results/native_gss2018_on_2018 --n-rows 1000
python3 run_crossmodel.py --raw WORK/surrogate_2018_raw.pkl --out results/crossmodel_gss2018
python3 run_baselines.py  --raw WORK/surrogate_2018_raw.pkl --out results/baselines_gss2018_holdout
python3 run_native_baselines.py --model-dir ~/.cache/dtag/models/gss/gss_2016 --audit WORK/native_gss_gss_2016_audit.json \
    --train-pkl WORK/gss2016_all.pkl --labels WORK/gss_value_labels.pkl --eval-rows WORK/native_gss_gss_2016.pkl \
    --out results/native_baselines_gss2016_on_2018
python3 make_tables.py results findings
```

Seeds: the README split seed is 20261002, plus split seeds 1–4. Corruption
RNGs are seeded from (split seed, repeat, level, mechanism).

Two runs were interrupted by job time limits and resumed into `_partN`
folders:

- native `gss_2018`: its sequential completion exists only at 5/10/20%;
- the surrogate detector comparison's split 2 was re-run whole.

`make_tables.py` merges the parts and keeps only complete splits.

Per-level summaries are in `findings/`; per-repeat CSVs and run metadata are in `findings/raw/`.
