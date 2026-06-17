# The Frequency ML — Scientific Dossier

**Author:** Gabriel Nascimento · **Status:** canonical record · **Last updated:** 2026-06-17
**Canonical manuscript:** [`PAPER_DRAFT_v6_crosssystem.md`](PAPER_DRAFT_v6_crosssystem.md)
**AI disclosure:** see [`AUTHOR.md`](../../AUTHOR.md) — AI was a disclosed tool; the science, decisions, and authorship are the author's.

## Purpose & how to read this

This is the single authoritative record of what the project investigated, how its focus
shifted, and **what is proven — positively and negatively** — with every claim tied to a
committed, reproducible output. It exists because the work passed through six manuscript
drafts and a dozen working logs; rather than ask a reader to reconstruct the arc from that
sprawl, this dossier states it once and **supersedes** those scattered files (see Inventory).

It is deliberately adversarial about its own claims. Where a finding was withdrawn or
downgraded, that is recorded as the audit functioning as designed, not as an afterthought.
Every numeric claim below cites a script and a committed `outputs/json/*.json`; all are
regenerable from public NHANES data via the pinned environment (`requirements-lock.txt`,
`tests/README.md`).

---

## 1. What stands (the load-bearing results)

Two findings are guarded to a standard well above typical applied-clustering work.

### 1.1 Auditory inter-ear asymmetry has a real far tail
The standardized inter-ear contrast `z = |mean(R) − mean(L)| / SD` carries a far-tail excess
far beyond what the data's own marginals and inter-side correlation produce:

| Level | Real | Null mean | p | Survives |
|---|---|---|---|---|
| \|z\|>4 | 92 | 2.5 | 5×10⁻⁴ | Gaussian, heteroscedastic (\|z\|≥3), **and** t-copula nulls |
| \|z\|>5 | 52 | 0.1 | 5×10⁻⁴ | all three |

Sources: `audit_01_mc_envelope.json` (B=2,000 Monte-Carlo envelope), `audit_04_heteroscedastic_null.json`,
`audit_05_t_copula_null.json`. The tail is **bilaterally symmetric** — of 69 cases with a >50 dB
inter-ear gap, 38 right-worse / 31 left-worse, binomial p=0.47 (`audit_03_side_decomposition.json`) —
it lives in the difference subspace (recovery r=0.92) and vanishes under binaural averaging
(`28_ipsative_check.json`), and it replicates in an independent cohort (`audit_07_ohhr_replication.json`).
**Nine independent guards; no material gap.**

### 1.2 Hearing shape is a continuum, not discrete subtypes
No clustering criterion supports stable discrete subtypes on the row-centered shape space:

- Best KMeans silhouette 0.281 @k=2, ≤0.18 for k≥3 (`26_method_comparison.json`).
- PC1 unimodal: Hartigan dip p≈1.0; OPTICS = 1 cluster; 5-family tendency battery converges on
  "no subtypes" (`31_cluster_tendency.json`).
- BIC interior minimum is shallow (1.5% depth) and unstable k=4↔5 — calibrated against a
  curved-manifold generator as a known false-positive mode (`spinoffs/frente2-bic-simulation/`).
- Non-circular minority diagnostics: LOPO recall 0.75 (25% boundary leak ⇒ not discrete),
  LOBO ARI 0.27 (fragile partition), dual-encoding ARI 0.99 (robust shape)
  (`spinoffs/frente1-methods-note/METHODS_NOTE.md`).

**New in this dossier — two additional guards lifting this claim to the auditory standard:**
- **Method breadth** (`26b_method_breadth.json`): extending beyond KMeans/GMM/HDBSCAN to DBSCAN,
  Spectral, and Agglomerative (complete + average) linkage, **no balanced split crosses the
  declared "substantial separation" bar (silhouette ≥ 0.5 with a minority cluster ≥ 5% of N).**
  Average linkage reaches silhouette 0.67 @k=2 — but that split is `[3999, 1]` (min_frac 0.0003):
  textbook outlier-peeling, a single point shaved off, **reported and disqualified, not hidden.**
  Verdict: `continuum_robust_to_method_choice`.
- **Row-centering ablation** (`35_row_centering_ablation.json`): the conclusion is **not** an
  artifact of the centering choice. With centering, silhouette 0.281 / dip p=1.0; without it,
  silhouette rises to 0.424 (level adds a severity gradient) but PC1 stays unimodal (dip p=1.0) —
  still a continuum, now of severity. The asymmetry tail is computed on raw thresholds and is
  centering-invariant by construction (|z|>4 = 91, reproducing §1.1).

---

## 2. Claims × evidence (positive *and* negative)

| # | Claim | Verdict | Evidence (committed) |
|---|---|---|---|
| A | Hearing shape is a continuum, not discrete subtypes | ✅ Supported | `26`, `26b`, `30`, `31`, `35`, frente1/frente2 |
| B | Auditory inter-ear far-tail asymmetry is real, not artifact | ✅ Supported | `audit_01`/`04`/`05`/`03`/`07`, `28`, `29` |
| C | Result is robust to method / hyperparameter / seed | ✅ Supported | `tests/test_pipeline_contract.py`, `audit_06`, `26b`, frente3 freeze |
| D | Cross-system *gradient* (auditory ≫ grip > vision) | ⚠️ Methods demo, not discovery | `cmp_dimensionless_asymmetry.json`, `audit_02` |
| — | Grip carries a genuine excess | ❌ **Withdrawn** → "Gaussian-null excess only" (fails t-copula) | `audit_05` |
| — | Vision shows asymmetry | ➖ **Negative control** (within/below null in general pop.) | `vis_03`, `vis_04` |
| — | Lateralized-trauma etiology | ❌ **Withdrawn** (tail is bilaterally symmetric) | `audit_03` |
| — | Cluster-1 is a validated phenotype | ⬇️ **Downgraded** to a continuum tail | `audit_06`, frente1 |
| — | PTA does not predict speech-in-noise (r≈0.015) | ❌ **Corrected** (ingestion bug; r=0.85) | `CORRECOES_2026-06-01.md` |
| — | Tinnitus ~ atypicality (38% vs 18%, cluster-based) | ♻️ **Reinstated, reframed** (see §5) | `34_tinnitus_reexamination.json` |

Negative results are first-class here: a discovery audit that never overturns its own claims
is not auditing anything.

---

## 3. Scrutiny matrix (claim → guards → residual gaps)

| Claim | Guards | Residual gap |
|---|---|---|
| **A** continuum | BIC-sim calibration, 5-family tendency battery, method comparison **+ method breadth (`26b`)**, null-vs-synthetic (`30`), LOPO/LOBO/dual-encoding, **row-centering ablation (`35`)** | None material |
| **B** asymmetry real | Gaussian / heteroscedastic / t-copula nulls (MC B=2,000), side-symmetry binomial, sum-vs-diff decomposition, per-case sanity (13/13), injection-recall, OHHR replication | None material |
| **C** method-robust | Contract test (golden values, parity 1e-13), hyperparameter sweep (`audit_06`), skfreeze round-trip, shared canonical loader, pinned lock | Minor: CI not yet wired (local gate) |
| **D** cross-system gradient | Per-system Gaussian null, grip t-copula (fails), **vision case-audit (`vis_04`)**, inclusion-policy robustness (`audit_02`), **triplet-overlap audit (`36`)** | **Known & stated:** 3 points; post-hoc; no within-person link (cohorts disjoint by calendar — `36` confirms 0 overlap). Framed as methods demonstration. |

**New guards added by this dossier:** `26b` (method breadth + multiple-testing accounting),
`35` (row-centering ablation), `vis_04` (vision extreme case-audit, 64/64 in physiological
range — parity with audio 13/13 and grip 15/15), `36` (cross-system SEQN/calendar overlap,
formalizing what was previously prose), `34` (tinnitus re-examination, §5). After these,
**no claim is held to a visibly weaker standard than the others without that being stated and bounded.**

---

## 4. Evolution of focus (and why each shift happened)

The project reframed four times in 21 days. Each shift is labeled by its driver — **evidence**
(a finding forced it) or **strategy** (framing/venue/audience) — because that distinction is
itself part of the honesty record.

1. **v1 "Phenotyping" (2026-05-26).** Claimed 2 discrete HDBSCAN clusters incl. a "clinically
   meaningful" unilateral phenotype. *Baseline; in hindsight, overclaimed.*
2. **v4 downgrade (→2026-06-01).** Cluster-1 → "exploratory signal, not validated phenotype."
   **Evidence-driven:** LOPO recall 0.75; AUC=1.0 shown circular; OHHR reframed exploratory.
3. **v5 "Reproducibility Audit" (2026-06-02).** Reframed from *discovery* to *do reported
   subtypes survive method/seed/spec?* — answer: no. **Strategy-driven (stated):** the continuum
   was already established (Allen & Eddins 2010; 2024–26 cohorts of 80–110k), so the project
   *concedes priority* and claims the method + scaffold instead (`MUDANCA_v5_AUDITORIA.md`).
   *The robustness battery that backs it is new evidence; the decision to lead with it was strategic.*
4. **v6 "Cross-System Audit" (2026-06-14).** Auditory as a worked example of a measure-agnostic,
   null-calibrated paired-organ audit, with vision as an internal negative control.
   **Mixed:** new nulls + two new systems (evidence); "sell as methods paper, not a trauma-gradient
   discovery" (strategy, `meta_analysis.md`). Withdrew lateralized-trauma (evidence: `audit_03`)
   and grip "genuine excess" (evidence: `audit_05`).

The throughline: **every step strengthened the method and weakened the need for a grand thesis.**
The accumulated asset is the apparatus (null calibration, negative control, anatomical
verification, extreme-case sanity), not a discovery about ears, hands, or eyes.

---

## 5. Honesty-log completion (claims dropped without a prior record)

The `CHANGELOG.md` honesty log covers the v5→v6 withdrawals well. Two claims present in v1–v4
disappeared between v4 and v5 with **no entry anywhere**. This dossier closes that gap.

### 5.1 Tinnitus association — re-examined and reinstated, reframed
The original claim was **cluster-based** (atypical/Cluster-1 ≈ 38–50% tinnitus vs ≈18% in the
bulk). Re-examined under current scrutiny (`34_tinnitus_reexamination.json`, N=4,713 with
non-missing AUQ191):

- **Reproduced:** minority cluster 44.4% vs dominant 18.3% (≈ the original gap).
- **The honest test.** Because the cluster is downgraded to a continuum tail, the defensible
  unit is the |z| asymmetry tail and continuous severity — and the obvious confound is that
  asymmetric extremes are also more severe overall. So:
  - **Severity main-effect** (point-biserial r=0.23, p=3×10⁻⁵⁹) is real but **non-novel**
    (tinnitus∼hearing-loss severity is textbook) and does **not** establish the dropped claim.
  - **Asymmetry-specific effect, severity-adjusted** (Mantel–Haenszel across severity quintiles):
    **OR = 1.43, 95% CI [1.08, 1.90] (full Robins–Breslow–Greenland variance), CMH χ² p = 0.012.**
    Severity quintiles leave residual within-stratum confounding (severity is continuous), so
    this adjusted association cannot be claimed as fully severity-independent — but it is not
    explained away by severity either.
- **Verdict:** `asymmetry_assoc_survives_severity_adjusted`. The association is **real but modest** —
  quote the adjusted **OR ≈ 1.4**, never the inflated univariate ratio (2.48). It was dropped in
  v5/v6 not because it was false but because cluster-associated symptom rates do not fit the
  robustness-audit frame. **Reinstated** as: *more inter-ear asymmetry is associated with modestly
  higher tinnitus prevalence beyond overall severity — a continuum gradient, not a cluster property.*

### 5.2 Cisplatin framing — recorded, and the apparent tension is moot
v1–v4 motivated the work via cisplatin ototoxicity (the author's own exposure). It vanished from
the v5/v6 manuscripts. An apparent tension was noted in review: cisplatin literature reports a
slight **left**-ear bias, whereas the salient cluster was **right**-worse. **That tension is moot:**
`audit_03` shows the auditory tail is **bilaterally symmetric** (38R/31L, p=0.47) — there is no
directional finding for the cisplatin direction to contradict. The cluster was the clustering
isolating one side of a symmetric tail. Cisplatin remains the honest *personal* motivation for the
**product** (see `docs/VISION_full_circle.md`), not a scientific claim in the manuscript.

---

## 6. Inventory (canonical vs superseded)

**Canonical (cite these):**
- Manuscript: `docs/en/PAPER_DRAFT_v6_crosssystem.md` + `table_crosssystem_asymmetry.tex`.
- Pipeline: `scripts/_shape_space.py` (single source of truth) + `00`–`33` + `audit_01`–`08`
  + grip (`grip_00`–`grip_04`) + vision (`vis_00`–`vis_04`) + **new guards `26b`, `34`, `35`, `36`**.
- Reusable engineering: `spinoffs/frente1` (methods note), `frente2` (BIC simulation),
  `frente3` (`skfreeze` freeze/score + round-trip test).
- Guards & reproducibility: `tests/` (`test_pipeline_contract.py`, `app_logic.test.js`, `README.md`),
  `requirements-lock.txt`. Committed results: `outputs/json/*.json`.
- Product: `app/` (live empathy tool) — see `docs/VISION_full_circle.md`.

**Superseded (kept for provenance; do not cite as current):**
- Drafts `PAPER_DRAFT.md`, `_v2`, `_v3`, `_v4`, `_v5_audit` — historical stages of §4.
- Working logs `RESULTADOS_SESSAO4/5.md`, `REVISAO_CIENTIFICA_2026-05-29.md`,
  `MUDANCA_v5_AUDITORIA.md`, `CORRECOES_2026-06-01.md`, `RESPOSTA_GEMINI_CLUSTER1.md`,
  `DIALETICA_FERRAMENTAS_ARXIV.md` — the raw audit trail this dossier consolidates.
- `meta_analysis.md` and `docs/en/SYNTHESIS.md` — **superseded by this dossier.**

(Physical archiving of superseded files is a deferred cleanup pass; they remain in-repo as history.)

---

## 7. What this proves

The defensible contribution is **an apparatus, honestly applied**: a null-calibrated, measure-agnostic
audit with an internal negative control, anatomical verification, and per-case sanity — exercised on
one robust positive (auditory asymmetry), one downgraded result (grip), and one negative control
(vision). The strongest evidence of competence is not any single finding but the **documented
self-correction**: claims raised, tested adversarially, and withdrawn or rescaled on the evidence,
with the residual uncertainty stated rather than hidden. That is the standard this dossier holds the
whole project to — and the standard a reader can verify, line by line, against committed outputs.
