# Verification Report
**Date:** 2026-06-02  
**Auditor:** Claude Sonnet 4.6 (Claude Code)  
**Scope:** Four independent checks on numerical claims and framing consistency

---

## Check 1 — JSON vs. Paper §3 (PAPER_DRAFT_v5_audit.md)

**Question:** Do the numbers in `outputs/json/26_method_comparison.json` match Section 3 of `docs/en/PAPER_DRAFT_v5_audit.md`?

### All values verified as matching (within stated rounding)

| Location in paper | Paper value | JSON value | Status |
|---|---|---|---|
| §3.1 / Table 1 — silhouette k=2 | 0.282 | 0.2819 | ✓ |
| §3.1 / Table 1 — silhouette k=3–10 | 0.158, 0.159, 0.176, 0.159, 0.106, 0.105, 0.097, 0.100 | 0.1577, 0.1586, 0.1763, 0.1587, 0.1064, 0.1052, 0.0969, 0.1002 | ✓ |
| §3.1 / Table 1 — mean seed ARI k=2–10 | 0.998, 0.813, 0.661, 0.664, 0.987, 0.901, 0.877, 0.759, 0.839 | 0.9981, 0.8126, 0.6613, 0.6638, 0.9873, 0.9005, 0.8774, 0.7590, 0.8391 | ✓ |
| §3.2 — ARI k=6 "partial recovery" | 0.99 | 0.9873 | ✓ |
| §3.3 / Table 2 — BIC range full covariance | 1.7% | 1.718% | ✓ |
| §3.3 / Table 2 — BIC range tied/diag/spherical | 3.8%, 1.5%, 8.3% | 3.809%, 1.476%, 8.296% | ✓ |
| §3.3 — k=4→k=5 with n_init increase | k=4 (n_init=3), k=5 (n_init=10) | `bic_min_k: 4` (top-level, n_init=2); `full.bic_min_k: 5` (n_init=10) | ✓ |
| §3.4 — HDBSCAN noise fraction | 7.6% | `noise_fraction: 0.076` | ✓ |
| §3.4 — dominant cluster | 92.2% | `largest_cluster_fraction: 0.9223` | ✓ |

### One divergence found

**§3.4 — Small HDBSCAN cluster size**

The paper states the small cluster has **N=12**. The JSON values imply **N=13**:

```
n_samples            = 7695
n_noise              = 585          (7.6%)
largest_cluster      = round(0.9223 × 7695) = 7097
small_cluster        = 7695 − 585 − 7097 = 13
```

Cross-check: `7097 / 7695 = 0.92227` rounds to **0.9223** (consistent with JSON). A cluster of N=12 would yield `7098/7695 = 0.9224`, not 0.9223. The N=12 figure and the leave-one-out recall of 0.75 cited in §3.4 are not present in `26_method_comparison.json` and likely came from a separate, more detailed HDBSCAN analysis. **The paper should cite whichever script produced N=12 and reconcile the discrepancy.**

---

## Check 2 — Fresh run of `scripts/26_method_comparison.py` (n_init=10)

**Question:** With n_init=10, does the BIC interior minimum still appear only under `covariance_type='full'`?

**Environment:** conda env `uirapuru` (Python 3.11.15, scikit-learn 1.8.0, hdbscan 0.8.44).  
`hdbscan` was absent and installed prior to the run.  
Run time: ~54 s.

### BIC robustness result — confirmed

| Covariance | `bic_min_k` | Interior minimum | BIC range (% of mean) |
|---|---|---|---|
| full | **5** | **yes (shallow)** | **1.718%** |
| tied | 10 (boundary) | no | 3.809% |
| diag | 10 (boundary) | no | 1.476% |
| spherical | 10 (boundary) | no | 8.296% |

All four summary values are **bit-for-bit identical** to the committed JSON. `diag` and `spherical` now also include `bic_by_k` curves (absent from the committed version). K-means and HDBSCAN blocks are also bit-for-bit identical.

### Structural differences between committed JSON and fresh run

| Field | Committed JSON | Fresh run |
|---|---|---|
| `bic_robustness_covariance` structure | Nested under `"results": {...}` with `"method"` and `"conclusion"` keys | Covariates at top level; `"_conclusion"` and `"_n_init"` keys |
| `diag` / `spherical` `bic_by_k` | Absent | Present |
| `interpretation` text | "RESOLVIDO (favorável a contínuo…)" — manual curation | "EVIDÊNCIA MISTA, INCLINADA A CONTÍNUO" — script heuristic |
| `review_decision_needed` | "RESOLVIDO com n_init=10: K natural não-robusto…" | Preliminary decision template text |
| `elapsed_s` | 17.9 | Absent |

**The fresh run overwrote `outputs/json/26_method_comparison.json` on disk.** The committed JSON's curated `interpretation` and `review_decision_needed` fields are not reproducible by the current script. If the JSON is committed from this run those editorial notes will be lost.

### Conclusion

The scientific claim holds: with n_init=10, a BIC interior minimum appears **only** under `covariance_type='full'`, and even there it is shallow (1.7% of mean) and unstable to optimization effort (k=4 at n_init=2 → k=5 at n_init=10). tied, diagonal, and spherical covariances show no interior minimum.

---

## Check 3 — OHHR ingestion fix in `scripts/25_external_validation_ohhr.py`

**Question:** Is the merge using `audiogram_line` and filtering `htl+ac`? What is the correct PTA×SRT?

### Schema verification

Key join fields confirmed present in the data:

| Table | Key fields |
|---|---|
| `audiogram_point` | `audiogramlineid`, `frequency`, `level` |
| `audiogram_line` | `audiogramlineid`, `audiogramid`, `type` (`htl`/`ucl`), `transducertype` (`ac`/`bc`), `side` |
| `audiogram` | `audiogramid`, `clientid` |

### Merge chain — correct

The script joins:
```
audiogram_point ──[audiogramlineid]──▶ audiogram_line ──[audiogramid]──▶ audiogram ──▶ clientid
```
Zero null `clientid` values after the merge (20,538 rows, all matched).

The old bug joined `audiogram_point.audiogramlineid` directly to `audiogram.audiogramid` — keys from different namespaces — matching only 3,433 of 20,538 points and mixing air/bone conduction and HTL/UCL levels.

### Filter `type='htl'` and `transducertype='ac'` — correct

`audiogram_line` contains exactly two values for each field: `type` ∈ {`htl`, `ucl`} and `transducertype` ∈ {`ac`, `bc`}. The filter removes UCL (uncomfortable loudness levels) and bone conduction, as intended.

After filter (HTL + AC + frequencies 500/1k/2k/4k Hz): **4,625 points** from **581 unique clients** (all OHHR participants represented, no client left behind).

### PTA×SRT — independently recalculated

PTA = best-ear mean of raw thresholds at 500/1k/2k/4k Hz (not row-centered).  
SRT = DTT binaural speech-in-noise SNR threshold.

| Metric | Recalculated | Committed JSON | Paper §3.5 |
|---|---|---|---|
| Pearson r | **0.8537** | 0.8537 | 0.85 |
| Spearman r | **0.9115** | 0.9115 | 0.91 |
| N | **581** | 581 | 581 |
| PTA range | −5.0 to 78.75 dB HL | −5.0 to 78.75 dB HL | — |
| PTA SD | 17.24 dB | 17.2 dB | — |
| Previous buggy value | — | 0.0151 | — |

All values match the committed JSON and the paper to stated precision. The ingestion bug is confirmed fixed.

### Structural note

The committed `outputs/json/25_ohhr_validation.json` contains fields (`ingestion_fix`, `pta_definition`, `srt_definition`, `pta_range_db`, `pta_sd`, `prev_buggy_pearson_r`) that the **current script would not produce** if run from scratch (the script has an early-exit guard when the output exists, and those fields are not in its `result` dict). These metadata fields are preserved only in the committed JSON. If the output file is ever deleted and the script re-run, they will be lost.

---

## Check 4 — Repository-wide search for r=0.015 and "validação externa" framing

**Question:** Which files still report r=0.015 or frame OHHR as "external validation" instead of "exploratory projection"?

### False positives (grep hit on correction-context text — status OK)

- **`LITERATURA_REVIEW.md:137`** — r=0.015 inside a "Correção (2026-06):" block followed by the corrected value 0.85. ✓  
- **`MODEL_CARD.md` (root) lines 209, 261** — table row with `~~0.015 (artifact)~~` and explicit retraction paragraph. ✓  
- **`outputs/json/25_ohhr_validation.json:41`** — field `"prev_buggy_pearson_r": 0.0151`. ✓  
- **`docs/en/PAPER_DRAFT_v4.md:193,257`** — r=0.015 cited as the ingestion artifact in a correction explanation. ✓  
- **`docs/en/PAPER_DRAFT.md`, `v2.md`, `v3.md`** — all carry ⚠️ SUPERSEDED banner with explicit bug disclosure. ✓

### A. Files presenting r=0.015 as a valid current result

#### Urgent — not tracked in `CORRECOES_2026-06-01.md`

| File | Lines | Issue |
|---|---|---|
| `docs/en/PAPER_DRAFT_v4.tex` | 439, 465, 602, 667 | Reports "near-zero correlation... r=0.015" as a finding, no correction notice. The `.md` counterpart was corrected but the `.tex` was not. This is the only renderable LaTeX file and would produce a wrong PDF. |
| `outputs/json/ohhr_any25_validation.json` | 11 | `"pearson_r": 0.015` with no bug flag — any script that reads this file would use the wrong value silently. |

#### Tracked in CORRECOES §7 as pending linguistic review

| File(s) | Lines | Note |
|---|---|---|
| `docs/en/MODEL_CARD.md` | 198, 252, 257 | r=0.015 in table and results section without correction notice |
| `docs/en/README.md` | 33, 56 | r=0.015 as summary statistic |
| `docs/en/README_final.md` | 33, 56 | Idem |
| `docs/en/README_tool.md` | 33, 56 | Idem |
| `docs/de/README.md`, `docs/de/MODEL_CARD.md` | 33/56; 252/257 | German translation |
| `docs/es/README.md`, `docs/es/MODEL_CARD.md` | 33/56; 252/257 | Spanish translation |
| `docs/fr/README.md`, `docs/fr/MODEL_CARD.md` | 33/56; 252/257 | French translation |
| `docs/pt/MODEL_CARD.md` | 196, 248, 253 | Portuguese translation |

#### Low-priority (internal session notes)

| File | Lines | Note |
|---|---|---|
| `RESULTADOS_SESSAO5.md` | 66 | "r=0.018 (vs 0.015)" — session log, not a published artifact |

### B. "Validação externa" / "External validation" framing (should be "projeção exploratória")

#### Urgent — not tracked in CORRECOES

| File | Lines | Issue |
|---|---|---|
| `scripts/25_external_validation_ohhr.py` | 4, 63, 222 | Docstring title, log message, and `status` field all say "Validação externa". The script is the authoritative source and produces JSON with that label. |
| `docs/en/PAPER_DRAFT_v4.tex` | Section 2.6, 3.6 titles | Section still titled "External Validation (OHHR)" — not updated when the `.md` was reframed. |
| `docs/en/PITCH_MICROSOFT_AI4A.md` | 30 | "External validation on OHHR dataset (581 people, Germany)" |

#### Tracked in CORRECOES §7 as pending

| File(s) | Lines | Note |
|---|---|---|
| `docs/en/MODEL_CARD.md` | 210 | "space used for OHHR **external validation**" |
| `docs/en/README.md` / `README_final.md` / `README_tool.md` | 122, 124, 142, 144, 162, 164 | Section heading "External validation" |
| `README.pt.md`, `THE_FREQUENCY_ML_README.md` | 122, 142 | Section heading "Validação externa" |

### Priority matrix

| Priority | Item | Action needed |
|---|---|---|
| **P1** | `docs/en/PAPER_DRAFT_v4.tex` | Update r=0.015→0.85 (4 occurrences) and rename §2.6/§3.6 from "External Validation" to "Exploratory Cross-Population Probe" |
| **P1** | `outputs/json/ohhr_any25_validation.json` | Add `"ingestion_bug": true` flag or `"pearson_r_note": "artifact — buggy ingestion"` to prevent silent misuse |
| **P2** | `scripts/25_external_validation_ohhr.py` | Update docstring, log string, and `status` field to "projeção exploratória" |
| **P2** | `docs/en/PITCH_MICROSOFT_AI4A.md` | Update framing |
| **P3** | All `docs/en/` and `docs/{de,es,fr,pt}/` public-facing files | Per CORRECOES §7 — update when native-language review is complete |

---

*Report generated from live checks run on 2026-06-02. All numerical results were independently reproduced from raw data files; no values were taken from memory or prior summaries.*
