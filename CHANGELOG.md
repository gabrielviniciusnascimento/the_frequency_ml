# Changelog

All notable changes to The Frequency ML. This project documents science as much as code,
so entries record what was *claimed*, *tested*, and *withdrawn*, not only what was built.

## [2.1.0] — 2026-06-17

Consolidation + scrutiny-leveling. Adds the canonical **Scientific Dossier** and brings the
previously lower-scrutiny analyses up to the standard of the auditory/continuum claims.

### Added
- **`docs/en/SCIENTIFIC_DOSSIER.md`** — single authoritative record: inventory, evolution of
  focus (evidence- vs strategy-driven), claims×evidence truth table, scrutiny matrix, and the
  honesty-log completion below. Supersedes `meta_analysis.md` and the scattered session logs.
- **New guards** (all deterministic, seed 42, committed JSON):
  - `26b_method_breadth.py` — continuum test extended to DBSCAN / Spectral / Agglomerative
    (complete+average), scored by the conservative max-silhouette-over-the-whole-grid statistic
    (multiple-comparison-aware, not a formal Bonferroni/FDR p-value — silhouette has no null here);
    degenerate outlier-peeling splits (high silhouette from a `[n-1,1]` shave) disqualified via a
    declared ≥5%-minority rule.
  - `35_row_centering_ablation.py` — pre-validates the row-centering choice; the continuum holds
    with and without it (asymmetry tail is centering-invariant).
  - `vis_04_sanity_extremes.py` — per-case audit of vision |OD-OS| extremes (64/64 in
    physiological range), parity with audio (13/13) and grip (15/15); closes a Table 1 footnote.
  - `36_triplet_overlap_audit.py` — formalizes cross-system SEQN/calendar overlap (0 extreme
    overlap across all pairs and the triple) as a committed artifact, not prose.

### Withdrawn / corrected (honesty log)
- **Tinnitus association (v1–v4, dropped silently in v5) — re-examined and reinstated, reframed.**
  The original 38%-vs-18% was cluster-based. `34_tinnitus_reexamination.py`: the severity main
  effect (r=0.23) is real but non-novel; the asymmetry-specific effect, **severity-adjusted
  (Mantel–Haenszel), is modest — OR 1.43, 95% CI [1.08, 1.90], CMH p=0.012**. Reinstated as a
  continuum gradient, quoting the adjusted OR≈1.4, never the univariate 2.48.
- **Cisplatin framing (v1–v4, dropped silently).** Recorded; the apparent left/right tension is
  **moot** — the auditory tail is bilaterally symmetric (`audit_03`), so there is no directional
  finding to contradict. Remains the personal motivation for the product, not a manuscript claim.

## [2.0.0] — 2026-06-15

Major reframe: from a single-system "audiometric phenotyping audit" (v5) to a measure-agnostic,
null-calibrated audit of inter-side asymmetry in paired-organ measurements (v6), hardened by an
8-task adversarial pre-commit audit.

### Added
- **Cross-system pipelines** for three NHANES paired organs: auditory (`27`–`30`), grip
  (`grip_00`–`grip_04`, anatomical side via `MGATHAND`), vision (`vis_00`–`vis_03`).
- **Pre-commit audit** (`HANDOFF_PRECOMMIT_AUDIT.md`, scripts `audit_01`–`audit_08`):
  Monte-Carlo null envelope (B = 2,000, empirical *p*), paired-inclusion robustness,
  side decomposition, heteroscedastic-measurement null, t-copula null, cluster-stability
  sweep, OHHR external replication, and a sensitivity-injection control.
- Manuscript `docs/en/PAPER_DRAFT_v6_crosssystem.md` + Table 1 (`table_crosssystem_asymmetry.tex`).
- `outreach/LINKEDIN_POST.md`; refreshed `CITATION.cff`.

### Changed
- Headline reframed to a **magnitude** gradient (auditory ≫ grip > vision) with **no direction**.
- All single-realization "X vs 0" counts replaced by null-envelope counts with empirical *p*.
- `README.md` and `CITATION.cff` aligned to the audited frame.

### Withdrawn / corrected (honesty log)
- The **lateralized-trauma** reading: the auditory contrast tail is bilaterally symmetric
  (`audit_03`), so no directional/etiologic claim is made. Cox & Ford citation removed.
- **Grip** "genuine excess" downgraded: survives a Gaussian null but not a t-copula (`audit_05`).
- "Vision below its null" scoped to the **general population** only (`audit_02`).
- Claims confined to the far tail (\|z\|≥3); the moderate \|z\|>2 shoulder dissolves under a
  measurement-noise null. The "N = 13 cluster" is one side of a symmetric tail and is
  hyperparameter-fragile (`audit_06`) — the defensible unit is the \|z\| threshold, not the cluster.

## [1.x] — 2026-05/06 (history)
- v4: HDBSCAN audiometric phenotyping on NHANES.
- v5: pivot to a reproducibility audit; OHHR ingestion bug fixed (PTA×SRT r 0.015 → 0.85).
