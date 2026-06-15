# Changelog

All notable changes to The Frequency ML. This project documents science as much as code,
so entries record what was *claimed*, *tested*, and *withdrawn*, not only what was built.

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
