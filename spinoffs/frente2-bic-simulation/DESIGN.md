# Simulation note — When is a shallow interior BIC minimum real vs. artifact?

> SKELETON ONLY. Design, not results. Nothing is run here.

## Question (TODO 1 line)
- A GMM BIC interior minimum that appears only under `full` covariance, is shallow (~1–2% of mean),
  and whose argmin-k migrates as n_init rises — signal of real k, or optimization/spec artifact?

## 1. Data-generating processes (known ground truth)
- **G0 — null / no discrete structure (k_true = 1):**
  - single anisotropic Gaussian (correlated covariance — the case that tempts `full`);
  - continuum variant: points along a 1-D curved manifold + noise (mimics shape data).
- **G1 — true structure (k_true ∈ {2,3,4}):** well-separated Gaussian mixture.
- **G2 — intermediate (the crux):** overlapping mixture, separation swept Δ/σ from "merged" to "clear".
- All: report ground-truth labels; match n, d, covariance anisotropy to the real dataset.

## 2. Factors to sweep
| factor | grid (TODO fill) |
|---|---|
| k_true | 1, 2, 3, 4 |
| separation Δ/σ (Mahalanobis) | 0 … large |
| dimensionality d | small … real-d |
| n | small … real-n |
| anisotropy / covariance shape | isotropic … strongly correlated |
| fitted covariance_type | full, tied, diag, spherical |
| n_init | 3, 10, 30, 50 |
| seed | R replicates |

## 3. Quantities to measure (per cell)
- interior-minimum present? (argmin-k ∉ grid edges) per covariance_type
- **depth** of minimum: (max−min)/mean BIC, in %
- **argmin-k stability** across n_init and across seeds (e.g., mode + dispersion of argmin-k)
- concordance of selected k with k_true; ARI(assignment, truth)

## 4. Output = a decision rule (the deliverable)
- Calibrate: under G0 (no structure), what depth / stability do spurious interior minima reach?
- Report depth as a *detector* of real k → threshold + its false-positive / true-positive rate.
- State guidance: "interior minimum with depth < X% AND argmin migrating across n_init ⇒ treat as null."

## 5. Figures/tables to report
- Heatmap: depth vs (separation × covariance_type), faceted by k_true.
- ROC/PR: depth-threshold as detector of "real k present".
- Table: argmin-k stability vs n_init under G0 vs G1.

## 6. Threats / scope (TODO)
- Gaussian DGPs may flatter GMM; include the non-Gaussian continuum case to avoid that.
