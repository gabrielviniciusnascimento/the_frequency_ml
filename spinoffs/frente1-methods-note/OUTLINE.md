# Methods note — Diagnostics for minority clusters (n small)

> SKELETON ONLY. Nothing below is written. Each technique lists: claim · figure/table it needs.

## Scope (1 paragraph — TODO)
- Problem: validating a small cluster (n≈12) without circular metrics (surrogate AUC inflation).
- Out of scope: how the cluster was found; choice of clustering algorithm.

## 0. Notation & setting (TODO)
- X ∈ ℝ^{n×d}; minority label set P (positives); batches B = {b_1..b_m}; censored-value code c.

## 1. Background — why standard validation is circular here (TODO)
- A surrogate trained on labels derived from the same features → AUC≈1.0, no information.
- Frames the need for the three diagnostics below.

## 2. Diagnostic A — Leave-one-positive-out recognition
- **Claim:** each minority member is individually recognizable from the majority when removed
  from training → the cluster is not carried by any single member / not a one-point artifact.
- **Figure/table needed:** table — per-member held-out membership prob + predicted class +
  recognized(y/n); summary recall = (#recognized)/|P|. Optional: dot/strip plot of the |P| held-out probs.
- **Distinguish from:** in-sample surrogate AUC (state explicitly why LOO is non-circular).

## 3. Diagnostic B — Leave-one-batch-out: project vs. re-cluster
- **Claim:** the cluster structure transfers to an unseen batch — labels assigned by projecting
  the held-out batch into the train-fitted model agree with labels from independently clustering
  that batch → structure is not a within-batch artifact.
- **Figure/table needed:** table — per-batch ARI(projected, independent) + n_batch +
  n_clusters/noise each side. Optional: bar chart ARI by batch with n annotated.

## 4. Diagnostic C — Dual-encoding of censored values
- **Claim:** the discovered structure is invariant to an arbitrary coding choice for censored
  values (encoding A vs B) → not an artifact of that decision.
- **Figure/table needed:** small table — ARI(clustering_A, clustering_B) + n_clusters & noise
  fraction under each encoding. Optional: 2×k contingency of the two label sets.

## 5. Worked example (TODO)
- The n≈12 case end-to-end; one combined figure referencing §2–§4.

## 6. Limitations (TODO)
- Small |P| bounds LOO resolution; ARI sensitive to noise label; batch count m small.

## 7. Reproducibility (TODO)
- Seeds, data access, script pointers.
