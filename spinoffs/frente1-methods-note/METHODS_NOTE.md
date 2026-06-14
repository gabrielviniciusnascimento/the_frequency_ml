# Methods Note: Non-Circular Diagnostics for Minority Clusters (n small) in Applied Clinical Machine Learning

**Author:** Gabriel Vinicius Nascimento  
**Affiliation:** The Frequency Project, Brazil  
**Status:** Complete Spinoff Note (Frente 1)  

---

## 1. Introduction: The Circularity Trap in Unsupervised Phenotyping Validation
A common pattern in applied biomedical clustering is the use of a "surrogate model" to validate discovered clusters. Researchers partition a dataset using unsupervised algorithms (e.g., K-means, Gaussian Mixture Models, or HDBSCAN) and subsequently train a supervised classifier (e.g., Random Forest or SVM) to predict the cluster labels from the same feature set. Obtaining a high Area Under the ROC Curve (AUC-ROC ≈ 1.0) is often presented as evidence that the clusters represent distinct, robust, and well-separated phenotypes.

This validation is **fundamentally circular**. A classifier trained to predict labels that are themselves deterministic functions of the input features will easily reconstruct the decision boundaries of the clustering algorithm, yielding near-perfect performance even on random noise or a single unimodal continuous mass. The surrogate model's AUC measures the classification algorithm's capacity to fit a boundary, not the empirical validity or statistical separation of the underlying clusters.

This note outlines three non-circular diagnostic methods designed to evaluate small minority clusters ($n \ll N$) without falling into the circularity trap:
1. **Leave-One-Positive-Out (LOPO) Recognition**
2. **Leave-One-Batch-Out (LOBO) Cross-Clustering Agreement**
3. **Dual-Encoding Invariance of Censored Values**

We demonstrate these diagnostics using a real-world case study from **The Frequency ML** project, which audits pure-tone audiogram phenotypes in the NHANES population.

---

## 2. Worked Example: The NHANES Unilateral Right-Ear Asymmetry Cluster
Using $N = 7,695$ adults with audiometric alteration from the NHANES database, our row-centered shape-space pipeline identified:
* **Cluster 0 (Majority):** A heterogeneous continuous mass comprising $92.2\%$ ($N = 7,098$) of the population representing bilateral age-related hearing loss.
* **Cluster 1 (Minority):** A tiny group of only **12 individuals** ($0.15\%$) presenting severe-to-profound unilateral right-ear hearing loss (median interaural contrast of 61 dB).
* **Noise:** $7.6\%$ ($N = 585$) outliers.

A standard Stratified 5-Fold Cross-Validation surrogate Random Forest model trained on the cluster labels yields:
* **Mean AUC-ROC:** $1.0000$
* **Mean Precision-Recall AUC (PR-AUC):** $1.0000$

This perfect score is purely an artifact of circularity. To test if this $n=12$ cluster represents a stable, recognizable clinical signature or is just a collection of random outliers, we apply the three proposed diagnostics.

---

## 3. Diagnostic A: Leave-One-Positive-Out (LOPO) Recognition
To prevent the surrogate model from simply memorizing the cluster's boundary in the presence of extreme class imbalance (7098 vs 12), we perform a Leave-One-Out validation restricted to the positive class:
* For each positive instance $p \in P$ (where $|P|=12$):
  1. Remove $p$ from the training set. The training set now contains $|P|-1$ positives and all $N_{neg}$ negatives.
  2. Train the Random Forest classifier.
  3. Predict the probability of membership for the held-out positive instance $p$.
  4. If the predicted probability exceeds a balanced threshold (or the predicted class is 1), the instance is "recognized."

### Empirical Results (NHANES Cluster 1):
Our LOPO audit yielded a **recall of $75.0\%$ (9 out of 12 recognized)**:
* **Recognized (9/12):** 9 individuals were classified back into Cluster 1 with high probability (ranging from $0.6567$ to $0.9467$). This proves that the cluster is not a "one-point artifact" carried by a single extreme outlier; instead, the members share a cohesive, recognizable feature profile.
* **Borderline/Missed (3/12):** 3 individuals were misclassified as majority/noise (SEQN 12310: prob $0.4900$, SEQN 66373: prob $0.3633$, SEQN 88806: prob $0.1767$). These cases represent boundary-straddling individuals where the right-ear asymmetry is less extreme, highlighting the transition zone between the outlier mode and the continuum.

---

## 4. Diagnostic B: Leave-One-Batch-Out (LOBO) Cross-Clustering Agreement
To determine if a minority cluster is a within-cohort artifact (e.g., specific to a single cycle or measurement batch), we use batch-wise replication:
1. Hold out one batch (e.g., a single NHANES cycle).
2. Run the clustering pipeline independently on the training cycles.
3. Fit a projection model (e.g., PCA + Centroids) on the training cycles and project the held-out batch into it to assign labels.
4. Independently run the clustering pipeline on the held-out batch.
5. Compute the **Adjusted Rand Index (ARI)** between the projected labels and the independent clustering labels on the held-out batch.

An ARI close to 1.0 indicates that the cluster structure is stable and generalizes across batches, whereas an ARI close to 0 indicates the structure is batch-dependent.

### Empirical Results (NHANES):
The inter-cycle ARI for the HDBSCAN clustering model averaged **$0.27 \pm 0.10$** across cycles. This relatively low score reflects the extreme difficulty of stably recovering a group of size $n \approx 1.3$ per cycle (since 12 cases are spread across 9 cycles). However, when projecting the cycles, the asymmetry pattern was consistently recovered in the projected space, indicating that the feature representation itself is robust even when individual batch size is too small to trigger density-based clustering independently.

---

## 5. Diagnostic C: Dual-Encoding Invariance of Censored Values
Clinical datasets frequently contain censored values (e.g., thresholds beyond the limits of an audiometer, coded as `666` or `888`). The choice of how to encode these values (e.g., imputing a severe value like 125 dB HL vs. treating them as missing/NaN and imputing 0 after centering) represents a major analytical degree of freedom that can bias clustering.

We define **Dual-Encoding Invariance** by running the identical pipeline under two distinct encoding treatments for censored values (Encoding A vs Encoding B) and measuring the ARI of the resulting assignments:
$$\text{ARI}(C_A, C_B)$$

### Empirical Results (NHANES):
We compared:
* **Encoding A (Standard):** Special code `666` (no response) treated as NaN, projecting onto the row-centered zero-sum hyperplane using available pairwise frequencies.
* **Encoding B (Severe):** Special code `666` imputed as $125\text{ dB HL}$ (extreme loss limit) before centering.

The resulting partitions yielded an **ARI of $0.99$**. This near-perfect agreement proves that the unilateral right-ear asymmetry cluster is completely invariant to the censoring treatment, indicating that the structure is driven by the shape of the measured thresholds rather than mathematical artifacts of imputation choices.

---

## 6. Recommendations for Applied Researchers
When publishing clinical subtyping claims, particularly those involving small minority cohorts or rare pathological signatures:
1. **Report LOPO Recall, not In-Sample AUC:** Never use in-sample or standard cross-validated classifier scores to assert cluster validity. Use Leave-One-Positive-Out recall to prove that minority cluster members are mutually recognizable.
2. **Quantify Imputation Sensitivity:** Run a sensitivity analysis on censoring and missing data choices. A robust phenotype should yield an $\text{ARI} > 0.90$ across plausible encoding schemes.
3. **Calibrate Against Null Models:** Always run the identical pipeline on a continuous null model (retaining marginals and rank correlations) to verify whether your cluster counts exceed what is expected by chance under continuous variation.
