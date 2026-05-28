# Model Card — The Frequency ML

**Version:** 1.0  
**Date:** 2026-05-26  
**Authors:** The Frequency Team  
**Status:** Experimental — no clinical label  

---

## 1. Overview

| Field | Value |
|-------|-------|
| **Model name** | HDBSCAN audiometric clustering (shape-only) |
| **Type** | Unsupervised Clustering |
| **Task** | Discover latent patterns of hearing loss in population data |
| **Training data** | NHANES AUX 1999–Mar2020 (9 cycles) |
| **N training** | 7,695 (after filters) |
| **Features** | 14 raw thresholds (500–8000 Hz, bilateral) |
| **Metrics** | ARI, outlier fraction, Gini importance |
| **Intended use** | Auditory empathy research + simulation (The Frequency) |
| **Use not recommended** | Individual clinical diagnosis |

---

## 2. Data

### 2.1 Source

NHANES (National Health and Nutrition Examination Survey), CDC/NCHS. US cross-sectional population survey, with pure tone audiometry per ear/frequency.

| Cycle | Archive | N gross | Frequencies |
|-------|--------|---------|-------------|
| 1999–2000 | AUX1.xpt | 1,807 | 500–8000 Hz |
| 2001–2002 | AUX_B.xpt | 2,046 | 500–8000 Hz |
| 2003–2004 | AUX_C.xpt | 1,889 | 500–8000 Hz |
| 2005–2006 | AUX_D.xpt | 3,034 | 500–8000 Hz |
| 2007–2008 | AUX_E.xpt | 1,210 | 500–8000 Hz |
| 2009–2010 | AUX_F.xpt | 2,368 | 500–8000 Hz |
| 2011–2012 | AUX_G.xpt | 4,500 | 500–8000 Hz |
| 2015–2016 | AUX_I.xpt | 4,582 | 500–8000 Hz |
| 2017–Mar2020 | P_AUX.xpt | 5,147 | 500–8000 Hz |
| **Total** | | **26,583** | |

### 2.2 Applied filters

| Filter | Justification | Before | After |
|--------|--------------------------|-------|--------|
| Age 20–69 | Remove cycles with different eligibility (teens, 70+) | 26,583 | 14,824 |
| Completeness ≥10/14 | Ensure sufficient data per individual | 14,824 | 13,433 |
| ANY25 (≥1 frequency >25 dB) | Remove healthy "sun" that swallowed density | 13,433 | 7,695 |

### 2.3 Known confounding variables

| Variable | Impact | Treatment |
|----------|---------|------------|
| Age | Strong (R² ~0.57 in PTA_high) | Filter 20–69 + row-centering |
| Cycle | Moderate (Cramér's V ~0.16) | Validation per cycle (ARI) |
| Sex | Weak (Cramér's V ~0.12) | Uncontrolled (future) |
| 666 (no response) | 511 lines (1.9%) | Primary policy: NaN + flag |

---

## 3. Preprocessing

### 3.1 Threshold handling

| Code | Meaning | Treatment |
|--------|-------------|------------|
| -10 to 120 dB | Valid | Preserved |
| 666 | No response (severe censorship) | → NaN + flag |
| 888 | Could not obtain | → NaN |
| Others | Missing | → NaN |

### 3.2 Row-centering

For each individual *i*:

$$\mu_i = \frac{1}{14} \sum_{f \in F} T_{i,f}$$

$$T^{shape}_{i,f} = T_{i,f} - \mu_i$$

It removes the average "level" of loss (how much the person loses on average) and preserves the "shape" of the curve (where the loss is greatest/smallest).

### 3.3 Scaling

RobustScaler (IQR-based, quantile_range=(25, 75)). It does not assume normality. Outlier resistant.

### 3.4 Dimensional reduction

PCA with 95% variance explained. Result: 10 components (out of 14).

---

## 4. Model

### 4.1 Algorithm

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

### 4.2 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| min_cluster_size | 10 | Smallest value that finds structure (tested grid: 5–200) |
| min_samples | 5 | HDBSCAN default, tested with 3–20 |
| metrics | euclidean | Standard for continuous data |
| cluster_selection_method | eom | Excess of Mass (default) |
| core_dist_n_jobs | -1 | Parallelism |

### 4.3 Grid tested

| min_cluster_size | min_samples | n_clusters | noise_fraction |
|-----------------|-------------|------------|----------------|
| 5 | 3 | 12 | 0.048 |
| 5 | 5 | 4 | 0.088 |
| **10** | **5** | **2** | **0.076** |
| 15 | 5 | 2 | 0.083 |
| 20 | 5 | 0 | 1,000 |
| 30+ | any | 0 | 1,000 |

---

## 5. Results

### 5.1 Clusters found

| Cluster | n | %

| Geometric description |
|---------|---|---|---------------------|
| 0 | 7,098 | 92.2% | Mild-moderate sloping loss, relatively symmetrical bilateral |
| 1 | 12 | 0.2% | Severe unilateral asymmetry (right ear ~80 dB, left ear ~16 dB) |
| Noise | 585 | 7.6% | Heterogeneous patterns, moderate-severe loss |

### 5.2 Metrics

| Metric | Value | Interpretation |
|---------|-------|---------------|
| HDBSCAN noise | 7.6% | Low (was ~90% before filters) |
| Cross-cycle ARI | 0.27 | Consistency between NHANES cycles with different eligibility (measures robustness to protocol/cohort variation) |
| Bootstrap ARI (medium) | 0.68 | Reproducibility within subsamples of the same population (measures internal stability) |
| Bootstrap ARI (conditional) | 0.60 | When clusters appear (85% of subsamples) |

> **Note:** Cross-cycle ARI and Bootstrap ARI are different metrics. The first measures consistency between different populations (NHANES cycles); the second measures reproducibility within the same population. Both are reported for transparency.
| RF AUC (cluster 0 vs 1) | 1.0 | Perfect separation (unbalanced classes) |

### 5.3 Black box (RF surrogate)

Top 7 discriminative features (all from the right ear):

| Feature | Gini importance |
|---------|----------------|
| thr_R_1000 | 0.2248 |
| thr_R_500 | 0.2203 |
| thr_R_2000 | 0.1453 |
| thr_R_4000 | 0.1175 |
| thr_R_3000 | 0.1174 |
| thr_R_6000 | 0.0711 |
| thr_R_8000 | 0.0427 |

### 5.4 Tinnitus

| Group | Rate | n valid |
|-------|------|----------|
| Cluster 0 | 18.3% | 4,397 |
| Cluster 1 | 50.0% | 8 |

> **Note:** Cluster 1 tinnitus rate is based on N=8 individuals with available data. Interpret as directionally suggestive, not statistically definitive.
| Outliers | 38.0% | 308 |

Chi² p<0.001, Cramér's V=0.126.

---

## 6. Sensitivity Analysis (H11)

| Politics | Treatment 666 | ARI vs nan | Impact |
|----------|----------------|---------------|---------|
| nan (primary) | 666 → NaN + flag | — | Reference |
| cap125 (alternative) | 666 → 125 dB + flag | 0.9914 | Minimum |

511 lines affected (1.9%). ARI 0.99 across policies → result insensitive to 666 treatment.

### 6.2 ANY25 Filter Sensitivity

| Configuration | N | Clusters | Noise | ARI vs primary |
|---------------|---|----------|-------|----------------|
| With ANY25 (primary) | 7,695 | 2 | 7.6% | — |
| Without ANY25 | 13,433 | 2 | 4.4% | 0.85 |

The ANY25 filter removes the "healthy core" but does not distort the discovered structure. 98.9% of Cluster 0 members and 75% of Cluster 1 members are preserved across filter settings.

### 6.3 OHHR Pipeline Consistency

| OHHRConfiguration | N | Noise | PTA×SRT r |
|--------------------|---|-------|-----------|
| Without ANY25 | 581 | 53.0% | 0.015 |
| With ANY25 | 537 | 54.0% | 0.018 |

The ANY25 filter applied to OHHR produces virtually identical results, confirming the pipeline is robust to this choice.

### 6.4 Bootstrap Stability in 4D Space

| Space | N dims | Median ARI | Runs with clusters | SD |
|-------|--------|------------|-------------------|-----|
| 14D (full thresholds) | 10 PCA | 0.68 | 85% | ~0.40 |
| 4D (binaural mean 500/1k/2k/4k) | 4 PCA | **0.74** | **100%** | **0.016** |

The 4-frequency binaural-mean space is *more stable* than the full 14-frequency space. This is the space used for OHHR external validation, strengthening the cross-population comparison.



---

## 7. Validation

### 7.1 Validation per cycle (approximate_predict)

| Cycle | n test | ARI |
|-------|------------|-----|
| 1999–2000 | 949 | 0.17 |
| 2001–2002 | 1,031 | 0.21 |
| 2003–2004 | 1,000 | 0.18 |
| 2011–2012 | 2,238 | 0.37 |
| 2015–2016 | 2,477 | 0.41 |

Average ARI: 0.27. More recent cycles (higher N) have higher ARI.

### 7.2 Bootstrap (100 runs × 80%)

- 85% of subsamples found 2 clusters
- Median ARI: 0.68
- Conditional ARI (when 2 clusters): 0.60
- 15% failure: Cluster 1 (12 people) does not form when subsampled

### 7.3 Va

external dealing — OHHR

**Executed.** OHHR (Oldenburg Hearing Health Record; Jafri et al., 2025): 581 adults (median age 71, median PTA 45 dB), CC BY 4.0.

**Applied pipeline:**
1. Extraction of the 4 common frequencies with NHANES (500, 1000, 2000, 4000 Hz)
2. Row-centering (same operation as NHANES)
3. RobustScaler with NHANES parameters (not re-tuned)
4. Projection onto NHANES-trained PCA (10 components)
5. `approximate_predict` with NHANES HDBSCAN clusterer

**Results:**
- 53% of OHHR fell as noise (vs 37.6% in NHANES) — expected as OHHR is older and clinical
- PTA × SRT correlation: Pearson r=0.015, Spearman r=−0.007 (N=581)
- Interpretation: audiogram does not predict speech in noise ("Factor D")

**Limitation:** OHHR does not separate R/L, making asymmetry comparison impossible. Frequencies limited to 500–4000 Hz — but bootstrapping in 4D showed *greater* stability than 14D (ARI 0.74 vs 0.68).

**Pipeline consistency:** OHHR with ANY25 filter (N=537): 54.0% noise, PTA×SRT r=0.018 — virtually identical to without filter (53.0%, r=0.015).

---

## 8. Limitations

### 8.1 Data limitations

1. NHANES is cross-sectional — there is no individual temporal progression.
2. NHANES has no history of pediatric cisplatin — "platinum-like" is proxy, not confirmation.
3. Frequencies limited to 500–8000 Hz — ototoxicity may begin >8 kHz.
4. Tinnitus is self-reported (AUQ191) — only available in 2005+ cycles.
5. No speech-in-noise — NHANES does not measure functional perception.

### 8.2 Model limitations

1. HDBSCAN is sensitive to min_cluster_size — small clusters may not form.
2. Row-centering removes level — doesn't capture "how bad" the loss is, just the shape.
3. 14 dimensions are few — but they capture 95% of the variance.
4. Cluster 1 (12 people) is too small for population generalization.
5. The 15% bootstrap failure shows sampling sensitivity.

### 8.3 Ethical limitations

1. No clusters received a clinical label — it's geometry, not diagnosis.
2. The founder's personal case was not used in the training.
3. Prevalences should not be inferred without survey weights.
4. The model should not be used for individual clinical decisions.

---

## 9. Recommended Use

| ✅ Can | ❌ Must not |
|---------|------------|
| Research into audiometric standards | Individual clinical diagnosis |
| Auditory empathy simulation | Prevalence inference without weights |
| Generation of clinical hypotheses | Replace audiologist |
| Personal case validation as an external point | Use personal data as a statistical basis |

---

## 10. Reproducibility

### 10.1 Environment

```
Python 3.13+
numpy, pandas, scipy, scikit-learn, hdbscan, joblib, plotly
```

### 10.2 Scripts

20 Python scripts, numbered sequentially. Each script has checkpointing (does not re-execute if output exists).

### 10.3 Data

Public NHANES XPT via CDC. URLs documented in `scripts/00_download_nhanes.py`.

### 10.4 Outputs

15+ JSON files with complete results. All reproducible from the scripts.

---

## 11. References

- NHANES: https://wwwn.cdc.gov/nchs/nhanes/
- HDBSCAN: McInnes, L., Healy, J. (2017). Accelerated Hierarchical Density Based Clustering. ICDM 2017.
- ARI: Hubert, L., Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193-218.

---

## 12. Contact

The Frequency — [insert contact]

---

*Model Card generated on 2026-05-26. No clinical labels were used in training.*