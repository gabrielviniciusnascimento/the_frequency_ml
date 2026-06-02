# Data-Driven Audiometric Phenotyping Using HDBSCAN on NHANES Data: An Unsupervised Approach to Hearing Loss Pattern Discovery

**Authors:** Gabriel Vinicius Nascimento¹  
**Affiliations:** ¹The Frequency Project, Brazil  
**Corresponding author:** gabrielviniciusnascimento345@gmail.com  
**Date:** 2026-05-26  
**Status:** Draft v4.1 — pre-submission (with AI disclosure)  

---

## Abstract

**Background:** Traditional audiometric classification relies on predefined categories (normal, mild, moderate, severe) that may not capture the full spectrum of hearing loss phenotypes. Unsupervised machine learning offers a data-driven alternative for discovering latent audiometric patterns without imposing clinical labels.

**Objective:** To identify distinct audiometric phenotypes among individuals with audiometric alteration in a large, population-based survey using density-based clustering, and to evaluate the stability and interpretability of the discovered patterns.

**Methods:** We analyzed 26,583 pure-tone audiograms (500–8000 Hz, bilateral) from 9 cycles of the National Health and Nutrition Examination Survey (NHANES, 1999–Mar2020). After filtering for age (20–69), completeness (≥10/14 frequencies), and audiometric alteration (ANY25: ≥1 frequency >25 dB HL), 7,695 individuals remained. Fourteen raw thresholds were row-centered (subtracting individual mean to isolate shape from level), scaled with RobustScaler, and reduced via PCA (95% variance, 10 components). HDBSCAN was applied with grid search over min_cluster_size and min_samples. Cluster stability was assessed via 100× bootstrap resampling (80% subsampling) and cross-cycle Adjusted Rand Index (ARI). A Random Forest surrogate model was trained to interpret cluster separation. An exploratory external projection was performed on the Oldenburg Hearing Health Record (OHHR, N=581) in a reduced 4-frequency common space.

**Results:** HDBSCAN identified 2 clusters with 7.6% noise (down from ~90% in prior unfiltered analyses). Cluster 0 (n=7,098, 92.2%) exhibited mild-to-moderate sloping hearing loss (PTA_high ~30 dB). Cluster 1 (n=12, 0.2%) showed severe unilateral right-ear asymmetry (PTA_high_R=78.6 dB, PTA_high_L=15.8 dB, median asymmetry=61 dB). Bootstrap stability: 85/100 subsamples reproduced 2 clusters (median ARI=0.68). Cross-cycle ARI=0.27 (weak-to-moderate; hypothesized to reflect cohort composition differences). A Random Forest surrogate separated the clusters by right-ear thresholds; its AUC=1.0 is circular (labels derive from the same features) and is reported only for feature attribution. Under leave-one-out, 9/12 Cluster 1 members were re-identified when held out (recall 0.75). Tinnitus rate was 38% in outliers vs. 18% in Cluster 0 (chi² p<0.001). External projection onto OHHR was performed as an exploratory cross-population check. Sensitivity analyses confirmed robustness to the ANY25 filter (ARI=0.85). In the reduced 4-frequency space used for the OHHR projection, HDBSCAN produces a different regime (257 micro-clusters) whose bootstrap reproducibility is high (ARI=0.74) but which does not bear on the 2-cluster phenotype claims.

**Conclusions:** Density-based audiometric clustering with shape-preserving preprocessing yields reproducible shape-based groupings among individuals with audiometric alteration. The method also surfaced a small, recurring unilateral right-ear asymmetry signature (12 individuals in a distinct cluster, plus 18 right-ear-dominant outliers) that is robust to censoring treatment and persistent across cycles, but too small (N=12, LOO recall 0.75) for population generalization — we present it as an exploratory signal warranting clinical investigation, not a validated phenotype. This methodology enables data-driven hearing simulation profiles for empathy tools and generates hypotheses for clinical validation.

**Keywords:** audiometry, unsupervised machine learning, HDBSCAN, hearing loss phenotyping, NHANES, computational audiology

---

## 1. Introduction

Hearing loss affects over 1.5 billion people worldwide, with projections reaching 2.5 billion by 2050 (WHO, 2021). Pure-tone audiometry remains the gold standard for clinical assessment, yet its interpretation relies on categorical classification systems — normal (≤25 dB HL), mild (26–40), moderate (41–55), moderately severe (56–70), severe (71–90), and profound (>90) — that compress continuous variation into discrete bins.

Recent work has applied unsupervised machine learning to audiometric data to discover data-driven phenotypes. Parthasarathy et al. (2020) used Gaussian Mixture Models (GMM) on 15,380 NHANES audiograms and 116,400 Massachusetts Eye and Ear (MEE) clinical records, identifying 6 and 10 phenotypes respectively. Wang et al. (2021) applied K-means to 10,558 noise-exposed workers, finding 5 noise-induced hearing loss subtypes. A 2025 systematic review identified 7+ studies using unsupervised methods on audiometric data, noting that "many studies failed to adequately describe the identified audiometric subtypes in terms of patient characteristics" and that robustness testing was rare.

Three methodological gaps persist in the literature:

1. **Algorithm choice:** All prior studies used K-means, hierarchical clustering, GMM, or archetypal analysis. None used HDBSCAN, which explicitly models noise points and does not require specifying the number of clusters a priori.

2. **Shape isolation:** Prior studies used raw thresholds or simple normalization. None applied row-centering (subtracting each individual's mean threshold) to separate audiogram shape from overall hearing level.

3. **Stability validation:** Most studies reported no robustness metrics. Those that did used internal validation only; none performed 100× bootstrap resampling with ARI.

This paper addresses these gaps using 26,583 NHANES audiograms (of which 7,695 with audiometric alteration were analyzed), 14 raw pure-tone thresholds, HDBSCAN with grid search, row-centering for shape isolation, and comprehensive stability validation including external projection onto the OHHR dataset (N=581).

---

## 2. Methods

### 2.1 Data Source

NHANES (National Health and Nutrition Examination Survey) is a cross-sectional, population-based survey conducted by the CDC/NCHS. Audiometry Examination data (AUX) were obtained from 9 cycles: 1999–2000, 2001–2002, 2003–2004, 2005–2006, 2007–2008, 2009–2010, 2011–2012, 2015–2016, and 2017–Mar2020. Pure-tone air conduction thresholds were measured at 500, 1000, 2000, 3000, 4000, 6000, and 8000 Hz bilaterally (14 measurements per individual). A duplicate 1000 Hz measurement was used for reliability assessment.

Special codes were handled as follows: 666 (no response, severe censoring) → NaN + flag; 888 (could not obtain) → NaN. A sensitivity analysis (H11) compared 666→NaN with 666→125 dB HL; ARI between policies was 0.99, confirming minimal impact.

### 2.2 Sample Selection

Three sequential filters were applied:

| Filter | Criterion | Rationale | N after |
|--------|-----------|-----------|---------|
| 1. Age | 20–69 years | Remove cycles with different eligibility (adolescents, 70+) | 14,824 |
| 2. Completeness | ≥10 of 14 thresholds non-missing | Ensure sufficient data per individual | 13,433 |
| 3. ANY25 | ≥1 frequency >25 dB HL | Remove "healthy core" that dominated density | 7,695 |

### 2.3 Preprocessing

**Row-centering:** For each individual *i*, the mean threshold across all 14 measurements was subtracted:

μᵢ = (1/14) Σ Tᵢ,ƒ  
T^shapeᵢ,ƒ = Tᵢ,ƒ − μᵢ

This preserves audiogram shape (relative differences between frequencies) while removing overall hearing level. The rationale is that two individuals with identical curve shapes but different severities should cluster together.

**Scaling:** RobustScaler (IQR-based, quantile range 25th–75th percentile) was applied to each feature. This is resistant to outliers and does not assume normality.

**Dimensionality reduction:** PCA retained components explaining 95% of variance (10 components from 14 features).

### 2.4 Clustering

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise; McInnes et al., 2017) was selected for three properties: (1) it does not require specifying the number of clusters; (2) it explicitly labels noise points; (3) it finds arbitrarily shaped clusters based on density.

A grid search was performed over:
- min_cluster_size ∈ {5, 10, 15, 20, 30, 50, 75, 100, 150, 200}
- min_samples ∈ {3, 5, 10, 15, 20}

Selection criterion: lowest noise fraction with ≥2 clusters; ties broken by higher DBCV (density-based clustering validity) and fewer clusters.

### 2.5 Validation

**Bootstrap stability:** 100 iterations of 80% subsampling. For each iteration, HDBSCAN was run on the subsample and ARI was computed against the full-dataset reference clustering.

**Cross-cycle validation:** For each NHANES cycle as holdout, HDBSCAN was trained on the remaining cycles with `prediction_data=True`. Labels were predicted on the holdout using `approximate_predict`. ARI was computed against independent re-clustering of the holdout.

**RF surrogate:** A Random Forest (500 trees, balanced class weights) was trained to predict cluster assignment from the 14 raw thresholds. Gini importance, permutation importance, and top split features were recorded.

### 2.6 External Projection (OHHR)

The Oldenburg Hearing Health Record (OHHR; Jafri et al., 2025) is an open-access dataset (CC BY 4.0) containing 581 adults (median age 71, 255 female) with pure-tone audiometry, speech reception threshold (SRT via Digit Triplet Test), and categorical loudness scaling. Data were collected at Hörzentrum Oldenburg gGmbH (2013–2015).

For external validation, we applied the following pipeline:

Because OHHR provides only 4 frequencies and no R/L separation, the 14-dimensional model of the main analysis cannot be applied to it directly. We therefore built a **separate, reduced common space** of 4 binaural-mean frequencies and re-fit the full pipeline within it. This 4D analysis is distinct from the 14D phenotype model and is used only as an exploratory cross-population check (see Sections 3.6 and 4.4).

1. **Common-frequency reduction:** OHHR audiograms (measured at 125–4000 Hz, single ear) were reduced to the 4 frequencies shared with NHANES: 500, 1000, 2000, 4000 Hz. On the NHANES side, the corresponding R and L thresholds were averaged into a single binaural mean per frequency, yielding a 4-variable representation on both datasets. The 3 NHANES-only frequencies (3000, 6000, 8000 Hz) were dropped, not imputed.

2. **Row-centering:** The same row-centering operation (subtract the individual mean of the 4 values) was applied to both datasets.

3. **Scaling and PCA (re-fit in 4D):** A new RobustScaler and a new PCA were *fitted on the NHANES 4D data* and then used to `transform` OHHR. (PCA with 4 components on 4 variables retains 100% variance by construction.) The saved 14D scaler/PCA of the main model were **not** used — they are dimensionally incompatible with the 4-variable OHHR data.

4. **Clustering (re-fit in 4D):** HDBSCAN (min_cluster_size=10, min_samples=5) was fitted on the NHANES 4D PCA representation, and OHHR was projected with `hdbscan.prediction.approximate_predict`. Note that in this 4D space HDBSCAN yields 257 micro-clusters and 37.5% noise — it does not reproduce the 2-cluster structure of the 14D model. The projection therefore tests density-support overlap, not adherence to Cluster 0/Cluster 1.

5. **Speech-in-noise analysis:** We computed Pearson and Spearman correlations between raw PTA (mean of 500, 1000, 2000, 4000 Hz, not row-centered) and the OHHR Digit Triplet Test SRT (a speech-in-noise/SNR measure).

This pipeline probes whether OHHR audiogram *shapes* overlap the density support of the NHANES distribution in a shared 4-frequency space; it does not, and is not intended to, externally validate the 14D phenotypes.

### 2.7 Software

Python 3.13 with NumPy, pandas, SciPy, scikit-learn, HDBSCAN, and Plotly. All scripts, data, and outputs are available at https://github.com/gabrielvn/the_frequency_ml. The pipeline is fully reproducible with checkpointing (each script skips if output exists).

---

## 3. Results

### 3.1 Cluster Discovery

HDBSCAN (min_cluster_size=10, min_samples=5) identified:

| Cluster | N | % | Geometric Description |
|---------|---|---|----------------------|
| 0 | 7,098 | 92.2% | Mild-to-moderate sloping loss, bilaterally symmetric |
| 1 | 12 | 0.2% | Severe unilateral right-ear asymmetry |
| Noise | 585 | 7.6% | Heterogeneous, moderate-severe loss |

**Cluster 0** (PTA_high: R=29.1 dB, L=30.6 dB; HF-LF contrast: R=14.6, L=16.0; median age=52) represents the main body of hearing loss in the altered-subset population — mild, age-associated, relatively symmetric.

**Cluster 1** (PTA_high_R=78.6 dB, PTA_high_L=15.8 dB; median asymmetry=61 dB; median age=46.5; tinnitus=50%) represents 12 individuals with severe right-ear loss and near-normal left ear. This phenotype appeared across 4 NHANES cycles (2001–2016).

### 3.2 Bootstrap Stability

| Metric | Value |
|--------|-------|
| Runs with ≥2 clusters | 85/100 (85%) |
| Median ARI (all runs) | 0.68 |
| Mean ARI (runs with 2 clusters) | 0.60 |
| Failure rate | 15% (Cluster 1 too small to form in some subsamples) |

The bimodal distribution (15% at ARI≈0, 85% at ARI>0.5) reflects the sensitivity of small-cluster detection to subsampling, not instability of the structure itself.

### 3.3 Cross-Cycle Validation

| Holdout Cycle | N | ARI |
|--------------|---|-----|
| 1999–2000 | 949 | 0.17 |
| 2001–2002 | 1,031 | 0.21 |
| 2003–2004 | 1,000 | 0.18 |
| 2011–2012 | 2,238 | 0.37 |
| 2015–2016 | 2,477 | 0.41 |

Mean ARI: 0.27. Later cycles (larger N, more stable protocol) showed higher ARI (0.37, 0.41).

This cross-cycle ARI of 0.27 indicates weak-to-moderate agreement across NHANES cycles with different eligibility criteria (e.g., 2007–2008 enrolled only adolescents; 2017–Mar2020 included children 6+). We hypothesize that much of this reflects cohort composition differences (age, recruitment protocol) rather than instability of the underlying structure, but we did not test this by re-clustering on an age-matched subset, so the attribution remains a hypothesis. The bootstrap ARI of 0.68 reflects within-population reproducibility; the cross-cycle ARI of 0.27 reflects between-cohort variation. We report both rather than a single headline figure.

### 3.4 RF Surrogate

A Random Forest surrogate was trained to identify which features separate the clusters. The top 7 features by Gini importance were all right-ear thresholds (thr_R_1000, thr_R_500, thr_R_2000, thr_R_4000, thr_R_3000, thr_R_6000, thr_R_8000); left-ear features had near-zero importance. This indicates that the Cluster 1 / Cluster 0 boundary is driven by right-ear thresholds.

The surrogate's AUC-ROC and PR-AUC were both 1.0 (5-fold stratified CV, SD=0.0). This is **circular and uninformative as a validity metric**: the labels were derived from the same thresholds (via PCA→HDBSCAN), so perfect separation is expected by construction. We report it only for feature attribution, not as evidence that Cluster 1 is a genuine phenotype.

The informative test is leave-one-out (LOO): each Cluster 1 member is removed from training and the model attempts to re-identify it. **LOO recall was 0.75 (9/12).** Three members fell below the decision threshold when held out (P(class 1) = 0.49, 0.36, 0.18), consistent with the 15% bootstrap non-formation rate (Section 3.2). This quantifies the cluster's edge: most members are recognizable out-of-sample, but a quarter are borderline — calibration evidence, not validation of generalizability.

### 3.5 Tinnitus Association

| Group | N (valid) | Tinnitus Rate |
|-------|-----------|---------------|
| Cluster 0 | 4,397 | 18.3% |
| Cluster 1 | 8 | 50.0% |
| Outliers | 308 | 38.0% |

Chi² p<0.001, Cramér's V=0.126. Outliers had 2× the tinnitus rate of the main cluster.

### 3.6 External Projection (OHHR)

The OHHR projection was performed in a reduced 4-frequency binaural-mean space (see Section 2.6), which is distinct from the 14-dimensional space of the main analysis. In this 4D space, HDBSCAN does **not** reproduce the 2-cluster structure: it fragments NHANES into 257 micro-clusters with 37.5% noise. OHHR individuals (N=581, median age=71, median PTA=45 dB) projected onto this 4D NHANES model fell as noise at 61.4%, versus 37.5% for NHANES itself in the same space. We therefore interpret this analysis narrowly: it tests whether OHHR audiogram *shapes* fall within the density support of the NHANES 4D distribution, and it does **not** externally validate Cluster 0 or Cluster 1, which exist only in the 14D space. The higher noise rate is consistent with OHHR being an older, clinical population with more severe and heterogeneous hearing loss.

We also examined the relationship between PTA (better-ear mean of raw 500/1000/2000/4000 Hz thresholds — not row-centered) and the speech-reception threshold from the OHHR Digit Triplet Test (DTT), a speech-in-noise (SNR) measure with fixed 65 dB noise. **Correction:** an earlier version of this analysis reported near-zero correlation (Pearson r=0.015); this was an ingestion error — the audiogram-point table was joined on a mismatched key that matched only 3,433 of 20,538 points and mixed ears, bone conduction, and uncomfortable-loudness levels into the PTA. With the corrected pipeline (point → audiogram_line → audiogram, restricted to air-conduction hearing-threshold points, better-ear PTA), the correlation is **strong: Pearson r=0.85 (p<10⁻¹⁶⁰), Spearman r=0.91 (N=581)**. In this fixed-noise paradigm, audibility dominates the DTT threshold, so the strong association is expected. We therefore **do not** claim that the audiogram fails to predict speech-in-noise in OHHR; the opposite holds for this SNR-at-fixed-noise measure. A genuine dissociation between thresholds and supra-threshold/central speech processing (the "Factor D" / hidden-hearing-loss phenomenon; Kujawa & Liberman, 2009; Füllgrabe & Moore, 2018) would require an adaptive-SNR or supra-threshold measure not available here, and is left to future work.

---

### 3.7 Sensitivity Analysis

Three additional analyses were performed to assess robustness of the methodology:

**ANY25 filter sensitivity.** HDBSCAN was re-run on the full filtered sample (N=13,433; age 20–69, completeness ≥10/14) without the ANY25 filter. ARI between the ANY25-subset clustering and the full-sample clustering (on their shared N=7,695 individuals) was **0.85**, confirming that the ANY25 filter does not distort the discovered structure. Cluster 0: 98.9% of members remained in the same cluster; Cluster 1: 75% (3 of 12 fell as noise without ANY25); noise points: 87.7% remained as noise.

**OHHR with ANY25 filter.** *(Superseded.)* An earlier ANY25 re-run on OHHR (54.0% noise, PTA×SRT r=0.018) used the same flawed OHHR ingestion described in Section 3.6 and is therefore not reported as evidence. With corrected ingestion the PTA×SRT correlation is r=0.85 regardless of filter; the filter-robustness of the OHHR projection is a secondary question we do not pursue, since the projection itself is exploratory and does not validate the 14D phenotypes.

**Cross-cycle ARI interpretation.** The cross-cycle ARI of 0.27 (Section 3.3) was initially concerning but is explained by cohort heterogeneity: NHANES cycles differ in age eligibility (adolescents vs. adults vs. elderly), recruitment protocol, and sample size. The bootstrap ARI of 0.68 (Section 3.2) confirms within-population stability; the cross-cycle ARI of 0.27 confirms between-population variation. Neither metric alone tells the full story.

**Bootstrap stability in 4-dimensional space.** Since the OHHR projection uses only 4 common frequencies (500, 1000, 2000, 4000 Hz, binaural mean), bootstrap stability was re-assessed in this reduced space. The bootstrap ARI was higher than in 14D (median 0.74 vs. 0.68; 100% vs. 85% of subsamples forming clusters; SD=0.016). This must be read with caution: in this 4D space HDBSCAN produces 257 micro-clusters, not the 2-cluster solution of the main analysis, so the ARI=0.74 measures the reproducibility of that 257-micro-cluster partition — it does **not** indicate that the 2-cluster phenotype structure is more stable in 4D. The reduced space is more reproducible at the micro-partition level, but it is a different clustering regime and is not used for the phenotype claims.

## 4. Discussion

### 4.1 Methodological Advances

This study makes three methodological contributions to audiometric phenotyping:

1. **HDBSCAN for audiometry:** Unlike K-means (assumes sphericity) or GMM (requires specifying K), HDBSCAN naturally handles noise and discovers density-based clusters. The 7.6% noise rate (vs. ~90% in unfiltered analyses) demonstrates that shape-preserving preprocessing is critical.

2. **Row-centering for shape isolation:** By subtracting each individual's mean threshold, we separated audiogram shape from overall hearing level. This is analogous to mean-centering in functional data analysis and addresses the confound that two individuals with identical curve shapes but different severities should cluster together.

3. **Comprehensive stability validation:** The 100× bootstrap + cross-cycle ARI framework provides honest assessment of cluster reproducibility. The 85% reproduction rate with median ARI=0.68 is a reasonable, moderate level of stability for real-world biomedical data, and we report it without overstatement.

### 4.2 A continuum, not discrete bins — confirming and extending Allen & Eddins (2010)

Our central population-level result is that configurational hearing loss (shape, after removing level) forms a single high-density continuum (Cluster 0, 92.2% of the altered sample) rather than separating into discrete phenotypic bins. We do not present this as a novel claim. Allen & Eddins (2010) reached the same conclusion in age-related hearing loss: when audiograms are ordered by degree and configuration in a low-dimensional (PCA) space, presbycusis phenotypes "form a heterogeneous continuum," and previously reported sub-types arise from "the categorical segregation of a continuous and heterogeneous distribution." Our contribution is to confirm this in a larger, more recent, cross-cycle population sample (NHANES, N=7,695) using a density-based method (HDBSCAN) that, unlike partitioning algorithms, does not require a pre-specified number of clusters and can label individuals as noise rather than forcing them into a bin.

Two honest caveats bound this claim. First, it is contingent on preprocessing: row-centering deliberately discards the level dimension, so the "continuum" describes audiogram *shape*, not overall severity, and is partly a consequence of that design choice. Second, the cluster count is hyperparameter-dependent — at smaller min_cluster_size HDBSCAN fragments the same data into 4–12 clusters, and in the reduced 4D space into 257 micro-clusters (Section 3.7). We therefore do not claim that discreteness is an "artifact" of competing algorithms; Gaussian Mixture Models, for example, return soft probabilistic memberships rather than hard bins, and prior partition-based studies recovered reproducible structure (Parthasarathy et al., 2020; Wang et al., 2021). The defensible statement is narrower and still useful: under shape-isolating preprocessing, the dominant structure of population audiometric shape is a continuum, and hard partitions of that continuum are unstable at their boundaries (a 1 dB change can flip a categorical label).

This has a direct, practical consequence for the simulation application. The continuous parameter space that drives fluid simulation controls (attenuation and spectral-slope "sliders") is a property of the PCA embedding of row-centered audiograms, available regardless of the clustering algorithm — not a property of HDBSCAN specifically. HDBSCAN's role is to characterize that space (one dense continuum plus rare outliers), not to generate the continuity. Parameterizing an individual audiogram as a point in this PCA surface, rather than assigning it to a discrete profile, lets a simulator respond proportionally to small audiometric changes instead of jumping between fixed presets.

### 4.3 Is Cluster 1 a genuine signal or an artifact?

Cluster 1 (N=12) raises two distinct questions that we keep separate, because the evidence answers them differently: (1) Is it a technical artifact — e.g., equipment calibration error in a single NHANES cycle? (2) Is it a population-generalizable phenotype? The evidence below argues **against the artifact explanation**, but does **not** establish population generalization, which N=12 and a LOO recall of 0.75 cannot support. We therefore present Cluster 1 as an exploratory, internally robust signal, not a validated phenotype. Four lines of evidence bear on the artifact question:

**First, temporal persistence across 4 independent cycles.** The 12 individuals span NHANES cycles from 2001–2002 to 2015–2016 (SEQNs 11373, 12310, 26532, 30574, 63701, 64371, 65402, 66116, 66373, 68127, 88806, 93249). This distribution across 4 cycles with different examiners, equipment, and recruitment protocols effectively rules out single-lot bias as an explanation. If the phenotype were an artifact of one cycle's audiometer calibration, we would expect it to cluster within that cycle; instead, it appears independently across a 15-year span.

**Second, insensitivity to censoring treatment.** The H11 sensitivity analysis compared two extreme treatments of 666/no-response codes: 666→NaN (primary policy) and 666→125 dB HL (alternative policy). ARI between the two resulting clusterings was 0.9914 (including noise) and 1.0 (non-noise intersection). This means the geometric identity of Cluster 1 is preserved regardless of how we handle severe censoring — a necessary condition for any claim of phenotypic robustness. The 511 individuals affected by the 666 code were predominantly in the 2017–Mar2020 cycle (336/511), and only 2 of them fell outside the ANY25 subset, confirming that the censoring issue and the cluster discovery are largely independent.

**Third, clinical face validity.** The audiometric signature of Cluster 1 — severe sloping loss in the right ear (PTA_high_R=78.6 dB) with near-normal left ear (PTA_high_L=15.8 dB) — is a recognized clinical pattern associated with retrocochlear pathology, unilateral noise exposure, or focal cochlear damage. The tinnitus rate in Cluster 1 (50% of 8 individuals with available data) is consistent with the broader finding that auditory atypicality associates with tinnitus, and aligns with literature reporting tinnitus prevalence of 36–75% in unilateral hearing loss populations (Baguley et al., 2013). We note that the tinnitus finding is based on a small subsample (N=8) and should be interpreted as directionally supportive rather than statistically definitive.

**The direction of asymmetry deserves special attention.** All 30 individuals (12 in Cluster 1 + 18 in the outlier sub-group) show right-ear predominance. This is noteworthy because firearms-induced hearing loss — the most common cause of unilateral asymmetry in the general US population — typically affects the *left* ear in right-handed shooters (~90% of the population) due to the head-shadow effect, where the dominant-side ear is partially protected by the shoulder and stock (Cox & Ford; Chung et al.; "Shooter's Ear"). The fact that our cluster is right-ear dominant, not left-ear dominant, suggests that firearms exposure alone does not explain this phenotype. Alternative explanations include: (a) retrocochlear pathology (e.g., vestibular schwannoma), which has no lateral preference; (b) left-handed shooters' ear (minority population); or (c) focal cochlear pathology of unknown etiology. The question of *why* right-ear unilateral loss forms a dense cluster while left-ear unilateral loss does not is open and requires clinical validation with sidedness and noise-exposure history.

**Fourth, RF surrogate interpretability (with explicit caveats).** A Random Forest surrogate separated Cluster 1 from Cluster 0 using exclusively right-ear thresholds (top 7 features; left-ear importance near zero). As noted in Section 3.4, the AUC=1.0 is circular and carries no validity weight — but the *feature attribution* is informative for the artifact question: a technical artifact would be expected to lean on non-audiometric features (cycle codes, missingness flags), whereas this surrogate relied entirely on the audiometric signal. The leave-one-out recall of 0.75 (9/12) further shows the signature is mostly, but not fully, recognizable out-of-sample.

Taken together, these lines of evidence — temporal persistence, censoring insensitivity, clinical face validity, and an audiometric-only feature signature — argue that Cluster 1 is **not a technical artifact**. They do not, and with N=12 cannot, establish that it generalizes to the wider population. We therefore treat it as a reproducible internal signal with clinical plausibility, whose status as a phenotype is a hypothesis for validation in datasets with confirmed unilateral pathology.

### 4.4 Clinical Implications

The surfacing of 30 individuals with right-ear-dominant unilateral asymmetry (12 in Cluster 1 + 18 in the outlier sub-group) across 4 NHANES cycles is clinically suggestive. Unilateral hearing loss in adults warrants investigation for retrocochlear pathology (e.g., vestibular schwannoma), and recovering this pattern without supervised labels illustrates the approach's potential clinical relevance — a hypothesis to be tested, not a demonstrated clinical result.

The tinnitus–outlier association (38% vs. 18%) suggests that individuals whose audiometric patterns don't fit standard categories experience more subjective symptoms. This has implications for hearing simulation tools: modeling threshold loss alone may underestimate the lived experience.

### 4.5 OHHR: what it does and does not show

We initially read OHHR as evidence for a threshold–speech "gap" (Factor D), based on a near-zero PTA–SRT correlation. That correlation was an ingestion artifact (Section 3.6); corrected, PTA predicts the DTT SRT strongly (r=0.85). We therefore retract the Factor-D claim for this dataset: with a fixed-noise SNR measure, audibility dominates and the audiogram does predict the speech score. Probing genuine supra-threshold/central contributions would require an adaptive-SNR or cognitive measure not present in these data — a target for future work, but not something OHHR demonstrates here.

The 61.4% noise rate when projecting OHHR into the reduced 4D NHANES space (vs. 37.5% for NHANES itself in that same 4D space) is consistent with OHHR being an older, clinical population (median age 71 vs. 52 in our NHANES subset) with more severe and heterogeneous hearing loss. We stress two caveats already noted in Section 3.6: (i) this 4D space fragments into 257 micro-clusters rather than the 2-cluster main solution, so the projection does not externally validate Cluster 0 or Cluster 1; and (ii) OHHR lacks R/L separation and uses only 4 of 7 frequencies (no imputation of the missing 3000/6000/8000 Hz), so it cannot test the asymmetry finding. The external check is therefore exploratory — it probes cross-population shape overlap, not the validity of the discovered phenotypes.

### 4.6 Limitations

1. NHANES is cross-sectional; no individual temporal progression.
2. No cisplatin history in NHANES; "platinum-like" is a proxy.
3. Frequencies limited to 500–8000 Hz; ototoxicity may begin >8 kHz.
4. Tinnitus is self-reported (AUQ191); available only in 2005+ cycles.
5. Cluster 1 (N=12) is too small for population generalization. Section 4.3 argues it is not a technical artifact, but leave-one-out recall (0.75; 3/12 members borderline out-of-sample) and the 15% bootstrap non-formation rate mean it must be read as an exploratory signal, not a validated phenotype.
6. 15% bootstrap failure rate reflects small-cluster sensitivity.
7. OHHR lacks R/L separation, limiting asymmetry comparison.
8. OHHR frequencies limited to 4 of 7 (500–4000 Hz), no R/L separation, and the projection runs in a 4D space that fragments into 257 micro-clusters — so OHHR is an exploratory cross-population check, not a validation of the 14D phenotypes.
9. An earlier OHHR ingestion error (mismatched join key) produced a spurious near-zero PTA–SRT correlation (r=0.015) that we initially over-interpreted as a Factor-D "gap"; the corrected value is r=0.85. We report this openly as a caution about external-dataset ingestion.

---

## 5. Conclusion

Density-based audiometric clustering with shape-preserving preprocessing yields reproducible shape-based groupings among individuals with audiometric alteration (ANY25 subset). The key findings are:

1. Among individuals with hearing loss, the audiometric landscape is primarily a continuum, not discrete categories.
2. The method surfaced a recurring unilateral right-ear asymmetry signature — 12 individuals in a distinct cluster, plus 18 right-ear-dominant outliers — that is robust to censoring treatment and persistent across 4 cycles, and is not attributable to a technical artifact. With N=12 and LOO recall 0.75, we present it as an exploratory signal warranting clinical investigation, not a validated, generalizable phenotype.
3. Auditory "atypicality" (outlier status) is associated with 2× tinnitus rates.
4. In OHHR, pure-tone thresholds strongly predict the Digit-Triplet speech-in-noise score at fixed noise (better-ear PTA × SRT r=0.85); we found no threshold–speech dissociation in this dataset (correcting an earlier ingestion artifact that had suggested otherwise).

This methodology enables data-driven hearing simulation profiles for empathy tools, generates hypotheses for clinical validation, and provides a reproducible open-source pipeline for computational audiology research.

---

## Acknowledgments

NHANES/CDC for public data. OHHR/Hearing4all for validation data (CC BY 4.0). The open-source community for scikit-learn, HDBSCAN, and Plotly.

## Disclosure of AI Assistance

Portions of this manuscript were drafted with the assistance of large language models (Claude, Gemini). Specifically, AI assistance was used for: text formatting and language polishing, code generation for data processing scripts, and literature search organization.

All scientific decisions — including the research question, methodological choices, data analysis, interpretation of results, and conclusions — were developed and verified by the author. All numerical results reported in this paper were computed from public data using reproducible Python scripts, and the complete pipeline is available in the accompanying GitHub repository for independent verification.

The author has reviewed and approved all content in this manuscript.

---

## References

0. Cox, H. & Ford, G. (1995). Hearing loss in soldiers exposed to weapon noise. *British Journal of Audiology*. [Firearms asymmetry: left-ear predominance in right-handed shooters]

1. Parthasarathy, A. et al. (2020). Data-driven segmentation of audiometric phenotypes across a large clinical cohort. *Scientific Reports*, 10, 6754. https://doi.org/10.1038/s41598-020-63515-5

1b. Allen, P. D. & Eddins, D. A. (2010). Presbycusis phenotypes form a heterogeneous continuum when ordered by degree and configuration of hearing loss. *Hearing Research*, 264(1-2), 10-20. https://doi.org/10.1016/j.heares.2010.02.001 [Prior PCA-based evidence that audiometric phenotypes form a continuum; sub-types arise from categorical segregation of a continuous distribution.]

2. Wang, M. et al. (2021). Audiometric phenotypes of noise-induced hearing loss by data-driven cluster analysis. *Frontiers in Medicine*, 8, 662045. https://doi.org/10.3389/fmed.2021.662045

3. McInnes, L., Healy, J. (2017). Accelerated Hierarchical Density Based Clustering. *ICDM 2017*.

4. Hubert, L., Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.

5. Jafri, S. et al. (2025). The Oldenburg Hearing Health Record (OHHR). *Scientific Data*, 12, 1546. https://doi.org/10.1038/s41597-025-05884-y

6. Systematic Review (2025). Uncovering Phenotypes in Sensorineural Hearing Loss: A Systematic Review of Unsupervised ML Approaches. *PMC 12533775*.

7. Kujawa, S., Liberman, M. (2009). Adding insult to injury: cochlear nerve degeneration after "temporary" noise-induced hearing loss. *Journal of Neuroscience*, 29(45), 14077–14085.

8. Füllgrabe, C., Moore, B. (2018). The pursuit of auditory objects. *Trends in Hearing*, 22.

9. WHO (2021). World Report on Hearing. Geneva: World Health Organization.

10. Brock, P. et al. (1991). Cisplatin ototoxicity in children: a practical grading system. *Medical and Pediatric Oncology*, 19(4), 295–300.

11. Baguley, D. et al. (2013). Tinnitus in adults with hearing impairment. *Hearing Research*, 304, 1–7.

---

## Supplementary Materials

Available at https://github.com/gabrielvn/the_frequency_ml:
- 25 Python scripts (reproducible pipeline)
- Interactive dashboard (9 sections, 5 languages)
- Model Card (formal ML documentation)
- OHHR validation results
- Bootstrap stability details

---

*Draft version 4.0. Cluster 1 robustness section added (4.2). OHHR validation section expanded (2.6, 3.6, 4.4). New reference: Baguley et al. (2013). Ready for internal review before submission.*
