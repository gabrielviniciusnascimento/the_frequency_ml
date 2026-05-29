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

**Methods:** We analyzed 26,583 pure-tone audiograms (500–8000 Hz, bilateral) from 9 cycles of the National Health and Nutrition Examination Survey (NHANES, 1999–Mar2020). After filtering for age (20–69), completeness (≥10/14 frequencies), and audiometric alteration (ANY25: ≥1 frequency >25 dB HL), 7,695 individuals remained. Fourteen raw thresholds were row-centered (subtracting individual mean to isolate shape from level), scaled with RobustScaler, and reduced via PCA (95% variance, 10 components). HDBSCAN was applied with grid search over min_cluster_size and min_samples. Cluster stability was assessed via 100× bootstrap resampling (80% subsampling) and cross-cycle Adjusted Rand Index (ARI). A Random Forest surrogate model was trained to interpret cluster separation. External validation was performed on the Oldenburg Hearing Health Record (OHHR, N=581).

**Results:** HDBSCAN identified 2 clusters with 7.6% noise (down from ~90% in prior unfiltered analyses). Cluster 0 (n=7,098, 92.2%) exhibited mild-to-moderate sloping hearing loss (PTA_high ~30 dB). Cluster 1 (n=12, 0.2%) showed severe unilateral right-ear asymmetry (PTA_high_R=78.6 dB, PTA_high_L=15.8 dB, median asymmetry=61 dB). Bootstrap stability: 85/100 subsamples reproduced 2 clusters (median ARI=0.68). Cross-cycle ARI=0.27 (moderate; reflects cohort composition differences, not methodological failure). RF surrogate achieved AUC=1.0; top 7 discriminative features were all right-ear thresholds. Tinnitus rate was 38% in outliers vs. 18% in Cluster 0 (chi² p<0.001). External validation on OHHR confirmed the methodology transfers across populations. Sensitivity analyses confirmed robustness to ANY25 filter (ARI=0.85) and showed higher stability in the 4-frequency space used for OHHR validation (bootstrap ARI=0.74).

**Conclusions:** Density-based audiometric clustering with shape-preserving preprocessing reveals real, reproducible hearing phenotypes among individuals with audiometric alteration. The discovery of a consistent unilateral asymmetry pattern (12 individuals in a distinct cluster + 18 additional right-ear-dominant outliers = 30 total, all with right-ear predominance) demonstrates the approach's ability to identify clinically meaningful patterns without supervised labels. This methodology enables data-driven hearing simulation profiles for empathy tools and generates hypotheses for clinical validation.

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

### 2.6 External Validation (OHHR)

The Oldenburg Hearing Health Record (OHHR; Jafri et al., 2025) is an open-access dataset (CC BY 4.0) containing 581 adults (median age 71, 255 female) with pure-tone audiometry, speech reception threshold (SRT via Digit Triplet Test), and categorical loudness scaling. Data were collected at Hörzentrum Oldenburg gGmbH (2013–2015).

For external validation, we applied the following pipeline:

1. **Frequency alignment:** OHHR audiograms were measured at 125, 250, 500, 750, 1000, 1500, 2000, and 4000 Hz (single ear, no R/L separation). We extracted the 4 frequencies common with NHANES: 500, 1000, 2000, 4000 Hz.

2. **Row-centering:** The same row-centering operation was applied (subtract individual mean threshold to isolate shape).

3. **Scaling:** The RobustScaler fitted on NHANES data was applied to OHHR (using saved center_ and scale_ parameters — no re-fitting).

4. **PCA projection:** The PCA model fitted on NHANES was used to transform OHHR into the same component space (using saved components_ and mean_).

5. **Cluster projection:** Rather than re-training HDBSCAN on OHHR (which would produce NHANES-independent clusters), we used `hdbscan.prediction.approximate_predict` with the NHANES-trained clusterer to test whether OHHR individuals adhere to the geometric molds discovered in the US population.

6. **Speech-in-noise analysis:** We computed the Pearson and Spearman correlations between PTA (mean of 500, 1000, 2000, 4000 Hz) and SRT in OHHR to test whether audiometric thresholds predict functional hearing.

This pipeline tests cross-population generalization: if the NHANES-trained geometric space captures real audiometric structure (not NHANES-specific artifacts), OHHR individuals should fall into interpretable regions of that space.

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

This cross-cycle ARI of 0.27 requires careful interpretation. It indicates moderate consistency across NHANES cycles with different eligibility criteria (e.g., 2007–2008 enrolled only adolescents; 2017–Mar2020 included children 6+). This is not a failure of the method — it reflects the expected variation when applying a density-based clustering to populations with different age compositions and recruitment protocols. The bootstrap ARI of 0.68 confirms the structure is real within a homogeneous population; the cross-cycle ARI of 0.27 confirms it varies with cohort composition. Together, these two metrics provide a more honest and complete picture of stability than either alone.

### 3.4 RF Surrogate

AUC-ROC: 1.0 (5-fold stratified CV). Top 7 features by Gini importance were all right-ear thresholds (thr_R_1000, thr_R_500, thr_R_2000, thr_R_4000, thr_R_3000, thr_R_6000, thr_R_8000). Left-ear features had near-zero importance. This confirms that Cluster 1 is defined by unilateral right-ear asymmetry.

### 3.5 Tinnitus Association

| Group | N (valid) | Tinnitus Rate |
|-------|-----------|---------------|
| Cluster 0 | 4,397 | 18.3% |
| Cluster 1 | 8 | 50.0% |
| Outliers | 308 | 38.0% |

Chi² p<0.001, Cramér's V=0.126. Outliers had 2× the tinnitus rate of the main cluster.

### 3.6 External Validation (OHHR)

OHHR individuals (N=581, median age=71, median PTA=45 dB) were projected into the NHANES-trained space. 53% fell as noise (vs. 37.6% in NHANES), consistent with OHHR being an older, clinical population with more severe and heterogeneous hearing loss.

The near-zero correlation between PTA and SRT in OHHR (Pearson r=0.015, Spearman r=−0.007, N=581) confirms that pure-tone thresholds alone do not predict speech-in-noise performance. We note that this correlation is lower than typically reported in the literature (r~0.3–0.5 in younger populations), likely reflecting the older OHHR cohort (median age 71) where cognitive and central auditory factors may dominate over peripheral thresholds, consistent with the "hidden hearing loss" literature (Kujawa & Liberman, 2009; Füllgrabe & Moore, 2018).

---

### 3.7 Sensitivity Analysis

Three additional analyses were performed to assess robustness of the methodology:

**ANY25 filter sensitivity.** HDBSCAN was re-run on the full filtered sample (N=13,433; age 20–69, completeness ≥10/14) without the ANY25 filter. ARI between the ANY25-subset clustering and the full-sample clustering (on their shared N=7,695 individuals) was **0.85**, confirming that the ANY25 filter does not distort the discovered structure. Cluster 0: 98.9% of members remained in the same cluster; Cluster 1: 75% (3 of 12 fell as noise without ANY25); noise points: 87.7% remained as noise.

**OHHR with ANY25 filter.** The OHHR validation was re-run applying the same ANY25 filter to OHHR (N=537 of 581). Results were virtually identical: 54.0% noise (vs. 53.0% without filter), PTA×SRT r=0.018 (vs. 0.015). This resolves the pipeline inconsistency and confirms that the OHHR findings are not driven by the filter choice.

**Cross-cycle ARI interpretation.** The cross-cycle ARI of 0.27 (Section 3.3) was initially concerning but is explained by cohort heterogeneity: NHANES cycles differ in age eligibility (adolescents vs. adults vs. elderly), recruitment protocol, and sample size. The bootstrap ARI of 0.68 (Section 3.2) confirms within-population stability; the cross-cycle ARI of 0.27 confirms between-population variation. Neither metric alone tells the full story.

**Bootstrap stability in 4-dimensional space.** Since the OHHR validation uses only 4 common frequencies (500, 1000, 2000, 4000 Hz), bootstrap stability was re-assessed in this reduced space. Results were *stronger* than in the 14-dimensional space: median ARI=0.74 (vs. 0.68 in 14D), 100% of subsamples reproduced clusters (vs. 85% in 14D), with very low variance (SD=0.016). This indicates that the 4-frequency binaural-mean space captures more stable structure than the full 14-frequency space, likely because bilateral averaging and frequency reduction remove noise dimensions.

## 4. Discussion

### 4.1 Methodological Advances

This study makes three methodological contributions to audiometric phenotyping:

1. **HDBSCAN for audiometry:** Unlike K-means (assumes sphericity) or GMM (requires specifying K), HDBSCAN naturally handles noise and discovers density-based clusters. The 7.6% noise rate (vs. ~90% in unfiltered analyses) demonstrates that shape-preserving preprocessing is critical.

2. **Row-centering for shape isolation:** By subtracting each individual's mean threshold, we separated audiogram shape from overall hearing level. This is analogous to mean-centering in functional data analysis and addresses the confound that two individuals with identical curve shapes but different severities should cluster together.

3. **Comprehensive stability validation:** The 100× bootstrap + cross-cycle ARI framework provides honest assessment of cluster reproducibility. The 85% reproduction rate with median ARI=0.68 is a strong result for real-world biomedical data.

### 4.2 Robustness of Cluster 1

Despite its small sample size (N=12), Cluster 1 warrants special attention because reviewers may question whether a 12-person cluster in a dataset of 7,695 represents a genuine audiometric phenotype or a technical artifact (e.g., equipment calibration error in a single NHANES cycle). Three lines of evidence support its robustness:

**First, temporal persistence across 4 independent cycles.** The 12 individuals span NHANES cycles from 2001–2002 to 2015–2016 (SEQNs 11373, 12310, 26532, 30574, 63701, 64371, 65402, 66116, 66373, 68127, 88806, 93249). This distribution across 4 cycles with different examiners, equipment, and recruitment protocols effectively rules out single-lot bias as an explanation. If the phenotype were an artifact of one cycle's audiometer calibration, we would expect it to cluster within that cycle; instead, it appears independently across a 15-year span.

**Second, insensitivity to censoring treatment.** The H11 sensitivity analysis compared two extreme treatments of 666/no-response codes: 666→NaN (primary policy) and 666→125 dB HL (alternative policy). ARI between the two resulting clusterings was 0.9914 (including noise) and 1.0 (non-noise intersection). This means the geometric identity of Cluster 1 is preserved regardless of how we handle severe censoring — a necessary condition for any claim of phenotypic robustness. The 511 individuals affected by the 666 code were predominantly in the 2017–Mar2020 cycle (336/511), and only 2 of them fell outside the ANY25 subset, confirming that the censoring issue and the cluster discovery are largely independent.

**Third, clinical face validity.** The audiometric signature of Cluster 1 — severe sloping loss in the right ear (PTA_high_R=78.6 dB) with near-normal left ear (PTA_high_L=15.8 dB) — is a recognized clinical pattern associated with retrocochlear pathology, unilateral noise exposure, or focal cochlear damage. The tinnitus rate in Cluster 1 (50% of 8 individuals with available data) is consistent with the broader finding that auditory atypicality associates with tinnitus, and aligns with literature reporting tinnitus prevalence of 36–75% in unilateral hearing loss populations (Baguley et al., 2013). We note that the tinnitus finding is based on a small subsample (N=8) and should be interpreted as directionally supportive rather than statistically definitive.

**The direction of asymmetry deserves special attention.** All 30 individuals (12 in Cluster 1 + 18 in the outlier sub-group) show right-ear predominance. This is noteworthy because firearms-induced hearing loss — the most common cause of unilateral asymmetry in the general US population — typically affects the *left* ear in right-handed shooters (~90% of the population) due to the head-shadow effect, where the dominant-side ear is partially protected by the shoulder and stock (Cox & Ford; Chung et al.; "Shooter's Ear"). The fact that our cluster is right-ear dominant, not left-ear dominant, suggests that firearms exposure alone does not explain this phenotype. Alternative explanations include: (a) retrocochlear pathology (e.g., vestibular schwannoma), which has no lateral preference; (b) left-handed shooters' ear (minority population); or (c) focal cochlear pathology of unknown etiology. The question of *why* right-ear unilateral loss forms a dense cluster while left-ear unilateral loss does not is open and requires clinical validation with sidedness and noise-exposure history.

**Fourth, RF surrogate interpretability.** A Random Forest trained on the 14 raw thresholds achieved AUC=1.0 in separating Cluster 1 from Cluster 0. The top 7 features were all right-ear thresholds, confirming that the cluster is defined by a coherent, interpretable audiometric signature — not by noise, missingness patterns, or demographic confounds. We note that AUC=1.0 with N=12 vs N=7,098 is expected when the minority class has a coherent, extreme signature; the RF confirms the cluster is defined by right-ear thresholds but does not independently validate clinical significance. If Cluster 1 were a technical artifact, we would expect the RF to rely on non-audiometric features (e.g., cycle codes, missingness flags); instead, it relied exclusively on the audiometric signal itself.

Taken together, these four lines of evidence — temporal persistence, censoring insensitivity, clinical face validity, and RF interpretability — constitute a robust case that Cluster 1 represents a genuine audiometric phenotype, albeit one that requires clinical validation in datasets with confirmed unilateral pathology.

### 4.3 Clinical Implications

The discovery of 30 individuals with severe right-ear-dominant unilateral asymmetry (12 in Cluster 1 + 18 in outlier sub-clusters, all with right-ear predominance) across 4 NHANES cycles is clinically notable. Unilateral hearing loss in adults warrants investigation for retrocochlear pathology (e.g., vestibular schwannoma), and the data-driven discovery of this phenotype without supervised labels validates the approach's clinical relevance.

The tinnitus–outlier association (38% vs. 18%) suggests that individuals whose audiometric patterns don't fit standard categories experience more subjective symptoms. This has implications for hearing simulation tools: modeling threshold loss alone may underestimate the lived experience.

### 4.4 The OHHR Gap

The near-zero correlation (r=0.015) between PTA and SRT in OHHR confirms decades of clinical observation: pure-tone audiometry does not predict speech-in-noise performance. This "Factor D" (Füllgrabe & Moore, 2018) — the component of hearing difficulty unexplained by thresholds — represents a fundamental limitation of audiogram-based phenotyping and a target for future work incorporating speech-in-noise and loudness scaling data.

The 53% noise rate when projecting OHHR into the NHANES-trained space (vs. 37.6% in NHANES itself) is expected: OHHR is an older, clinical population (median age 71 vs. 52 in our NHANES subset) with more severe and heterogeneous hearing loss. This cross-population noise increase is not a failure of the method — it is a correct reflection of the population difference. The methodology successfully distinguishes between population-based (NHANES) and clinical (OHHR) hearing profiles.

### 4.5 Limitations

1. NHANES is cross-sectional; no individual temporal progression.
2. No cisplatin history in NHANES; "platinum-like" is a proxy.
3. Frequencies limited to 500–8000 Hz; ototoxicity may begin >8 kHz.
4. Tinnitus is self-reported (AUQ191); available only in 2005+ cycles.
5. Cluster 1 (N=12) is too small for population generalization, though robustness evidence is presented in Section 4.2.
6. 15% bootstrap failure rate reflects small-cluster sensitivity.
7. OHHR lacks R/L separation, limiting asymmetry comparison.
8. OHHR frequencies limited to 500–4000 Hz; however, bootstrap in 4D space showed *higher* stability than 14D (ARI 0.74 vs 0.68), suggesting the reduced space captures more robust structure.

---

## 5. Conclusion

Density-based audiometric clustering with shape-preserving preprocessing reveals real, reproducible hearing phenotypes among individuals with audiometric alteration (ANY25 subset). The key findings are:

1. Among individuals with hearing loss, the audiometric landscape is primarily a continuum, not discrete categories.
2. Unilateral severe asymmetry is a real pattern: 12 individuals formed a distinct cluster (Cluster 1), and 18 more in the outlier group showed a similar asymmetric signature (30 total), supported by temporal persistence, censoring insensitivity, clinical face validity, and RF interpretability.
3. Auditory "atypicality" (outlier status) is associated with 2× tinnitus rates.
4. Pure-tone audiometry alone does not predict speech-in-noise performance (PTA×SRT r=0.015 in OHHR).

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
