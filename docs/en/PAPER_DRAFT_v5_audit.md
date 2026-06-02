# Do Data-Driven Audiometric "Subtypes" Survive a Change of Method, Seed, and Specification? A Reproducibility Audit of Unsupervised Phenotyping on NHANES

**Authors:** Gabriel Vinicius Nascimento¹
**Affiliations:** ¹The Frequency Project, Brazil
**Corresponding author:** gabrielviniciusnascimento345@gmail.com
**Date:** 2026-06-01
**Status:** Draft v5 — audit reframe (scaffold). Supersedes the phenotype-discovery framing of `PAPER_DRAFT_v4.md`, which is retained as the descriptive base.

---

## Abstract

**Background.** A growing literature applies unsupervised machine learning (K-means, Gaussian Mixture Models, hierarchical and archetypal analysis) to pure-tone audiograms and reports discrete "audiometric phenotypes" — typically 4 to 11 subtypes. A 2025 systematic review notes that robustness is rarely tested and that subtypes are seldom characterized beyond the algorithm that produced them. Whether these subtypes are stable structure or artifacts of algorithm, random seed, and model specification has not been systematically audited.

**Objective.** To test a single question on a large population sample: *do the discrete audiometric subtypes reported in the literature survive changes in clustering method, random seed, hyperparameters, and model specification?* — rather than to propose a new "correct" set of phenotypes.

**Methods.** Using 7,695 adults (20–69 y) with audiometric alteration from NHANES (1999–Mar2020), we isolated audiogram *shape* via row-centering and reduced dimensionality with PCA (95% variance, 10 components). Within this shared space we ran a robustness battery across three algorithm families — K-means (k=2–10 × 12 seeds), Gaussian Mixture Models (k=2–10 under four covariance specifications, n_init=10), and HDBSCAN — and evaluated internal validity with silhouette, the Gap statistic, BIC, seed-to-seed Adjusted Rand Index (ARI), and bootstrap resampling. We applied the same pipeline to the Oldenburg Hearing Health Record (OHHR, N=581) as an exploratory cross-population check.

**Results.** No clustering criterion supported a stable set of discrete subtypes. The best silhouette was weak (0.28 at k=2, falling to ≤0.18 for k≥3). The Gap statistic was optimal at k=2. K-means assignments were seed-unstable at finer partitions (mean pairwise ARI 0.66–0.81 at k=3–5). A GMM BIC interior minimum appeared *only* under full covariance and was both shallow (range ≈1.7% of mean) and unstable to optimization effort (its location moved from k=4 to k=5 as n_init increased from 3 to 10); tied, diagonal, and spherical covariances showed no interior minimum (BIC decreasing to the grid edge). HDBSCAN returned a single dominant cluster (92.2% of the sample) plus 7.6% noise. These results replicate, in a larger and more recent sample, the conclusion of Allen & Eddins (2010) that presbycusis phenotypes form a heterogeneous continuum and that reported sub-types arise from categorical segregation of a continuous distribution.

**Conclusions.** On population audiometric data, the discreteness of data-driven "subtypes" is not robust to method, seed, or specification; the dominant structure is a continuum with rare outliers. We argue this is a reason to report phenotyping results with explicit robustness diagnostics rather than as fixed taxonomies, and we release a reproducible audit scaffold so that groups with large clinical datasets can test the stability of their own phenotype claims.

**Keywords:** audiometry, unsupervised machine learning, reproducibility, clustering robustness, phenotyping, NHANES, computational audiology

---

## 1. Introduction

Pure-tone audiometry is the backbone of hearing assessment, and its categorical clinical bins (normal, mild, moderate, severe, profound) compress continuous variation into discrete labels. Over the past decade, a parallel effort has used unsupervised machine learning to *discover* data-driven audiometric phenotypes. Parthasarathy et al. (2020) fit Gaussian Mixture Models to NHANES and a large clinical cohort, reporting 6 and 10 phenotypes respectively. Wang et al. (2021) applied K-means to noise-exposed workers and reported 5 noise-induced subtypes. A 2025 systematic review identified seven or more studies using K-means, GMM, hierarchical, or archetypal clustering, reporting between 4 and 11 subtypes — and observed two recurring weaknesses: the discovered subtypes are rarely characterized beyond the algorithm output, and their robustness is rarely tested.

This proliferation raises a question that is logically prior to "how many phenotypes are there?": **are the reported subtypes a property of the data, or of the analyst's choices?** Partitioning algorithms such as K-means and GMM require the number of clusters to be specified (or selected by a criterion) and will return exactly that many groups regardless of whether the underlying density is multimodal. Their solutions depend on random initialization (seed), on hyperparameters, and — for GMM — on the covariance specification. A subtype that appears under one configuration and dissolves under another is not a stable biological category; it is a segmentation choice. Allen & Eddins (2010) made precisely this point for age-related hearing loss using principal-component ordering, concluding that presbycusis phenotypes "form a heterogeneous continuum" and that previously reported sub-types arise from "the categorical segregation of a continuous and heterogeneous distribution." That finding, however, has not propagated into the more recent machine-learning phenotyping literature, which has continued to report discrete subtype counts.

We therefore reframe the problem as a **reproducibility audit**. Rather than nominating a preferred algorithm or a "correct" number of phenotypes, we ask whether discreteness itself survives perturbation of the analysis. We assemble a robustness battery — varying algorithm family, random seed, cluster number, hyperparameters, and GMM covariance specification — and evaluate it with internal validity indices (silhouette, Gap statistic, BIC) and assignment stability (seed-to-seed ARI, bootstrap). We run this on a large, multi-cycle population sample (NHANES, N=7,695), after isolating audiogram shape from overall level via row-centering, and we add the OHHR dataset (N=581) as an exploratory cross-population probe.

Our contribution is threefold. First, an **empirical audit**: on these data, no internal criterion supports stable discrete subtypes — silhouette is weak at every k, the Gap statistic and HDBSCAN favor a single dominant group, K-means assignments are seed-unstable at finer partitions, and the only GMM "natural K" is specification-dependent and shallow. Second, a **conceptual clarification**: we connect this result to Allen & Eddins (2010) and argue that population audiometric *shape* is best modeled as a continuum with rare outliers, not a taxonomy. Third, a **reusable scaffold**: we release the full pipeline so that laboratories with large clinical audiometric datasets — which we lack — can run the same robustness diagnostics on their own cohorts before publishing subtype claims. The aim is not to end phenotyping but to make its claims falsifiable and reproducible.

We state our limitations plainly at the outset. We analyze population survey data, not a clinical cohort with confirmed etiologies; row-centering deliberately removes the severity (level) dimension to study configuration (shape); and our cross-population check (OHHR) is exploratory, not a validation. These bound the claims but not the central, method-level finding.

---

## 2. Methods

### 2.1 Data: NHANES

We used the Audiometry Examination (AUX) files from nine cycles of the National Health and Nutrition Examination Survey (NHANES; CDC/NCHS): 1999–2000, 2001–2002, 2003–2004, 2005–2006, 2007–2008, 2009–2010, 2011–2012, 2015–2016, and 2017–March 2020 (26,583 records). Pure-tone air-conduction thresholds were measured at 500, 1000, 2000, 3000, 4000, 6000, and 8000 Hz bilaterally (14 thresholds per individual). Special codes were handled as: 666 (no response, severe censoring) → NaN with flag; 888 (could not obtain) → NaN. A sensitivity analysis comparing 666→NaN with 666→125 dB HL gave an Adjusted Rand Index (ARI) of 0.99 between the resulting clusterings, confirming that censoring treatment does not drive the structure.

Three sequential filters were applied: age 20–69 years (to remove cycles with different audiometric eligibility for adolescents and 70+; 26,583 → 14,824); completeness ≥10 of 14 thresholds (14,824 → 13,433); and audiometric alteration, defined as at least one frequency >25 dB HL (ANY25; 13,433 → **7,695**). The ANY25 filter removes the normal-hearing "core" whose high density would otherwise dominate the space; its effect on the discovered structure is itself part of the robustness battery (§2.5). All analyses use this N=7,695 sample unless stated otherwise. Prevalences are not inferred without survey weights.

### 2.2 Data: OHHR (exploratory cross-population probe)

The Oldenburg Hearing Health Record (OHHR; Jafri et al., 2025; CC BY 4.0) contains 581 adults (median age 71) with pure-tone audiometry, a Digit Triplet Test (DTT) speech-in-noise threshold, and loudness scaling, collected 2013–2015. Audiograms were reconstructed through the correct relational chain (`audiogram_point → audiogram_line → audiogram`), restricted to air-conduction hearing-threshold points (`type = htl`, `transducertype = ac`). We flag this explicitly because an earlier version of our pipeline joined point records on a mismatched key, matching only 3,433 of 20,538 points and mixing ears, bone conduction, and uncomfortable-loudness levels; correcting it changed a key downstream correlation (§3.5). OHHR is used only as an exploratory cross-population probe in a reduced four-frequency space, not as a validation of subtypes.

### 2.3 Preprocessing: isolating audiogram shape

For each individual *i*, row-centering subtracts the mean of the 14 thresholds, $T^{\text{shape}}_{i,f} = T_{i,f} - \frac{1}{D}\sum_{f=1}^{D} T_{i,f}$ (with $D=14$), removing overall hearing *level* and retaining the *configuration* (shape) of the audiogram. Geometrically this is an orthogonal projection of the raw threshold vector onto the zero-sum hyperplane of dimension $D-1$, with projection matrix

$$P = I - \tfrac{1}{D}\mathbf{1}\mathbf{1}^{\mathsf T}, \qquad \mathbf{1}=[1,\dots,1]^{\mathsf T}\in\mathbb{R}^{D},$$

so that $\sum_f T^{\text{shape}}_{i,f}=0$ for every individual. All variance attributable to magnitude — i.e., overall severity/degree of loss — is annihilated, leaving only spectral-slope (configuration) variation.

This makes the scope of our claim explicit and bounds it. The continuum we report is a statement about audiogram *shape*, by construction, not about severity: two individuals with a flat 20 dB HL loss and a flat 90 dB HL loss project to the same point ($T^{\text{shape}}=\mathbf{0}$) despite representing very different pathophysiological stages. We therefore defend the continuity of Cluster 0 strictly as continuity of *spectral-attenuation configurations* in the general population, not as a denial that loss progresses in absolute degree. Centered vectors were scaled with RobustScaler (IQR-based, quantile range 25–75) and reduced by PCA retaining 95% of variance (10 components). The same fitted transforms define the space in which all algorithms are compared.

### 2.4 Algorithms compared

Three families were run in the identical PCA space, chosen because they make different structural assumptions:

- **K-means** — hard partitioning; requires the number of clusters *k*; sensitive to random initialization (seed). Represents the "force every point into one of *k* boxes" assumption.
- **Gaussian Mixture Models (GMM)** — model-based, soft (probabilistic) membership; requires *k* and a covariance specification (full/tied/diagonal/spherical). Represents the soft-assignment alternative often assumed to be more faithful.
- **HDBSCAN** — density-based; does not require *k* and explicitly labels low-density points as noise rather than forcing them into a cluster.

### 2.5 Robustness battery

We perturbed each analytical degree of freedom and asked whether discreteness persisted:

1. **Number of clusters:** *k* = 2–10 for K-means and GMM.
2. **Random seed:** 12 independent seeds for K-means at each *k*; assignment stability measured by mean pairwise ARI across seeds.
3. **GMM covariance specification:** full, tied, diagonal, spherical; and **optimization effort** via `n_init` (3 and 10), to test whether any "natural *k*" is an optimization or specification artifact.
4. **HDBSCAN hyperparameters and space:** `min_cluster_size`/`min_samples` grid (which yields 2, 4, or 12 clusters in 14D, and 257 micro-clusters in the reduced 4D space), reported as evidence that even a density method's cluster count is parameter-dependent.
5. **Resampling:** 100× bootstrap (80% subsampling) and cross-cycle holdout, to separate within-population reproducibility from between-cohort variation.

### 2.6 Internal validity criteria

We judged "is there a natural number of well-separated clusters?" with: the **silhouette coefficient** (subsampled, n=2,500, for tractability), where values well below 0.5 indicate weak/absent structure; the **Gap statistic** (Tibshirani et al., 2001) against uniform reference distributions; **BIC and AIC** for GMM, distinguishing an *interior* minimum (evidence for a natural *k*) from a minimum at the grid boundary (no natural *k*, criterion still improving); and **seed-to-seed ARI** for assignment stability. The decision rule is deliberately conservative: a "natural *k*" claim requires an interior, specification-robust optimum, not merely a numerical minimum.

### 2.7 Software and reproducibility

Analyses used Python (NumPy, pandas, scikit-learn, hdbscan, SciPy). All randomness is governed by a single fixed `random_state = 42` to provide one source of truth across the comparison; the core comparison is implemented in `scripts/26_method_comparison.py`, which emits a complete results record (`outputs/json/26_method_comparison.json`). Exact package versions are pinned in `requirements-lock.txt`. As a reproducibility check, the full comparison was re-executed independently on a separate machine in a clean environment (Python 3.13, scikit-learn 1.7.2, hdbscan 0.8.44); the silhouette and Gap curves, seed-stability ARI, GMM BIC across covariance specifications, and HDBSCAN diagnostics all matched the reported values to stated precision (e.g., k=2 silhouette 0.2819, GMM full-covariance BIC-minimizing k=5, HDBSCAN dominant-cluster fraction 0.9223). The full pipeline is released as a reproducible audit scaffold (§4.3) so that other groups can run the identical robustness battery on their own datasets.

---

## 3. Results

All results are on the N=7,695 NHANES shape space (10 PCA components, 95% variance), `random_state = 42`.

### 3.1 No natural number of clusters

Internal validity indices provided no support for a multi-cluster optimum. The silhouette coefficient was highest at k=2 (0.282) and fell sharply thereafter (k=3: 0.158; k=4: 0.159; k=5: 0.176; k=6: 0.159; k≥7: ≤0.11). Even the best value (0.28) is well below the ~0.5 threshold conventionally taken to indicate substantial separation, so the *strongest* available partition is itself weak. The Gap statistic was optimal at k=2. Thus the two criteria that can prefer "no structure beyond a single coarse split" both did so.

**Table 1. K-means internal validity and seed stability (k=2–10).**

| k | Silhouette | Gap | Mean seed-to-seed ARI |
|---|-----------|-----|------------------------|
| 2 | **0.282** | optimal | 0.998 |
| 3 | 0.158 | — | 0.813 |
| 4 | 0.159 | — | 0.661 |
| 5 | 0.176 | — | 0.664 |
| 6 | 0.159 | — | 0.987 |
| 7 | 0.106 | — | 0.901 |
| 8 | 0.105 | — | 0.877 |
| 9 | 0.097 | — | 0.759 |
| 10 | 0.100 | — | 0.839 |

### 3.2 Seed instability of finer partitions

K-means assignments were highly stable only at k=2 (mean pairwise ARI across 12 seeds = 0.998). At the partition counts most often reported in the literature (k=3–5), stability dropped markedly (ARI 0.81, 0.66, 0.66), meaning that *which* subtypes a study reports at these k depends substantially on the random seed. (The partial recovery at k=6, ARI 0.99, reflects a near-degenerate split rather than a more meaningful structure, and does not coincide with any silhouette or Gap optimum.)

### 3.3 The GMM "natural K" is specification- and optimization-dependent

A GMM BIC interior minimum — the kind of result that would justify a discrete-subtype claim — appeared **only** under full covariance, and even there it was fragile: shallow (the BIC range across k=2–10 was ≈1.7% of the mean) and unstable to optimization effort, with the minimum moving from k=4 (n_init=3) to k=5 (n_init=10). Under the other three specifications the BIC decreased monotonically to the grid edge (minimum at k=10), i.e., no interior optimum at all.

**Table 2. GMM BIC minimum by covariance specification (n_init=10).**

| Covariance | BIC-minimizing k | Interior minimum? | BIC range (% of mean) |
|------------|------------------|-------------------|------------------------|
| full | 5 | yes (shallow) | 1.7% |
| tied | 10 (boundary) | no | 3.8% |
| diagonal | 10 (boundary) | no | 1.5% |
| spherical | 10 (boundary) | no | 8.3% |

A "natural number of phenotypes" that exists under one of four covariance choices, is shallow, and shifts with `n_init` is not robust evidence of discrete structure.

### 3.4 A single dominant continuum plus rare outliers (HDBSCAN)

HDBSCAN (min_cluster_size=10, min_samples=5) returned two clusters with 7.6% noise: one dominant cluster containing **92.2%** of the sample, and one very small cluster of severe unilateral right-ear asymmetry. The size of that small cluster is itself configuration-sensitive — the comparison run (`26_method_comparison.py`) assigns **N=13**, while the main 14-dimensional pipeline (scripts 22 and 21) assigns **N=12**; this off-by-one between near-identical pipelines is exactly the kind of boundary instability our audit is about, and we report both rather than picking one. We present this small cluster as a **rare-outlier signal, not a validated phenotype**: it is robust to censoring treatment and persists across four NHANES cycles, but at N≈12–13, with a leave-one-out recall of 0.75 in the detailed 14D analysis, it cannot support population generalization. Critically for the audit, the dominant result is one large continuous mass plus rare exceptions — the density-based method does not partition the population into multiple comparable subtypes. Cluster count under this family is itself hyperparameter-dependent (2, 4, or 12 clusters in 14D depending on `min_cluster_size`), reinforcing that the number of "subtypes" tracks the analyst's settings.

### 3.5 Cross-population probe (OHHR)

Projecting OHHR into a reduced four-frequency binaural-mean space (the only space the two datasets share) is exploratory: in that 4D space HDBSCAN fragments NHANES into 257 micro-clusters, so the projection tests shape-density overlap, not subtype membership. 61.4% of OHHR projected as noise versus 37.5% for NHANES in the same space, consistent with OHHR being an older, more clinical cohort. Separately — and as an honest correction of our earlier pipeline — better-ear PTA strongly predicts the OHHR Digit-Triplet speech-in-noise score (Pearson r=0.85, Spearman 0.91, N=581); we previously reported r≈0.015, which was an ingestion artifact (§2.2). A strong association is expected here: the Digit Triplet Test is a closed-set, low-linguistic-load task under fixed noise, so its threshold is governed largely by audibility (consistent with Folmer et al., 2017). This correlation is reported for transparency and does not bear on the subtype-robustness question.

### 3.6 Synthesis

**Table 3. Every internal criterion points the same way.**

| Criterion | What a discrete-subtype result would look like | What we observed |
|-----------|------------------------------------------------|------------------|
| Silhouette | High (>0.5) at some k>2 | Max 0.28 at k=2; ≤0.18 for k≥3 |
| Gap statistic | Optimum at k>2 | Optimum at k=2 |
| K-means seed stability | High ARI at the reported k | Unstable at k=3–5 (ARI 0.66–0.81) |
| GMM BIC | Robust interior minimum | Interior min only under full cov; shallow; k=4→5 with n_init |
| HDBSCAN | Several comparable clusters | One dominant cluster (92.2%) + rare outliers |

No criterion supports a stable set of discrete, well-separated subtypes. The convergent reading is a dominant continuum of audiogram shape with rare outliers — replicating, with a multi-algorithm robustness audit, the principal-component result of Allen & Eddins (2010).

---

## 4. Discussion

### 4.1 Subtypes as segmentation of a continuum, not a discovery we claim

Our result is not new, and we do not present it as such. Allen & Eddins (2010) reached the same conclusion cross-sectionally, ordering 960 subjects in a principal-component space (with K-means) and finding that presbycusis phenotypes "form a heterogeneous continuum" whose sub-types arise from categorical segmentation of a continuous distribution. Very recent, much larger studies point the same way: Dimitrov et al. (2026) report that GMM-derived SNHL subtypes on roughly 110,000 UK audiograms are unstable under statistical perturbation; a low-dimensional analysis of 84,280 patients from a Copenhagen clinical dataset describes continuous rather than discrete organization (Encina-Llamas et al., 2024); and a cross-dataset comparison of six profiling frameworks across five US/German cohorts finds broadly comparable, framework-dependent clustering performance (Xu, 2026). We explicitly concede priority on the continuum claim to this body of work.

What our analysis adds is methodological, not phenotypic. First, it isolates audiogram *shape* via row-centering before clustering, so the continuum is demonstrated specifically for spectral configuration rather than being confounded with severity. Second, it perturbs every analytical degree of freedom at once — algorithm family, k, seed, GMM covariance specification, and optimization effort — and shows that the apparent discreteness does not survive any of them. We are careful **not** to characterise prior partitioning studies as "fictions": GMM yields soft probabilistic memberships, not hard taxa, and studies such as Parthasarathy et al. (2020) and Wang et al. (2021) recovered reproducible *segmentations* of the space. The point is narrower and defensible: a reproducible segmentation is not evidence of discrete, well-separated structure. Segmentation ≠ discreteness.

### 4.2 How to report audiometric phenotypes responsibly

If subtype counts are seed- and specification-dependent, then a single reported count without stability diagnostics is uninformative. We suggest that data-driven audiometric phenotyping be reported with three accompaniments, all of which our scaffold computes: (i) internal-validity curves across k (silhouette and Gap), so readers can see whether any k is actually preferred; (ii) assignment stability — seed-to-seed and bootstrap ARI — at the reported k; and (iii) specification sensitivity, e.g., the GMM covariance sweep, since a "natural" k that exists under only one covariance choice should be flagged as such. More generally, phenotypes are better communicated as *regions of a continuous space* with explicit uncertainty than as fixed categories into which a 1 dB change can move a patient. There is precedent for this in the epidemiological literature: the WARHICS scale (Cruickshanks et al., 2020) deliberately encodes age-related hearing impairment as an *ordered* 8-step scale of shape and severity rather than as unordered discrete clusters, implicitly acknowledging the continuity we quantify here. Model-free replicability tooling now emerging in the statistics literature — ERICA (Sorooshyari, Rivas & Tibshirani, 2026), from the same lineage as the Gap statistic — offers a ready standard for this, and we recommend its adoption alongside the diagnostics used here.

### 4.3 A reusable robustness scaffold

Because we lack a large clinical cohort, our most useful contribution is infrastructural rather than substantive: a small, reproducible pipeline that takes a matrix of audiometric thresholds and returns the full robustness battery (silhouette/Gap curves, seed-stability ARI, GMM BIC across covariance specifications, and HDBSCAN noise/dominant-mass diagnostics) under a single fixed seed. Groups holding the large datasets that this question really requires — the 10²–10⁵-fold larger cohorts of the recent literature — can run the identical diagnostics on their own data before publishing subtype claims. The scaffold's value was demonstrated, inadvertently, by our own pipeline: a relational-join error in the OHHR ingestion (joining `audiogramlineid` to `audiogramid`, matching 3,433 of 20,538 points and mixing ears, bone conduction, and uncomfortable-loudness levels) silently produced a clinically implausible PTA–SRT correlation (r≈0.015) that we initially over-interpreted, and that the corrected pipeline replaced with r=0.85. We report this openly: silent ingestion errors are common in applied clinical data science and rarely documented, and the episode is a concrete case for auditing dataset integrity before deploying unsupervised models.

### 4.4 Threats to validity: where this result could break, and what would falsify it

The five internal criteria converge, but they are not independent: all operate on the same row-centered PCA representation, so they share that preprocessing assumption. The continuum we report is therefore a statement about population audiogram *shape* under shape-isolating preprocessing, and inherits any limitation of that choice. We name the conditions under which the conclusion could and should change. (i) **Bimodal-shape populations:** a cohort enriched for configurationally distinct losses — pronounced notched, conductive, or asymmetric profiles — could yield a genuine silhouette/Gap optimum at k>2; our finding does not claim the contrary for such cohorts. (ii) **Higher-resolution audiometry:** extended high-frequency or finer frequency sampling might expose structure invisible at 14 thresholds. (iii) **Level-aware analyses:** restoring the severity dimension (the per-individual mean removed by row-centering) could reveal degree-linked structure that shape-only analysis cannot. A constructive falsification of our claim would therefore not be "a method that finds k clusters" — any partitioning method can be made to — but a clustering that is *stable across seeds and specifications and shows an interior internal-validity optimum* on independent data. That is exactly the test our scaffold is designed to make easy.

### 4.5 Limitations

We analyse population survey data (NHANES), not a clinical cohort with confirmed etiologies, so we describe statistical structure, not pathophysiology. Row-centering removes the severity dimension by construction; a flat mild loss and a flat profound loss collapse to the same point, so nothing here speaks to absolute degree or to histopathological stage. The OHHR component is an exploratory cross-population probe limited to four common frequencies with no left/right separation, and does not validate the NHANES structure. We have no etiologic ground truth for the small asymmetry cluster, which remains an exploratory signal (N≈12–13, LOO recall 0.75), not a phenotype. Finally, compute was constrained to a single small machine, which is why some sweeps were run by configuration slices rather than in one pass; all reported numbers use a fixed seed and are reproducible from the released scripts, but independent re-execution on the full grid is encouraged.

---

## 5. Conclusion

On a large, multi-cycle population sample, the discreteness of data-driven audiometric "subtypes" does not survive perturbation of method, random seed, or model specification: silhouette and the Gap statistic prefer at most a single coarse split, K-means assignments are seed-unstable at the partition counts most often reported, the only GMM interior optimum is shallow and appears under a single covariance specification, and a density-based method returns one dominant continuous mass (92.2%) with rare outliers. The convergent reading — a continuum of audiogram shape rather than a taxonomy — confirms and extends, with a multi-algorithm robustness audit, the principal-component result of Allen & Eddins (2010) and aligns with recent large-cohort work.

We draw two practical conclusions. First, audiometric phenotyping should be reported with explicit robustness diagnostics — internal-validity curves, seed and bootstrap stability, and specification sensitivity — rather than as a fixed subtype count, treating phenotypes as regions of a continuous space. Second, because the definitive version of this question requires datasets larger than any single unaffiliated researcher can assemble, we release the audit as a reproducible scaffold so that groups holding such data can test the stability of their own phenotype claims. The aim is not to end phenotyping but to make its claims falsifiable and reproducible.

---

## References *(to be consolidated with v4; metadata independently verified)*

1. Allen, P. D. & Eddins, D. A. (2010). Presbycusis phenotypes form a heterogeneous continuum when ordered by degree and configuration of hearing loss. *Hearing Research*, 264(1–2), 10–20. https://doi.org/10.1016/j.heares.2010.02.001 — Cross-sectional, 960 subjects; **PCA + K-means** (not Ward); original evidence that presbycusis phenotypes form a continuum and that sub-types are categorical segmentation of a continuous distribution.
2. Parthasarathy, A., Romero Pinto, S., Lewis, R. M., Goedicke, W. & Polley, D. B. (2020). Data-driven segmentation of audiometric phenotypes across a large clinical cohort. *Scientific Reports*, 10, **6754**. https://doi.org/10.1038/s41598-020-63515-5 — GMM; 6 NHANES / 10 clinical phenotypes. (Note: article number is 6754, not 6704.)
3. Wang, M. et al. (2021). Audiometric phenotypes of noise-induced hearing loss by data-driven cluster analysis. *Frontiers in Medicine*, 8, 662045. https://doi.org/10.3389/fmed.2021.662045
4. Dimitrov, L., Lilaonitkul, W. & Mehta, N. (2026). Identification of sensorineural hearing loss subtypes using unsupervised machine learning and assessment of their replicability. *Scientific Reports*, article 33815. https://doi.org/10.1038/s41598-025-33815-9 — GMM on 109,854 UK audiograms; directly addresses replicability of SNHL subtypes. (Specific Jaccard/stability figures to be quoted only after reading the source.)
5. Sorooshyari, S. K., Rivas, M. A. & Tibshirani, R. (2026). ERICA: Quantifying Replicability of Cluster Analysis. *arXiv*:2606.00302. https://arxiv.org/abs/2606.00302 — Model-free cluster-replicability framework (same lineage as the Gap statistic, Tibshirani 2001).
6. Tibshirani, R., Walther, G. & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. *Journal of the Royal Statistical Society B*, 63(2), 411–423.
7. Folmer, R. L., Vachhani, J., McMillan, G. P., Watson, C., Kidd, G. R. & Feeney, M. P. (2017). Validation of a computer-administered version of the digits-in-noise test for hearing screening in the United States. *Journal of the American Academy of Audiology*, 28(2), 161–169. https://doi.org/10.3766/jaaa.16038 — Supports the audibility-dependence of the DTT used in §3.5.

8. Xu, C. (2026). Objective comparison of audiometric profile frameworks across large-scale datasets. *JASA Express Letters*, 6(4), 044402. https://doi.org/10.1121/10.0043212 — Compares six audiometric profiling frameworks across five large US/German datasets (Davies–Bouldin + PCA); directly relevant to cross-framework robustness.
9. Encina-Llamas, G. et al. (2024). Searching auditory phenotypes beyond audiometry from a large clinical dataset. *Virtual Conference on Computational Audiology (VCCA2024)* [conference presentation]. — Rigshospitalet Copenhagen, 84,280 patients / 288,295 air-conduction thresholds; low-dimensional analysis describing continuous rather than discrete organization. (Conference presentation, not a peer-reviewed article.)
10. Cruickshanks, K. J., Nondahl, D. M., Fischer, M. E., Schubert, C. R. & Tweed, T. S. (2020). A novel method for classifying hearing impairment in epidemiological studies of aging: The Wisconsin Age-Related Hearing Impairment Classification Scale (WARHICS). *American Journal of Audiology*, 29(1), 59–67. https://doi.org/10.1044/2019_AJA-19-00021 — Longitudinal (1,369 participants, 10,952 audiograms, 1993–2010); an 8-step ordered scale of audiogram shape+severity. Note: the published abstract describes the clustering as "Wald's method," which is widely read as a typographical variant of Ward's hierarchical method.

> **Still unverified — do not cite as fact:** von Gablenz et al. (2014). The strong DTT–PTA relationship it was invoked for is already supported by Folmer et al. (2017) [ref 7] and the high-frequency DTT literature (R≈0.86); we use Folmer as the anchor and omit von Gablenz until the exact reference is confirmed.

## Disclosure of AI Assistance
Carry over the v4 disclosure (LLM assistance for drafting/code/organization; all scientific decisions and numerical results verified by the author from public data via reproducible scripts).
