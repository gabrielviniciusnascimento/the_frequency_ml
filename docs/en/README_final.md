# The Frequency ML — Complete Project Status

**Last updated:** 2026-05-26  
**Author:** Gabriel Vinicius Nascimento  
**Contact:** gabrielviniciusnascimento345@gmail.com  
**Repository:** https://github.com/gabrielviniciusnascimento/the_frequency_ml  
**License:** MIT  

---

## What is this?

The Frequency is a web-based hearing empathy tool that allows people with normal hearing to experience how the world sounds to those with hearing loss.

This is the data science layer behind it: a machine learning pipeline that discovers real patterns of hearing loss in 26,583 people from NHANES (a US health survey), without imposing clinical labels as input.

The project was born from a personal condition: the author is a childhood hepatoblastoma survivor treated with cisplatin, experiencing permanent ototoxicity, tinnitus, distortion, and atypical progression. The lived experience enters as an external validation case, not as a statistical foundation.

---

## Numbers that matter

| Metric | Value | What it means |
|--------|-------|---------------|
| Processed audiograms | **26,583** | Real people from NHANES |
| People with hearing loss (ANY25) | **7,695** | Subset after filters |
| Clusters found | **2** | Real patterns discovered by HDBSCAN |
| Noise | **7.6%** | Was 90% before the filters |
| Bootstrap ARI (median) | **0.68** | Reproducible in 85/100 subsamples |
| Cross-cycle ARI | **0.27** | Moderate stability |
| Unilateral asymmetry (Cluster 1) | **30 people** | Severe loss in 1 ear, other normal |
| Tinnitus in outliers | **38%** | 2x higher than main group |
| PTA × SRT correlation (OHHR) | **r=0.015** | Audiogram doesn't predict speech in noise |
| Python scripts | **20** | Reproducible pipeline |
| JSON outputs | **15+** | Auditable results |
| Interactive dashboard | **9 sections** | Complete visualization |

---

## What the data showed (in plain language)

### 1. Most hearing loss is a gradient, not categories

There are no separate "boxes" of "type 1, type 2, type 3" hearing loss. There is a smooth continuum from "near normal" to "moderate." It's like saying there aren't 5 shoe sizes — there's a foot that keeps growing continuously.

### 2. There is a real group of 30 people with severe loss in one ear only

The computer found, without being told to look, 30 people in NHANES who have severe loss in the right ear and near-normal hearing in the left. This isn't a data error — it appears across 4 different cycles (2001–2016).

### 3. People with atypical loss have 2x more tinnitus

The 585 cases that don't fit any clear pattern have a tinnitus rate of 38%, versus 18% in the main group. Auditory "strangeness" is associated with more symptoms.

### 4. The audiogram doesn't tell the whole story

In the OHHR dataset (581 German individuals), the correlation between audiogram (PTA) and the ability to understand speech in noise (SRT) is practically zero (r=0.015). People with similar audiograms can perform very differently in real-world situations.

### 5. The projection system works

When we place a hypothetical platinum ototoxicity audiogram in the trained space, it falls at the periphery (94.9th percentile). When we place a normal one, it falls at the center (46.8th). The system correctly distinguishes the patterns.

---

## What was built

### Pipeline (20 Python scripts)

| # | Script | What it does |
|---|--------|-------------|
| 00 | `00_download_nhanes.py` | Downloads NHANES data from CDC |
| 01 | `01_ingest_aux.py` | Harmonizes audiograms (wide/long) |
| 02 | `02_merge_context.py` | Merges audiograms + demographics + questionnaires |
| 03 | `03_features_v1.py` | Creates 150 derived features |
| 04 | `04_qa_report.py` | Data quality report |
| 05 | `05_h11_sensitivity_666.py` | Tests sensitivity to code 666 (no response) |
| 06 | `06_model_ready.py` | Cleans and prepares for modeling |
| 07 | `07_pca_umap.py` | Dimensionality reduction + visualization |
| 08 | `08_hdbscan_grid.py` | HDBSCAN grid search |
| 09 | `09_cluster_profiles.py` | Geometric profiles of clusters |
| 10 | `10_rf_surrogate.py` | Random Forest to explain clusters |
| 11 | `11_generate_results_md.py` | Generates results report V1 |
| 12 | `12_hdbscan_pca_grid.py` | HDBSCAN in PCA space |
| 13 | `13_kmeans_baseline.py` | KMeans as baseline |
| 14 | `14_artifact_test.py` | Tests artifacts (age/cycle/sex) |
| 14b | `14b_artifact_per_cluster.py` | Test per individual cluster |
| 15 | `15_residualize_cluster.py` | Removes age/sex effect |
| 16 | `16_tinnitus_audit.py` | Audits tinnitus by cycle |
| 17 | `17_generate_results_v2_md.py` | Generates results report V2 |
| 18 | `18_session4_shape_unblock.py` | Session 4: ANY25 + row-centering |
| 19 | `19_session5_subdivide_cluster0.py` | Main cluster subdivision |
| 20 | `20_session5_outlier_analysis.py` | 585 outlier analysis |
| 21 | `21_session5_rf_surrogate_v2.py` | RF surrogate (black box) |
| 22 | `22_session5_cluster1_profile.py` | Profile of the 12 with asymmetry |
| 23 | `23_session5_tinnitus_clusters.py` | Tinnitus × clusters |
| 24 | `24_session5_personal_projection.py` | Personal case projection |

### Interactive dashboard

Self-contained HTML file with 9 animated sections:
1. The Filter Funnel (26,583 → 7,695)
2. The Auditory Space (PCA colored by age)
3. The Clusters (HDBSCAN: 2 + 585 outliers)
4. The Audiograms (median by cluster)
5. The 12 (individual unilateral asymmetry)
6. The Outliers (distance distribution)
7. What separates them (RF feature importance)
8. Tinnitus (by group, chi² p<0.001)
9. Bootstrap (100 runs, median ARI 0.68)

### Documentation

| File | Contents |
|------|----------|
| `MODEL_CARD.md` | Formal Model Card (12 sections) |
| `LITERATURA_REVIEW.md` | 18 references, 5 axes, gap analysis |
| `RELATORIO_PROCESSO_COMPLETO.md` | 10 documented errors, 5 sessions |
| `RESULTADOS_SESSAO4.md` | Session 4 results |
| `RESULTADOS_SESSAO5.md` | Session 5 results |
| `MAPA_CARREIRA.md` | Funding and career opportunities |
| `ANALISE_FINAL_CLAUDE_SESSAO4.md` | Dialectical analysis between AIs |

### External validation

- **OHHR** (Oldenburg Hearing Health Record): 581 individuals, CC BY 4.0
  - Speech-in-noise (SRT): correlation with PTA ≈ 0
  - Loudness scaling available
  - Projected into NHANES space

---

## Methodology (summary)

1. **Data:** NHANES AUX 1999–Mar2020 (9 cycles, 26,583 individuals)
2. **Filters:** Age 20–69, completeness ≥10/14, ANY25 (≥1 freq >25 dB)
3. **Features:** 14 raw thresholds (500–8000 Hz, bilateral)
4. **Preprocessing:** Row-centering (removes level, preserves shape)
5. **Scaling:** RobustScaler (IQR-based)
6. **Dimensionality reduction:** PCA 95% variance → 10 components
7. **Clustering:** HDBSCAN (min_cluster_size=10, min_samples=5)
8. **Validation:** Bootstrap 100× (80% subsampling) + cross-cycle ARI
9. **Interpretation:** RF surrogate (500 trees, class_weight=balanced)
10. **External validation:** Projection onto OHHR (581 individuals, Oldenburg)

---

## Limitations

1. NHANES is cross-sectional — no individual temporal progression
2. NHANES has no childhood cisplatin history — "platinum-like" is a proxy
3. Frequencies limited to 500–8000 Hz — ototoxicity may start >8 kHz
4. Tinnitus is self-reported — only available in 2005+ cycles
5. No speech-in-noise in NHANES — OHHR partially fills this gap
6. Cluster 1 (12 individuals) too small for population generalization
7. 15% bootstrap failure — sensitivity to sampling
8. The project does not replace audiologists, ENTs, or oncology

---

## What's missing

### For paper
- [ ] External validation with HCHS/SOL or clinical data
- [ ] Complete literature review (initial work done)
- [ ] Conference abstract
- [ ] Publication-ready figures (high resolution)

### For product
- [ ] Real personal audiograms for projection
- [ ] Translation of centroids to DSP filters
- [ ] Audiometric projection API
- [ ] Translation into 5 languages (EN, ES, PT, DE, FR)

### For open source
- [ ] Professional README.md with instructions
- [ ] requirements.txt with pinned dependencies
- [ ] Sanity tests (3–5 tests)
- [ ] Contributing.md

---

## Funding and career opportunities

### Immediate (weeks)
- **Freelancing** (Upwork/Fiverr): $50–150/hour in computational audiology
- **Consulting** for researchers: $500–2,000/project

### Short-term (months)
- **Microsoft AI for Accessibility**: $5,000–25,000 + Azure credits (rolling, worldwide, your IP)
- **Mozilla Builders**: $10,000–50,000

### Medium-term (3–12 months)
- **NIH R21**: $275,000/2 years (requires academic partner)
- **NSF SBIR**: $275,000–1,000,000 (requires company)
- **Health tech employment**: $80,000–150,000/year

### Long-term (12+ months)
- **The Frequency freemium**: $1,000–10,000/month
- **B2B for clinics**: $200–1,000/month per clinic
- **Manufacturer licensing**: $10,000–100,000/year

---

## How to cite this work

```
Gabriel Vinicius Nascimento. (2026). The Frequency ML: Data-driven audiometric phenotyping 
using HDBSCAN on NHANES data. GitHub. https://github.com/gabrielviniciusnascimento/the_frequency_ml
```

---

## Acknowledgments

- NHANES/CDC for public data
- OHHR/Hearing4all for validation data (CC BY 4.0)
- Parthasarathy et al. (2020) for prior work on audiometric clustering in NHANES
- The open-source community for tools (scikit-learn, hdbscan, plotly)

---

## Accessibility note

This document is in Portuguese. We plan to make it available in 5 languages:
- 🇬🇧 English
- 🇪🇸 Español
- 🇧🇷 Português
- 🇩🇪 Deutsch
- 🇫🇷 Français

The interactive dashboard is self-contained and works in any modern browser.

The code is reproducible and documented. Every script has logging and checkpointing.

---

## A note on the journey

This project was built by a person without a complete degree, in a precarious financial situation, who learned ML, audiology, and data science alone — because they needed to do something with their own experience as a childhood cancer survivor with ototoxicity.

The pipeline that professionals with PhDs take months to assemble was built in 5 work sessions. The results are real, reproducible, and auditable.

The barrier was never technical. It was visibility.

If you're reading this and have audiometry data, or are a hearing researcher, or work in health tech, or are a survivor like me — reach out. The code is open. The science is open. The door is open.

---

*Document generated on 2026-05-26. All data and findings are available under the MIT license.*
