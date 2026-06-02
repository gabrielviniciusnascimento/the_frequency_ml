# The Frequency ML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)

**Data-driven audiometric phenotyping using HDBSCAN on NHANES data.**

An unsupervised machine learning pipeline that discovers real patterns of hearing loss in 26,583 people from the US National Health and Nutrition Examination Survey (NHANES), without imposing clinical labels.

> 🇧🇷 [Português](docs/pt/) | 🇪🇸 [Español](docs/es/) | 🇩🇪 [Deutsch](docs/de/) | 🇫🇷 [Français](docs/fr/)

---

## What it does

1. **Ingests** 26,583 audiograms from 9 NHANES cycles (1999–2020)
2. **Filters** to 7,695 individuals with audiometric alteration
3. **Extracts shape** via row-centering (removes level, preserves curve)
4. **Clusters** with HDBSCAN → 2 clusters + 585 outliers (7.6% noise)
5. **Validates** via 100× bootstrap (ARI median 0.68) and external projection onto OHHR (N=581)

## Key findings

| Finding | Evidence |
|---------|----------|
| Population hearing loss is a continuum, not discrete types | HDBSCAN found 2 clusters, not 10+ |
| 30 people have severe unilateral right-ear asymmetry | Cluster 1 (12) + outlier sub-group (18), across 4 NHANES cycles |
| Tinnitus is 2× higher in outliers than main cluster | Chi² p<0.001 |
| In OHHR, the audiogram strongly predicts the fixed-noise speech score | Better-ear PTA × DTT SRT r=0.85 |


## Web API

A live API projects any audiogram into the NHANES-trained space:

```bash
curl -X POST https://the-frequency-api.onrender.com/api/project \
  -H "Content-Type: application/json" \
  -d '{"thr_R_500":65,"thr_R_1000":70,"thr_R_2000":75,"thr_R_3000":80,"thr_R_4000":85,"thr_R_6000":90,"thr_R_8000":85,"thr_L_500":10,"thr_L_1000":10,"thr_L_2000":15,"thr_L_3000":15,"thr_L_4000":20,"thr_L_6000":20,"thr_L_8000":25}'
```

**Returns:** nearest cluster, distance, percentile, PTA summaries, PCA coordinates.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Interactive awareness screen (Web Audio simulation) |
| `/docs` | GET | Swagger UI |
| `/api/project` | POST | Project an audiogram |
| `/api/clusters` | GET | Cluster information |
| `/api/health` | GET | Health check |

**Stack:** FastAPI, NumPy, SciPy. Deployable on Render free tier. Zero database.

**Input validation:**
- Threshold range: -10 to 130 dB HL (Pydantic `ge`/`le`)
- Minimum 4 valid frequencies required
- Type enforcement (rejects strings, nulls)
- Request logging with latency

## Awareness Screen

An interactive web experience where anyone can hear what hearing loss sounds like, based on real data from 26,583 people. Features:

- 3 hearing loss profiles (mild-moderate, severe asymmetry, atypical)
- Real-time audio processing via Web Audio API
- Spectrum visualizer
- Web Share API (WhatsApp, Twitter, LinkedIn)
- Mobile-first, works offline after loading
- Honest disclaimer about filter limitations

## Quick start

```bash
pip install -r requirements.txt
python scripts/00_download_nhanes.py
python scripts/01_ingest_aux.py
python scripts/02_merge_context.py
python scripts/03_features_v1.py
python scripts/04_qa_report.py
python scripts/05_h11_sensitivity_666.py
```

Each script has checkpointing (skips if output exists). Run in order.

## Project structure

```
the_frequency_ml/
├── scripts/                          # 27 Python scripts (reproducible pipeline)
│   ├── 00_download_nhanes.py         # Download NHANES XPT files
│   ├── 01_ingest_aux.py              # Harmonize audiograms
│   ├── 02_merge_context.py           # Merge with demographics
│   ├── 03_features_v1.py             # Feature engineering (150 features)
│   ├── 04_qa_report.py               # Quality assurance report
│   ├── 05_h11_sensitivity_666.py     # Sensitivity to 666 code
│   ├── 06_model_ready.py             # Clean for modeling
│   ├── 07_pca_umap.py                # PCA + UMAP visualization
│   ├── 08_hdbscan_grid.py            # HDBSCAN grid search
│   ├── 09_cluster_profiles.py        # Cluster geometric profiles
│   ├── 10_rf_surrogate.py            # Random Forest surrogate
│   ├── 11_generate_results_md.py     # Results report V1
│   ├── 12_hdbscan_pca_grid.py        # HDBSCAN in PCA space
│   ├── 13_kmeans_baseline.py         # KMeans baseline
│   ├── 14_artifact_test.py           # Artifact testing (age/cycle/sex)
│   ├── 14b_artifact_per_cluster.py   # Per-cluster artifact test
│   ├── 15_residualize_cluster.py     # Residualization by age/sex
│   ├── 16_tinnitus_audit.py          # Tinnitus audit
│   ├── 17_generate_results_v2_md.py  # Results report V2
│   ├── 18_session4_shape_unblock.py  # ANY25 + row-centering
│   ├── 19_session5_subdivide_cluster0.py
│   ├── 20_session5_outlier_analysis.py
│   ├── 21_session5_rf_surrogate_v2.py
│   ├── 22_session5_cluster1_profile.py
│   ├── 23_session5_tinnitus_clusters.py
│   ├── 24_session5_personal_projection.py
│   └── 25_external_validation_ohhr.py
├── data/
│   ├── processed/                    # Feature matrix (CSV)
│   └── external/ohhr/                # OHHR validation data (CC BY 4.0)
├── outputs/
│   ├── json/                         # 15+ JSON results
│   └── dashboards/                   # Interactive HTML dashboards
├── docs/
│   ├── en/                           # English translations
│   ├── es/                           # Spanish
│   ├── de/                           # German
│   └── fr/                           # French
├── MODEL_CARD.md                     # Formal ML model card
├── LITERATURA_REVIEW.md              # Literature review (18 references)
├── RELATORIO_PROCESSO_COMPLETO.md    # Full process report
├── MAPA_CARREIRA.md                  # Career/funding opportunities
├── LICENSE                           # MIT
└── requirements.txt
```

## Documentation

| Document | Description |
|----------|-------------|
| [Model Card](MODEL_CARD.md) | Formal ML documentation (12 sections) |
| [Literature Review](LITERATURA_REVIEW.md) | 18 references, 5 axes, gap analysis |
| [Process Report](RELATORIO_PROCESSO_COMPLETO.md) | 10 documented errors, 5 sessions |
| [Session 4 Results](RESULTADOS_SESSAO4.md) | ANY25 + HDBSCAN results |
| [Session 5 Results](RESULTADOS_SESSAO5.md) | Subdivision + outliers + RF |

## How to cite

```bibtex
@software{the_frequency_ml_2026,
  author = {Gabriel Vinicius Nascimento},
  title = {The Frequency ML: Data-driven audiometric phenotyping using HDBSCAN on NHANES data},
  year = {2026},
  url = {https://github.com/gabrielviniciusnascimento/the_frequency_ml}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- NHANES/CDC for public data
- OHHR/Hearing4all for validation data (CC BY 4.0)
- Parthasarathy et al. (2020) for prior work on audiometric clustering
- The open-source community (scikit-learn, hdbscan, plotly)

## Contact

Gabriel Vinicius Nascimento — gabrielviniciusnascimento345@gmail.com
