# The Frequency ML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)

**A reproducibility audit of unsupervised audiometric phenotyping on NHANES.**

This project asks a methodological question — *do the data-driven "subtypes" of hearing loss reported in the literature survive a change of clustering method, random seed, and model specification?* — rather than claiming to discover new phenotypes. On 7,695 NHANES adults, after isolating audiogram shape via row-centering, no internal criterion supports stable discrete subtypes; the structure is a dominant continuum with rare outliers. We **concede priority** on the continuum finding to prior work (Allen & Eddins, 2010) and recent large-cohort studies (Dimitrov 2026; Encina-Llamas 2024; Xu 2026), and contribute a multi-algorithm robustness audit plus a reproducible scaffold others can run on their own data.

> 📄 Current manuscript: [`docs/en/PAPER_DRAFT_v5_audit.md`](docs/en/PAPER_DRAFT_v5_audit.md) · context: [`MUDANCA_v5_AUDITORIA.md`](MUDANCA_v5_AUDITORIA.md)
> 🇧🇷 [Português](docs/pt/) | 🇪🇸 [Español](docs/es/) | 🇩🇪 [Deutsch](docs/de/) | 🇫🇷 [Français](docs/fr/) *(translations pending review — old framing)*

---

## What it does

1. **Ingests** 26,583 audiograms from 9 NHANES cycles (1999–2020); filters to 7,695 with audiometric alteration
2. **Extracts shape** via row-centering (orthogonal projection onto the zero-sum hyperplane — isolates configuration from severity)
3. **Audits robustness** across three algorithm families: K-means (k×seed), GMM (BIC × covariance specification), HDBSCAN
4. **Scores** with silhouette, Gap statistic, seed-to-seed ARI, and BIC — testing whether any "natural" number of clusters exists
5. **Probes** cross-population transfer onto OHHR (N=581) — exploratory, not a validation

## Key findings

| Finding | Evidence |
|---------|----------|
| Discrete subtypes are **not robust** to method/seed/specification | Silhouette ≤0.28; Gap optimal at k=2; K-means seed-unstable at k=3–5; GMM BIC interior minimum only under `full` covariance (shallow, k=4→5) |
| Structure is a dominant continuum + rare outliers | HDBSCAN: one cluster = 92.2% of sample, 7.6% noise |
| Result independently reproduced + environment pinned | Re-run on a separate machine matches to precision; `requirements-lock.txt` |
| (Process) A silent ingestion bug, found and fixed | OHHR PTA × DTT-SRT was r≈0.015 (artifact) → **r=0.85** after correcting the relational join |


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
├── scripts/                          # Python scripts (pipeline + method-comparison audit, e.g. 26_method_comparison.py)
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
| **[Audit manuscript (v5)](docs/en/PAPER_DRAFT_v5_audit.md)** | **Current paper — reproducibility audit of phenotyping** |
| [What changed & why](MUDANCA_v5_AUDITORIA.md) | The pivot from "discovery" to "audit" |
| [Verification report](outputs/VERIFICATION_REPORT.md) | Independent re-run + numeric checks |
| [Model Card](MODEL_CARD.md) | Formal ML documentation |
| [Literature Review](LITERATURA_REVIEW.md) | Reference axes + gap analysis |
| [Process Report](RELATORIO_PROCESSO_COMPLETO.md) | Documented errors across sessions |

> Note: `MODEL_CARD.md`, `LITERATURA_REVIEW.md`, the session reports, and the translations still carry the earlier "discovery" framing in places and are being aligned to the audit frame.

## How to cite

```bibtex
@software{the_frequency_ml_2026,
  author = {Gabriel Vinicius Nascimento},
  title  = {The Frequency ML: A reproducibility audit of unsupervised audiometric phenotyping on NHANES},
  year   = {2026},
  url    = {https://github.com/gabrielviniciusnascimento/the_frequency_ml}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- NHANES/CDC for public data
- OHHR/Hearing4all for the exploratory cross-population dataset (CC BY 4.0)
- Parthasarathy et al. (2020) for prior work on audiometric clustering
- The open-source community (scikit-learn, hdbscan, plotly)

## Contact

Gabriel Vinicius Nascimento — gabrielviniciusnascimento345@gmail.com
