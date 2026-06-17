# The Frequency ML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)

**A null-calibrated audit that tells a real inter-side asymmetry from artifact in paired-organ measurements (ears, hands, eyes).**

Many clinical measurements are made on paired organs, and each yields a difference between the two sides. This project asks a measure-agnostic question — *does a paired-organ measure carry a real non-Gaussian tail of inter-side asymmetry, beyond what its own marginals, correlation, and measurement noise can produce?* — and answers it with a reusable 4-step audit: anatomical-pairing check, sum/difference decomposition, a **Monte-Carlo null envelope** (empirical *p*), and an internal **negative control**. Applied to three NHANES paired systems with identical code, it finds a real far-tail of inter-ear asymmetry in **audiometry** that is **bilaterally symmetric**, robust to Gaussian, heteroscedastic-measurement, and tail-dependent (t-copula) nulls, **replicated externally** (OHHR), and **erased by binaural averaging** — while a general-population **visual** control shows no excess and **grip** shows excess only over a Gaussian null. We make no directional or lateralized-trauma claim; the contribution is the method and the auditory finding. We concede the audiogram-shape *continuum* to prior work (Allen & Eddins 2010; Dimitrov 2026; Encina-Llamas 2024).

> 📄 Current manuscript: [`docs/en/PAPER_DRAFT_v6_crosssystem.md`](docs/en/PAPER_DRAFT_v6_crosssystem.md) · the pre-commit audit that hardened it: [`HANDOFF_PRECOMMIT_AUDIT.md`](HANDOFF_PRECOMMIT_AUDIT.md) (scripts `audit_01`–`audit_08`)
> Motivated by the author's lived experience as a childhood cisplatin ototoxicity survivor: *making audible the loss that averaging hides.*
> 🇧🇷 [Português](docs/pt/) | 🇪🇸 [Español](docs/es/) | 🇩🇪 [Deutsch](docs/de/) | 🇫🇷 [Français](docs/fr/) *(translations pending review — old framing)*

---

## What it does

1. **Verifies** the anatomical pairing of the two channels from source docs (e.g. grip side via `MGATHAND`, not test order)
2. **Decomposes** each bilateral pair into inter-side *sum* (level) and *difference* (contrast); a real unilateral signal must live in the difference subspace
3. **Calibrates** the contrast tail against a **Monte-Carlo null envelope** (B = 2,000 copula regenerations → empirical *p*), plus heteroscedastic-measurement and tail-dependent (t-copula) nulls
4. **Controls** with the general-population visual system (no real excess expected) as an internal negative control
5. **Sanity-checks** every extreme case against the raw measurements, and replicates the auditory finding on a second cohort (OHHR)

## Key findings

| Finding | Evidence |
|---------|----------|
| Real far-tail of inter-ear asymmetry in audiometry | \|z\|>4: 92 real vs null mean 2.5 (95% CI [0,6]), *p* = 5×10⁻⁴ (Bonferroni-sig); `audit_01` |
| Robust — not selection, not measurement noise, not tail dependence | survives unfiltered inclusion (`audit_02`), a heteroscedastic null (`audit_04`), and a t-copula (`audit_05`), all at \|z\|≥3 |
| **Bilaterally symmetric** (no lateralization) | >50 dB tail: 38 right-worse / 31 left-worse, binomial *p* = 0.47; replicated in OHHR (`audit_03`, `audit_07`) |
| **Binaural averaging erases it** | the extreme-contrast group collapses to noise when the two ears are averaged before clustering (`27`) |
| Grip excess is Gaussian-only; vision is a clean negative control | grip fails the t-copula; vision within/below its null in the general population (`audit_05`, `audit_02`) |
| Built with an adversarial self-audit | 8-task pre-commit audit walked back an earlier lateralized-trauma claim (`HANDOFF_PRECOMMIT_AUDIT.md`) |


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
├── LICENSE                           # MIT
└── requirements.txt
```

## Documentation

| Document | Description |
|----------|-------------|
| **[Cross-system manuscript (v6)](docs/en/PAPER_DRAFT_v6_crosssystem.md)** | **Current paper — null-calibrated paired-organ asymmetry audit** |
| [Pre-commit audit (8 tasks)](HANDOFF_PRECOMMIT_AUDIT.md) | Adversarial self-audit (`audit_01`–`audit_08`) that hardened the v6 claims |
| [Audit manuscript (v5)](docs/en/PAPER_DRAFT_v5_audit.md) | Prior single-system reproducibility audit (history) |
| [What changed & why](MUDANCA_v5_AUDITORIA.md) | The pivot from "discovery" to "audit" |
| [Verification report](outputs/VERIFICATION_REPORT.md) | Independent re-run + numeric checks |
| [Model Card](MODEL_CARD.md) | Formal ML documentation |
| [Literature Review](LITERATURA_REVIEW.md) | Reference axes + gap analysis |
| [Process Report](RELATORIO_PROCESSO_COMPLETO.md) | Documented errors across sessions |

> Note: `MODEL_CARD.md`, `LITERATURA_REVIEW.md`, the session reports, and the translations still carry the earlier "discovery" framing in places and are being aligned to the audit frame.

## How to cite

See [`CITATION.cff`](CITATION.cff) (a Zenodo DOI will be minted on the first GitHub release).

```bibtex
@software{the_frequency_ml_2026,
  author = {Gabriel Vinicius Nascimento},
  title  = {The Frequency ML: a null-calibrated audit of interaural asymmetry in paired-organ measurements},
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
