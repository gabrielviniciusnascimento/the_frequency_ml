# Microsoft AI for Accessibility — Grant Application Draft

**Project:** The Frequency ML  
**Applicant:** Gabriel Vinicius Nascimento  
**Email:** gabrielviniciusnascimento345@gmail.com  
**Country:** Brazil  
**Date:** 2026-05-26  

---

## 1. The Accessibility Challenge

Over 1.5 billion people live with hearing loss worldwide (WHO, 2021). Yet hearing empathy tools use generic presets — "mild," "moderate," "severe" — that don't reflect the real diversity of auditory experiences. People with normal hearing cannot truly understand what it means to live with hearing loss, tinnitus, distortion, or difficulty in noise.

This gap has real consequences: family members don't understand why their loved ones struggle in restaurants. Teachers don't know why a student with "normal" audiograms can't follow in a noisy classroom. Employers underestimate the impact of hearing loss on workplace performance.

## 2. The Technical Solution

The Frequency ML is an open-source pipeline that:

1. **Processes 26,583 real audiograms** from the US NHANES survey (9 cycles, 1999–2020)
2. **Discovers real hearing patterns** using HDBSCAN (density-based clustering) — not presets
3. **Projects any audiogram** into the population-trained space — see where you fall among 26,000 people
4. **Generates data-driven simulation profiles** for The Frequency hearing empathy tool

### Key technical innovations:
- **Row-centering** isolates audiogram shape from level (to our knowledge, first documented application in unsupervised audiometric clustering on population data)
- **HDBSCAN** explicitly models noise (90% → 7.6% after preprocessing)
- **100× bootstrap validation** (ARI median 0.68)
- **External validation** on OHHR dataset (581 people, Germany, CC BY 4.0)
- **Robustness confirmed:** ANY25 filter sensitivity ARI=0.85; 4D bootstrap ARI=0.74 with 100% cluster reproduction
- **27 reproducible Python scripts** with checkpointing

## 3. Impact on People with Disabilities

### Direct impact:
- **Hearing empathy simulation** based on real data, not guesses
- **Personal audiogram projection** — individuals can see where they fall in the population
- **Data-driven profiles** for hearing aid fitting, rehabilitation, and counseling

### Indirect impact:
- **Education:** Teachers can experience their students' hearing loss
- **Employment:** Employers can understand workplace accommodations
- **Research:** Open-source pipeline enables further accessibility research
- **Advocacy:** Data-backed evidence for hearing loss impact

## 4. Data

- **NHANES:** 26,583 audiograms, publicly available, no restricted access
- **OHHR:** 581 audiograms + speech-in-noise, CC BY 4.0
- **All data is privacy-compliant** (de-identified, publicly available)

## 5. Feasibility (12-month plan)

| Quarter | Deliverable |
|---------|-------------|
| Q1 | Open-source pipeline on GitHub + published paper |
| Q2 | Personal audiogram projection API | ✅ **Done** — FastAPI endpoint + awareness screen |
| Q3 | The Frequency web app with data-driven profiles | In progress — awareness screen live, DSP profiles next |
| Q4 | Multilingual deployment (5 languages) + accessibility audit |

## 6. Team Capability

- **Technical:** Complete ML pipeline already built (27 scripts, 15+ outputs, interactive dashboard)
- **Domain:** Lived experience as childhood cancer survivor with cisplatin-induced ototoxicity
- **Open source:** MIT licensed, fully reproducible
- **Documentation:** Model Card, Literature Review, Process Report

## 7. Sustainability

- **Open source** ensures community maintenance
- **Public data** ensures reproducibility
- **Modular design** allows incremental improvement
- **The Frequency web app** provides ongoing user base

## 8. Budget

| Item | Cost |
|------|------|
| Azure compute (data processing, ML training) | $2,000 |
| Web hosting (The Frequency app) | $1,200 |
| Conference presentations (2 conferences) | $3,000 |
| OHHR/HCHS-SOL data acquisition | $0 (public) |
| **Total** | **$6,200** |

## 9. IP Statement

All IP remains with the applicant. The pipeline is MIT licensed. The Frequency web app is the applicant's product.

## 10. Showcase

- Interactive dashboard with 12 sections, 5 languages
- Published paper (target: Interspeech or AAA conference)
- GitHub repository with 27 reproducible scripts
- Personal audiogram projection demo

---

*This project was built by a self-taught developer without formal education in audiology or ML, driven by personal experience with hearing loss. The barrier was never technical — it was visibility. This grant would provide both.*
