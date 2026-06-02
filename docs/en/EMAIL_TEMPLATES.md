# Email Templates for Researchers

---

## Email 1: Parthasarathy et al. (Massachusetts Eye & Ear)

**To:** [Parthasarathy's email — find via paper or institution]  
**Subject:** Extending your NHANES audiometric phenotyping with HDBSCAN

Dear Dr. Parthasarathy,

I recently read your 2020 paper "Data-driven segmentation of audiometric phenotypes" (Scientific Reports) and found it foundational for my own work.

I have extended your approach using HDBSCAN (density-based clustering) on the same NHANES dataset, with three key differences:

1. **Row-centering** to isolate audiogram shape from level (removing the confound that two individuals with identical curve shapes but different severities should cluster together)
2. **HDBSCAN** instead of GMM (explicitly models noise points, doesn't require specifying K)
3. **100× bootstrap validation** (ARI median 0.68)

Key findings:
- 2 clusters with 7.6% noise (vs. your 6 GMM clusters)
- Discovery of 30 individuals with severe unilateral asymmetry across 4 NHANES cycles
- Exploratory cross-population projection onto OHHR (581 individuals, Germany)

The pipeline is open-source (MIT) with 25 reproducible scripts.

I would welcome the opportunity to discuss how these approaches compare and whether there is potential for collaboration.

**Paper draft:** [link to PAPER_DRAFT_v4.md]  
**Repository:** [GitHub link]  
**Dashboard:** [link to interactive dashboard]
**Live API:** https://the-frequency-api.onrender.com/api/project (Swagger: /docs)

Best regards,  
[Your Name]

---

## Email 2: Oldenburg / Hearing4all (OHHR creators)

**To:** hearing4all.de / [Jafri's email]  
**Subject:** Exploratory projection of OHHR into a NHANES-trained audiometric space

Dear Hearing4all team,

I am writing to share an exploratory analysis using the OHHR dataset (Jafri et al., 2025, Scientific Data) as a cross-population reference for audiometric shape discovery, and to ask a question about the Digit-Triplet data.

**What I did:**
- Trained HDBSCAN clustering on 7,695 NHANES audiograms (US, 1999–2020)
- Built a reduced 4-frequency (500/1k/2k/4k Hz) binaural-mean common space and projected 581 OHHR individuals into it via `approximate_predict`
- Computed the correlation between better-ear PTA and the OHHR Digit-Triplet SRT

**Key results (honest framing):**
- In the 4-frequency common space, HDBSCAN fragments into 257 micro-clusters, so this projection probes shape overlap — it is **not** an external validation of the 2-cluster phenotypes. 61.4% of OHHR projected as noise (vs. 37.5% for NHANES in the same space), consistent with an older, clinical cohort.
- Better-ear PTA × DTT-SRT correlation: **r=0.85** (Spearman 0.91). With the DTT's fixed-noise paradigm, audibility predicts the speech score strongly. (An earlier version of my analysis reported r≈0; that was an ingestion bug on my side — a mismatched join key — which I have since corrected. I mention it transparently.)

**Why I'm reaching out:**
- I would value your guidance on the correct way to align the Digit-Triplet SRT with audiometric data, and whether an adaptive-SNR or supra-threshold measure in OHHR would be more appropriate for probing threshold-independent speech difficulty.
- Opens the door for future work combining OHHR's loudness scaling with NHANES-derived shape profiles.

The full pipeline is open-source (MIT). I would be grateful for any feedback on the methodology.

**Paper draft:** [link]  
**Repository:** [GitHub link]

Best regards,  
[Your Name]

---

## Email 3: General outreach (audiology researchers)

**To:** [Researcher's email]  
**Subject:** Open-source audiometric clustering pipeline — seeking collaboration

Dear Dr. [Name],

I am reaching out because of your work in [specific area — e.g., computational audiology, hearing loss phenotyping, pediatric ototoxicity].

I have built an open-source pipeline that applies HDBSCAN clustering to 26,583 NHANES audiograms, discovering real hearing phenotypes without supervised labels. Key innovations:

- Row-centering to isolate shape from level
- 100× bootstrap validation (ARI median 0.68)
- Exploratory cross-population projection onto OHHR (581 individuals, Germany)
- Interactive dashboard with 12 sections, 5 languages

**What I am looking for:**
- Feedback on the methodology
- Potential collaboration on [specific area]
- Access to [specific dataset — e.g., CCSS, clinical ototoxicity data]

**What I offer:**
- Complete open-source pipeline (27 scripts, MIT license)
- Published paper (target: [conference name])
- Lived experience as childhood cancer survivor with cisplatin-induced ototoxicity

**Paper draft:** [link]  
**Repository:** [GitHub link]  
**Dashboard:** [link]
**Live API:** [API link] — project any audiogram into the population space

I understand your time is valuable. Even a brief comment on the approach would be greatly appreciated.

Best regards,  
[Your Name]

---

## Tips for sending

1. **Personalize each email** — mention specific papers or findings from the recipient
2. **Keep it short** — researchers get hundreds of emails
3. **Lead with the result** — "I found 30 people with unilateral asymmetry in NHANES" is more compelling than "I built a pipeline"
4. **Include links** — paper draft, GitHub, dashboard
5. **Don't ask for too much** — a brief comment is easier to give than a collaboration
6. **Follow up once** — if no response after 2 weeks, send a brief reminder
7. **Use institutional email if possible** — more likely to be read
