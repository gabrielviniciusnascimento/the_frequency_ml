# The Frequency — Product v2 spec (empathy tool that underpins everything)

**Date:** 2026-06-15
**Status:** spec to review before building. Supersedes the product framing in `docs/VISION_full_circle.md` (2026-06-13) where the pre-commit audit changed the science. Companion: `docs/ARCHITECTURE_HANDOFF.md`.

> One-line thesis (unchanged, now earned): **make audible the loss that averaging hides.**

---

## 0. Where we literally are now (so the product stops looking "back there")

This is the gap the v2 must close. The repo has moved far past the current live product:

- **Science is audited and reframed.** Paper v6 (`docs/en/PAPER_DRAFT_v6_crosssystem.md`) survived an 8-task adversarial audit (`audit_01`–`audit_08`). The defensible claims are now: audiogram **shape is a continuum** (not discrete subtypes); there is a **real inter-ear contrast tail** that is **bilaterally symmetric**, robust to three null models, replicated externally (OHHR), and **erased by binaural averaging**. Grip downgraded; vision is a general-population negative control. **No lateralized-trauma claim.**
- **The current live product contradicts this.** `api/index.html` still sells "Cluster 0 / Cluster 1 / Outliers" as discrete subtypes and a "12-person right-ear cluster" — exactly the framing the author publicly withdrew (see `CHANGELOG.md`). This is the #1 thing to fix.
- **Authorship/priority track is in motion.** ORCID + Zenodo DOI (on next GitHub release) + medRxiv preprint package ready (`docs/en/preprint/`). The product should link the **paper + DOI**, not stale stats.
- **Funding map exists** (`MAPA_CARREIRA.md` + this session): GitHub Sponsors/Open Collective + a hackathon + LinkedIn now; Remarkable (disability-tech accelerator) and NLnet/NGI Zero (FOSS/accessibility, individuals eligible) next; Microsoft AI4A only with a reframe (it narrowed to generative AI). **Every one of these requires the product to be real and itself accessible.**

### What changed vs `VISION_full_circle.md` (do not repeat the old framing)
| VISION (2026-06-13) said | v2 says (post-audit) |
|---|---|
| `unilateral_right_severe` = "the 13 real cases / the cluster" | A **real severe inter-ear asymmetry** example (mean of real NHANES cases). The tail is **symmetric**, so present it as "one ear can be the affected ear" — *not* a discovered right-ear subtype, *not* "N=13 cluster." |
| Emphasis on the unilateral *finding* | Emphasis on the **demonstration**: averaging erases a real contrast. True whether or not it's lateralized. |
| (implicit) clusters/subtypes | **Continuum** everywhere: severity + shape sliders; profiles are *illustrative points on the continuum*, labelled as such. |

---

## 1. Product thesis & three honest demos

The tool lets anyone *hear* what standard summaries hide. Three demos, each tied to a defensible result and to the author's lived experience:

1. **The one nobody simulates right — true binaural unilateral.** One ear near-normal, one ear severe. On headphones the sound **collapses to one side**. This is the empathy headline and mirrors the paper's binaural-averaging result. (Honest framing: real severe asymmetry; either ear.)
2. **The continuum, not boxes.** Sliders move severity + shape along the real PCA axes — you *hear* the continuum the science describes, instead of 3 fake categories.
3. **The author's own geometry — high-frequency loss.** Cisplatin-type bilateral high-frequency loss: the speech average hides it, the high frequencies vanish. Personal, and the opposite geometry to demo 1 — together they make the thesis.

Plus the hook that turns a demo into a tool: **"hear your own / a loved one's audiogram."**

---

## 2. Architecture (client-first, single app)

```
the-frequency-app/                      (static, deployable to GitHub Pages/edge)
  index.html                            ← ONE app (retire root + api/ duplicates)
  src/
    audio/binauralLoss.(js|ts)          ← true per-ear loss graph (extract+upgrade from api/index.html)
    audio/samples.(js|ts)               ← CC0 speech/music/street clips (replace synthetic sines)
    projection/project.(js|ts)          ← client-side row-center → scale → PCA (from artifacts.json)
    sliders/continuum.(js|ts)           ← severity + PC1/PC2 → thresholds via inverse_transform
    ui/                                  ← screens, a11y, i18n (PT/EN)
  data/
    profiles.json                       ← illustrative continuum points (severe-asym, presbycusis, HF)
    artifacts.json                      ← scaler+PCA (reuse api/artifacts.json)
    pca_inverse.json                    ← from outputs/json/session4_pca_scaler_params.json
```

Principle: **100% in the browser, offline-after-load.** The API (`api/app.py`) becomes optional, for integrators only. Stack: vanilla TS + Vite, or plain HTML/JS to start (keep it lean). No heavy framework.

---

## 3. Screens / UX flow

1. **Intro** — the thesis in one sentence + the author's story (1–2 lines) + a **headphone check**.
2. **Choose what to hear** — three illustrative continuum points (labelled "example, not a category") **or** the sliders **or** "enter an audiogram."
3. **Hear it** — Play (voice / music / street / your mic), an A/B **"normal ↔ with loss"** toggle, a **visual indicator of what changed** (per-ear spectrum + which bands dropped), and a **live transcript** of the spoken sample so the experience is itself accessible.
4. **Understand** — short, honest explainer: continuum, what a simulation can/can't show, link to the **paper + Zenodo DOI**.
5. **Act/Share** — share a deep link to a specific profile; links to support the project (Sponsors/Open Collective) and to collaborate.

---

## 4. Audio model (fix the signature feature)

- **Per-ear independent chains** → `ChannelMerger(2)` (already works; keep). Left = ch0, right = ch1.
- **Honest mapping:** attenuation(f) = threshold(f) − headroom (start 1:1, **not** the current 0.8× with a −40 dB floor). Severe loss should approach inaudibility — that's the point. Q ≈ 1.4 (~1 octave).
- **Headphone detection / warning** before binaural playback (the effect is meaningless on speakers). Use available heuristics + an explicit "I'm on headphones" confirm.
- **Level calibration note** + a one-time "set a comfortable volume on normal first" step.
- **Honest disclaimer (keep & expand):** simulates *audibility*, not distortion/recruitment/tinnitus/speech-in-noise. Optional "perceptual approximation" mode clearly marked as not-measured.

---

## 5. Accessibility checklist (this is an accessibility tool — it must pass)

- [ ] Full **keyboard** operation; visible focus; logical tab order.
- [ ] **ARIA** roles/labels on all controls; state announced (playing/stopped, profile selected).
- [ ] **Captions/transcript** for every audio sample; never audio-only information.
- [ ] **Visual representation** of the audio change (per-ear band drop) so a Deaf/HoH user also "gets it."
- [ ] `prefers-reduced-motion` respected (visualizer calms/stops).
- [ ] Color contrast ≥ WCAG AA; not relying on color alone (R/L also labelled/patterned).
- [ ] Screen-reader pass (NVDA/VoiceOver); target **WCAG 2.2 AA**.
- [ ] Works without mic permission; graceful denials.

---

## 6. Data & assets it reuses

- `api/artifacts.json` (scaler + PCA + centroids) → client projection.
- `outputs/json/session4_pca_scaler_params.json` → `pca_inverse.json` for sliders.
- Real **severe inter-ear asymmetry** profile (mean of real NHANES cases; from `outputs/json/28_ipsative_check.json`) — framed honestly per §0 table.
- Illustrative presbycusis + cisplatin-type HF profiles (marked illustrative).
- Links: paper v6, Zenodo DOI (once minted), repo.

---

## 7. Definition of done (MVP = Phase 1)

> A normal-hearing visitor, on headphones, selects the severe-asymmetry example and **clearly hears the sound collapse to one ear** — the moment that mirrors the paper's headline — and a Deaf/HoH visitor gets the same point **visually + in text**. No "Cluster 0/1" anywhere. The page passes a keyboard + screen-reader smoke test and links the paper/DOI.

---

## 8. Build sequence (phased; each phase shippable)

- **Phase 1 — MVP empathy core** (highest leverage): one consolidated app; kill the cluster framing → continuum; **true binaural 1:1 + headphone detection**; **accessibility baseline** (keyboard, ARIA, transcript, visual diff, reduced-motion); refreshed honest copy + paper/DOI links. *This is what unlocks grants/showcase.*
- **Phase 2 — personalization**: "hear your own / a loved one's audiogram" (client-side projection).
- **Phase 3 — continuum sliders** (severity + PC1/PC2 via inverse_transform).
- **Phase 4 — polish**: CC0 real audio samples, PT/EN i18n, deep-link sharing, Sponsors/Collective links, deploy.

---

## 9. How it maps to money (what each funder needs to see)

- **GitHub Sponsors / Open Collective:** a real, usable tool + your story → a "Sponsor" button that isn't aspirational. Set up after Phase 1.
- **Hackathon (Hack for Humanity, summer 2026):** Phase 1 is a strong, demoable, health-accessibility entry.
- **Remarkable / Adaptation Ventures:** disabled founder + working disability-tech demo + traction = the pitch.
- **NLnet / NGI Zero:** FOSS + accessibility + individual-eligible; the 1–2 page application maps directly to this spec. (Check active funds; Commons call closed 2026-06-01.)
- **Microsoft AI4A:** only revisit with a generative component (e.g., generated narration/soundscapes) — not required for the others.
