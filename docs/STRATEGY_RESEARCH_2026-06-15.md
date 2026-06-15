# Strategy research — direction for The Frequency (paper + product + you)

**Date:** 2026-06-15 · **Prepared overnight, to set direction before the next session.**
Companion to `docs/PRODUCT_v2_SPEC.md`, `MAPA_CARREIRA.md`, `meta_analysis.md`.

> Each section = **finding → what it means for you → action**. Sources at the end. Treat grant
> dates as "verify before relying" — programs move.

---

## EXECUTIVE SUMMARY

**The one-line direction:** lead with the *work and competence* (rigorous, open, self-audited science + a tool nobody builds right), with your lived experience as **authentic motivation and authority — not a pity story.** Stack income in layers: freelance + sponsors now, grants/accelerators next, an optional paid layer later. The product is the key that unlocks all of it, and the timing is unusually good.

Seven things the research changed or sharpened:

1. **Your story is an asset — but framed as *achievement*, not *tragedy*.** "Inspiration porn" is a documented, researched failure mode that triggers *pity* and actually lowers trust/support. Achievement-framed narratives trigger *admiration* and raise it. Rule from the disability community: **"nothing about us without us"** — and you ARE the user. So: you built rigorous science and a real tool; the condition is why you're the right person, not a plea. (Your drafted LinkedIn post already leans this way — keep tilting it toward competence.)
2. **The product has a real, defensible niche.** Phonak, Starkey, NIOSH all have hearing-loss simulators — but they're **manufacturer lead-gen, closed, preset, speaker-based, and not themselves accessible.** None do **true binaural *unilateral* from real population data, open-source, and WCAG-accessible**. That gap is your wedge.
3. **The timing is rare.** **World Hearing Day 2026 theme = "hearing care for every child."** PanCare runs an **ototoxicity** working group; **45–60% of cisplatin-treated childhood-cancer survivors develop permanent hearing loss, ~half needing aids.** Your exact lived experience sits dead-center of a live global priority.
4. **Open-source donations alone won't pay rent.** <12% of OSS devs earn anything; ~$1k/month is the realistic ceiling and only with **2+ methods**. So money = layered, not a single Sponsors button.
5. **Fastest cash is freelance + a hackathon; grants are the catalytic middle.** Remote health-data / computational-audiology consulting demand is real. Accelerators (Remarkable, Fast Forward $25k+) and FOSS/accessibility grants (NLnet, WHO small grants, Grand Challenges) fund the work — but all want a *working, accessible* demo first.
6. **You can ship fast with AI — but a11y must be hand-tested.** Automated accessibility tools catch only 30–40%; screen-reader/keyboard testing is manual. Build fast, verify a11y by hand.
7. **Distribution is the moat.** On LinkedIn 2026, personal profiles get ~8× company-page engagement, inbound converts ~8× outbound, and the win is being "remembered for something specific." **Build in public** around one specific identity: *open, rigorous, accessible computational audiology, by someone who lives it.*

**Recommended direction (the sequence):**
- **Now (free, you-only):** finish authorship (ORCID → Zenodo DOI → medRxiv); post the competence-led LinkedIn; set up GitHub Sponsors + Open Collective.
- **Now (build):** ship **Product v2 Phase 1** (true binaural + continuum + accessible) — in public.
- **Weeks:** land **one freelance gig** for cash; enter **one hackathon**; do **World Hearing Day / PanCare** outreach.
- **1–3 months:** apply to **Remarkable / Fast Forward / NLnet / WHO small grants**, anchored by the working MVP + preprint + DOI.

---

## 1. Your narrative — how to use the lived experience without "inspiration porn"

**Finding.** Peer-reviewed work (J. Business Ethics, 2025; UMich) shows *inspiration-porn* framing — disability-as-tragedy-overcome — elicits **pity** and *reduces* brand authenticity and support, while **achievement-focused** framing elicits **admiration** and increases it. Disability-journalism guidance: center the person's ideas and competence; **"nothing about us without us."**

**What it means for you.** Your strongest, most honest pitch is *not* "cancer survivor who lost his hearing built an app." It's "**a self-taught researcher built rigorous, open, self-audited science and a tool the industry doesn't — and he happens to be exactly the user it's for.**" The condition is your *authority and motivation*, mentioned once, plainly — not the hook you sell on.

**Action.** Keep the lived-experience line in everything (it's real authority), but lead every artifact (LinkedIn, grant, pitch) with the *work* and the *gap you fill*. Avoid pity verbs ("suffering", "battle", "despite"). Use competence verbs ("built", "audited", "withdrew a claim", "open-sourced").

## 2. The product landscape — who exists, and your wedge

**Finding.** Established simulators: **Phonak** (comprehensive, preset audiograms, education), **Starkey** (speech-in-noise, record-your-voice), **NIOSH/CDC** (quiet vs noise), plus the classic "Unfair Hearing Test." They are mostly **hearing-aid-manufacturer marketing**, **closed-source**, **preset**, and **not designed for headphones/true-binaural unilateral**, and they are **not themselves accessible**.

**What it means for you.** You don't need to out-feature Phonak. Your differentiators are a coherent, ownable niche:
- **Open-source & free** (vs closed manufacturer tools).
- **Data-driven** from a public population dataset (NHANES), with a **peer-style audited** method behind it.
- **True binaural *unilateral*** — the "sound collapses to one ear" moment **nobody does right**.
- **The tool is itself accessible** (WCAG) — rare, and the point for an accessibility tool.
- **Authored by someone who lives it** — credibility manufacturers can't buy.

**Action.** Position v2 explicitly against this gap in the README/landing: "the open, accessible, true-binaural alternative to manufacturer demos." Don't hide the competitors — contrast with them.

## 3. The market & the money model

**Finding.** Assistive-technology market ≈ **US$30.5B (2026)**; digital-accessibility software ~US$0.9→1.9B. Buyers: individuals/caregivers, healthcare systems (insurance-funded), **hospitals (fastest-growing)**, and **corporate DEI/disability-awareness training** (simulation-based empathy is an established method). Dominant OSS monetization = **open-core** (e.g., Deque open-sources axe-core, sells enterprise extensions). OSS donations are thin: **<12% of devs earn anything; ~$1k/mo needs 2+ methods.**

**What it means for you.** Treat money as **layers**, not one stream:
- **Layer A — services (now):** freelance computational-audiology / health-data analysis & dashboards (real remote demand). Your repo is the portfolio.
- **Layer B — community (now):** GitHub Sponsors (0 fees) + Open Collective (transparent) via a single `donatr.ee` link. Realistic, modest, compounds with visibility.
- **Layer C — catalytic (1–3 mo):** grants/accelerators that *pay you to build* (Section 5).
- **Layer D — paid product (later, optional):** an "empathy/awareness" license for **corporate DEI / audiology clinics / schools**, or a "hear-your-audiogram" hosted tool — open-core: the simulator stays free, the org-facing layer is paid.

**Action.** Stand up Layer A + B this week; design v2 so Layer D is *possible later* (clean core + an org-facing surface) without compromising the free tool.

## 4. Building it — fast, solo, and actually accessible

**Finding.** AI-assisted dev lets solo builders ship production web apps in days–weeks; but **automated a11y tooling catches only 30–40%** of WCAG issues — screen-reader, keyboard, and cognitive checks are **manual**. WCAG 2.2 AA is the practical bar (3.0 is emerging).

**What it means for you.** You (with AI help) can build Product v2 Phase 1 quickly, but the credibility-defining part — *it's genuinely accessible* — needs hand testing (NVDA/VoiceOver, keyboard-only, captions/transcripts). That manual pass is itself a story-worthy "build in public" artifact.

**Action.** Build Phase 1 from `PRODUCT_v2_SPEC.md`; budget a manual a11y test pass; document it publicly. Attract contributors by labeling good-first-issues (a11y, translations, audio samples).

## 5. Where the money actually is — submit targets (verify dates)

| Target | What / amount | Fit | Note |
|---|---|---|---|
| **Freelance** (Upwork/Toptal/Indeed remote) | Health-data / audiology analysis, dashboards | **Now** | Real demand; fastest cash; repo = portfolio |
| **GitHub Sponsors + Open Collective** | Recurring donations | **Now** | Thin alone; set up anyway, compounds with LinkedIn |
| **Hackathons** (e.g., Hack for Humanity, summer 2026) | Cash prizes + visibility, remote | Weeks | Phase-1 demo is a strong entry |
| **WHO World Hearing Day small grants** | Small grants for hearing activities | Weeks–months | 2026 theme "hearing care for every child" = your story |
| **Remarkable** | 16-wk disability-tech accelerator, seed funding | 1–3 mo | **Strong** — disabled founder + disability tech |
| **Fast Forward** | 3-mo accelerator, **$25k+ unrestricted**, lived-experience-friendly, global | 1–3 mo | Requires **tech-nonprofit** structure |
| **NLnet / NGI Zero** | FOSS + accessibility, **individuals eligible**, 1–2 pg app, ~€5–50k | 1–3 mo | Commons call **closed 2026-06-01**; check active funds |
| **Google.org / Google for Startups** | GenAI-for-good; Cloud credits $2k–350k | Opportunistic | Credits useful; grant skews GenAI |
| **Grand Challenges (global health)** | Bold health-innovation grants 2026 | Opportunistic | Broad; needs a sharp framing |
| **Microsoft AI4A** | Accessibility grant | Later | 2026 narrowed to **generative AI** — needs reframe |
| **PanCare / ototoxicity & survivorship orgs** | Collaboration, audience, credibility | Ongoing | Not cash directly, but the *right room* + data partners |

**What it means for you.** The realistic flow: **freelance + sponsors (cash now) → hackathon + WHO small grant (small, fast) → Remarkable/NLnet/Fast Forward (catalytic) → optional paid layer.** Grants/accelerators almost all want: a working accessible demo, an open repo with a DOI, and a crisp, achievement-framed story — all of which you're days away from.

## 6. Distribution — LinkedIn & build-in-public as the moat

**Finding.** LinkedIn 2026: personal profiles ≈ **8× company-page engagement**; algorithm rewards **authentic engagement, video, consistency**; inbound converts **8× outbound**; winners are "remembered for something specific"; Creator Mode unlocks a **newsletter** (your email list). Indie playbook: **build in public**, narrow niche, MVP in 4–6 weeks, aim **$1k→$3k MRR**, optimize retention.

**What it means for you.** A consistent, specific public identity — *open, rigorous, accessible computational audiology, by someone who lives it* — is your distribution moat and your inbound-lead engine (freelance, collaboration, grant attention). Posting your build-in-public journey (the audit that made you withdraw a claim is GOLD content) compounds.

**Action.** Post the LinkedIn draft now. Then a cadence: the audit/integrity story; the binaural demo (video/GIF — the collapse-to-one-ear moment); the a11y build pass; the preprint/DOI. Turn on Creator Mode; start a short newsletter later.

---

## Sources (accessed 2026-06-15)
- Inspiration-porn / authentic disability framing: J. Business Ethics 2025 (link.springer.com/article/10.1007/s10551-025-06062-1); UMich news; Center for Disability Rights journalism guidelines.
- Competitor simulators: Phonak, Starkey, NIOSH/CDC hearing-loss simulators; Value Hearing "best tools 2025."
- Market size: market.us / custommarketinsights / mordorintelligence (assistive tech ~US$30.5B 2026; digital accessibility software).
- Corporate DEI simulation training: Understood, eLearning Industry, CultureAlly.
- Build-in-public / indie income: superframeworks, monolit.sh, buildmvpfast (2026).
- OSS income reality: markaicode "monetize open source 2026"; dev.to GitHub Sponsors revenue 2022 (~$1k/mo example); <12% earn.
- LinkedIn 2026: lagrowthmachine, hootsuite, supergrow, influenceflow.
- AI-assisted dev + a11y limits (30–40%): rocket.new, levelaccess, d2itechnology WCAG 3.0 2026.
- Accelerators/grants: Remarkable (remarkable.org / cparf.org); Fast Forward (ffwd.org, $25k+); Google.org / Google for Startups; NLnet (nlnet.nl/funding.html, Commons closed 2026-06-01).
- Childhood-cancer ototoxicity: PanCare / PanCareLIFE (PMC6444213, PMC5916870); IGHG ototoxicity surveillance (PubMed 30614474); 45–60% cisplatin HL.
- WHO World Hearing Day 2026 "hearing care for every child" (who.int); Global Grand Challenges 2026.
- Computational audiology demand: PMC8417156; remote health-data roles (Glassdoor/Indeed 2026).
