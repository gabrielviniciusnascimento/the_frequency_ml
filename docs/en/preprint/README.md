# Preprint package — The Frequency v6 (for medRxiv)

Generated from `docs/en/PAPER_DRAFT_v6_crosssystem.md` with Pandoc 3.10.

## Files
- `the_frequency_v6_preprint.docx` — manuscript (Word; medRxiv accepts this directly). Math as
  equations, figures + Table 1 embedded.
- `the_frequency_v6_preprint.html` — self-contained (figures + math embedded). **To get a PDF:**
  open it in a browser → Print → "Save as PDF". This gives a clean PDF with correct math.

Regenerate after editing the manuscript:
```
pandoc PAPER_DRAFT_v6_crosssystem.md -f gfm+tex_math_dollars -o preprint/the_frequency_v6_preprint.docx --resource-path=".;<repo-root>"
pandoc PAPER_DRAFT_v6_crosssystem.md -f gfm+tex_math_dollars -t html5 -s --mathml --embed-resources -o preprint/the_frequency_v6_preprint.html --resource-path=".;<repo-root>"
```

## medRxiv submission — ready-to-paste fields

- **Title:** Signal or Artifact? A Null-Calibrated, Cross-System Audit of Interaural Asymmetry in Paired-Organ Measurements (NHANES: Auditory, Motor, Visual)
- **Author:** Gabriel Vinicius Nascimento — **Affiliation:** Independent Researcher — add your ORCID.
- **Corresponding author:** gabrielviniciusnascimento345@gmail.com
- **Subject category (suggested):** Otolaryngology (or Health Informatics).
- **Abstract:** use the Abstract section of the manuscript.

### Required declarations (paste as-is, verify they are true)
- **Funding:** None. This work received no external funding.
- **Competing interests:** The author declares no competing interests.
- **Author contributions:** G.V.N. conceived the study, performed all analyses, and wrote the manuscript. (Large language models assisted with drafting and code; see Disclosure of AI Assistance.)
- **Data availability:** All data are public — NHANES (CDC/NCHS) and the Oldenburg Hearing Health Record (OHHR, CC BY 4.0). All analysis code and intermediate JSON outputs are openly available at https://github.com/gabrielviniciusnascimento/the_frequency_ml (archived on Zenodo, DOI to be added).
- **Ethics / IRB:** Not required. This is a secondary analysis of publicly available, de-identified survey data; no human-subjects approval was needed.
- **License (suggested):** CC-BY 4.0.

## Notes
- medRxiv is health-sciences and accepts "Independent Researcher" — no institutional endorsement needed (unlike arXiv, which since Jan 2026 requires endorsement for unaffiliated authors).
- If you also want an arXiv version later (stat.AP), you'll need a personal endorsement from an established author in that area.
- Strengthen the repo link by replacing it with the Zenodo DOI once the GitHub release is made.
