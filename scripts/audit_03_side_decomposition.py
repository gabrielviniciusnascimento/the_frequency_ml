#!/usr/bin/env python3
"""
audit_03_side_decomposition.py  (HANDOFF Task 3 — tail decomposition by side)

Confirms/quantifies D3: the auditory extreme tail is unanimously RIGHT-worse, which
contradicts Cox & Ford (1995, left-worse in weapons-noise exposure). We tabulate the
full interaural-contrast tail (native >40/>50 dB; standardized |z|>2..5) split by sign
(R-worse vs L-worse) and test the R-worse fraction against 50/50 with an exact binomial.

Same inclusion as 30_null_model.py: age 20-69, completeness >=10/14, ANY25 (>25 dB in
any of the 14 thresholds). Contrast c = PTA_R - PTA_L (per-ear mean of 7 frequencies).
NO Monte Carlo here (a binomial sign test needs none). Seed-free.

Output: outputs/json/audit_03_side_decomposition.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FEATURE = ROOT / "data/processed/frequencia_feature_matrix_v1.csv"
OUT = ROOT / "outputs/json/audit_03_side_decomposition.json"

FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R14 = [f"thr_R_{f}" for f in FREQS]
L14 = [f"thr_L_{f}" for f in FREQS]
COLS14 = R14 + L14
AGE_MIN, AGE_MAX, MIN_COMPLETENESS, ANY25 = 20, 69, 10, 25


def binom_rworse(n_r, n_tot):
    """Exact two-sided binomial of R-worse count vs p=0.5; returns frac and p."""
    if n_tot == 0:
        return None, None
    res = stats.binomtest(n_r, n_tot, 0.5, alternative="two-sided")
    return round(n_r / n_tot, 4), float(res.pvalue)


def main():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)].copy()
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce")
    keep = thr.notna().sum(axis=1) >= MIN_COMPLETENESS
    thr = thr[keep]
    thr = thr[(thr > ANY25).any(axis=1)]
    seqn = df.loc[thr.index, "SEQN"].astype("int64").to_numpy() if "SEQN" in df.columns else None

    pta_r = np.nanmean(thr[R14].to_numpy(np.float64), axis=1)
    pta_l = np.nanmean(thr[L14].to_numpy(np.float64), axis=1)
    c = pta_r - pta_l                       # >0 => right ear worse
    sd = float(np.nanstd(c))
    N = int(np.sum(~np.isnan(c)))

    rows = {}
    # native dB thresholds
    for t in (40, 50):
        m = np.abs(c) > t
        n_tot = int(m.sum())
        n_r = int((c[m] > 0).sum())
        frac, p = binom_rworse(n_r, n_tot)
        rows[f"native_abs_gt_{t}dB"] = {
            "n_tail": n_tot, "n_right_worse": n_r, "n_left_worse": n_tot - n_r,
            "frac_right_worse": frac, "binom_p_vs_50_50": p,
        }
    # standardized |z| thresholds
    for lv in (2, 3, 4, 5):
        m = np.abs(c) / sd > lv
        n_tot = int(np.nansum(m))
        n_r = int(np.nansum((c > 0) & m))
        frac, p = binom_rworse(n_r, n_tot)
        rows[f"abs_z_gt_{lv}"] = {
            "n_tail": n_tot, "n_right_worse": n_r, "n_left_worse": n_tot - n_r,
            "frac_right_worse": frac, "binom_p_vs_50_50": p,
        }

    # whole-population baseline sign split (is right-worse already the population norm?)
    pop_r = int(np.nansum(c > 0)); pop_l = int(np.nansum(c < 0))
    pop_frac, pop_p = binom_rworse(pop_r, pop_r + pop_l)

    out = {
        "script": "audit_03_side_decomposition.py",
        "task": "HANDOFF Task 3 — tail decomposition by side (D3)",
        "N": N, "sd_native_db": round(sd, 4),
        "contrast_def": "c = PTA_R - PTA_L (mean of 7 freq per ear); c>0 = right ear worse",
        "population_sign_split": {
            "n_right_worse": pop_r, "n_left_worse": pop_l,
            "frac_right_worse": pop_frac, "binom_p_vs_50_50": pop_p,
        },
        "tail_by_side": rows,
        "test_order_variable_available": False,
        "note_test_order": "No test-order / ear-first variable in the processed matrix "
                           "(would require re-ingesting raw AUX). Quality codes AUAREQC/AUALEQC exist.",
        "status": "EXECUTED",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # console table
    print(f"N={N}  SD(R-L)={sd:.2f} dB")
    print(f"Population baseline: R-worse {pop_r} / L-worse {pop_l}  "
          f"(frac_R={pop_frac}, binom p={pop_p:.3g})")
    print(f"{'tail':>16} | {'n':>4} | {'R-worse':>7} | {'L-worse':>7} | {'frac_R':>7} | {'binom p':>10}")
    print("-" * 70)
    for k, v in rows.items():
        print(f"{k:>16} | {v['n_tail']:>4} | {v['n_right_worse']:>7} | {v['n_left_worse']:>7} | "
              f"{str(v['frac_right_worse']):>7} | {('%.3g' % v['binom_p_vs_50_50']) if v['binom_p_vs_50_50'] is not None else 'NA':>10}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
