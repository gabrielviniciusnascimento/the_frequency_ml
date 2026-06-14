#!/usr/bin/env python3
"""
audit_01_mc_envelope.py  (HANDOFF Task 1 — Monte Carlo envelope of the null)

Addresses D1: the original "X vs 0" counts come from a SINGLE copula realization
(RandomState(42)). Here we regenerate each copula B=1000 times and turn every tail
count into  real | null mean [2.5,97.5] | empirical p, with
    p = (#{null_count >= real_count} + 1) / (B + 1).
Bonferroni target over the 3 systems x 4 |z| levels = 12 cells: alpha=0.01 -> 0.01/12.

Reuses the EXACT copula machinery of the original null models (rank-corr Gaussian copula
with empirical marginals; auditory = 14-dim + 5 dB dequantize), changing only the seed.
Rank-correlation and sorted marginals are computed ONCE; only the MVN draw varies per b.

Output: outputs/json/audit_01_mc_envelope.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/json/audit_01_mc_envelope.json"
B = 2000
Z_LEVELS = [2, 3, 4, 5]
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R14 = [f"thr_R_{f}" for f in FREQS]
L14 = [f"thr_L_{f}" for f in FREQS]
COLS14 = R14 + L14
AGE_MIN, AGE_MAX = 20, 69


def nearest_psd_corr(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


def precompute_copula(real):
    """Rank-corr (PSD) + sorted marginals, computed once; reused for every draw."""
    n, p = real.shape
    ranks = np.zeros_like(real)
    sorted_cols = []
    for j in range(p):
        col = real[:, j]
        sorted_cols.append(np.sort(col[~np.isnan(col)]))
        r = stats.rankdata(col, nan_policy="omit")
        ranks[:, j] = np.where(np.isnan(col), np.nan, r)
    Rcorr = pd.DataFrame(ranks).corr().to_numpy()
    Rcorr = np.nan_to_num(Rcorr, nan=0.0); np.fill_diagonal(Rcorr, 1.0)
    return nearest_psd_corr(Rcorr), sorted_cols


def draw_copula(Rcorr, sorted_cols, n, seed):
    p = Rcorr.shape[0]
    Z = np.random.RandomState(seed).multivariate_normal(np.zeros(p), Rcorr, size=n)
    U = stats.norm.cdf(Z)
    syn = np.empty((n, p))
    for j in range(p):
        sv = sorted_cols[j]
        syn[:, j] = sv[np.clip((U[:, j] * len(sv)).astype(int), 0, len(sv) - 1)]
    return syn


def dequantize(M, step, seed):
    r = np.random.RandomState(seed)
    return np.asarray(M, float) + r.uniform(-step / 2, step / 2, size=np.asarray(M).shape)


def tail_counts(c, sd):
    z = np.abs(c) / sd
    return np.array([int(np.nansum(z > lv)) for lv in Z_LEVELS])


def envelope(real_contrast_fn, Rcorr, sorted_cols, n, sd, is_audio):
    """Return real counts and the B-length null count matrix (B x 4)."""
    null = np.zeros((B, len(Z_LEVELS)), dtype=int)
    for b in range(B):
        syn = draw_copula(Rcorr, sorted_cols, n, seed=b)
        if is_audio:
            syn = dequantize(syn, 5.0, seed=b + 100000)
            c = np.nanmean(syn[:, :7], 1) - np.nanmean(syn[:, 7:], 1)
        else:
            c = syn[:, 0] - syn[:, 1]
        null[b] = tail_counts(c, sd)
    return null


def summarize(name, real_counts, null, native_thr, native_real, native_null):
    cells = {}
    for i, lv in enumerate(Z_LEVELS):
        col = null[:, i]
        r = int(real_counts[i])
        p = (int(np.sum(col >= r)) + 1) / (B + 1)
        cells[str(lv)] = {
            "real": r,
            "null_mean": round(float(col.mean()), 2),
            "null_ci95": [int(np.percentile(col, 2.5)), int(np.percentile(col, 97.5))],
            "null_max": int(col.max()),
            "p_emp": round(p, 5),
            "sig_bonferroni_0.01_over_12": bool(p < 0.01 / 12),
        }
    return {
        "native_threshold": native_thr,
        "native_real": native_real,
        "native_null_single_seed42": native_null,
        "z_cells": cells,
    }


def main():
    results = {}

    # ---- Auditory (ANY25, 14-dim) ----
    df = pd.read_csv(ROOT / "data/processed/frequencia_feature_matrix_v1.csv", low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)]
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce")
    thr = thr[thr.notna().sum(axis=1) >= 10]
    thr = thr[(thr > 25).any(axis=1)]
    realA = thr.to_numpy(np.float64)
    RcorrA, sortedA = precompute_copula(realA)
    realA_dq = dequantize(realA, 5.0, seed=107)
    cA = np.nanmean(realA_dq[:, :7], 1) - np.nanmean(realA_dq[:, 7:], 1)
    sdA = float(np.nanstd(cA))
    nullA = envelope(None, RcorrA, sortedA, realA.shape[0], sdA, is_audio=True)
    nat_real_A = int(np.nansum(np.abs(cA) > 50))
    results["audicao"] = summarize("audicao", tail_counts(cA, sdA), nullA,
                                   "|R-L| > 50 dB", nat_real_A, 0)
    results["audicao"]["n"] = realA.shape[0]
    results["audicao"]["null_dist_z_gt4"] = nullA[:, 2].tolist()

    # ---- Grip / Vision (2-var) ----
    for name, path, rc, lc, thr_native in [
        ("grip", "data/processed/grip_feature_matrix.csv", "grip_R_max", "grip_L_max", 20.0),
        ("visao", "data/processed/vis_feature_matrix.csv", "vis_R", "vis_L", 10.0),
    ]:
        d = pd.read_csv(ROOT / path, low_memory=False)
        a = pd.to_numeric(d["RIDAGEYR"], errors="coerce")
        d = d[(a >= AGE_MIN) & (a <= AGE_MAX)]
        d = d[d[rc].notna() & d[lc].notna()]
        real2 = d[[rc, lc]].to_numpy(np.float64)
        Rc, sc = precompute_copula(real2)
        c = real2[:, 0] - real2[:, 1]
        sd = float(np.nanstd(c))
        null = envelope(None, Rc, sc, real2.shape[0], sd, is_audio=False)
        nat_real = int(np.sum(np.abs(c) > thr_native))
        results[name] = summarize(name, tail_counts(c, sd), null,
                                  f"|R-L| > {thr_native}", nat_real, 0)
        results[name]["n"] = real2.shape[0]
        results[name]["null_dist_z_gt4"] = null[:, 2].tolist()

    OUT.write_text(json.dumps({
        "script": "audit_01_mc_envelope.py",
        "task": "HANDOFF Task 1 — Monte Carlo envelope (D1)",
        "B": B, "p_def": "(#{null>=real}+1)/(B+1)", "bonferroni": "alpha 0.01 / 12 cells",
        "systems": results, "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # console
    for name in ("audicao", "grip", "visao"):
        s = results[name]
        print(f"\n=== {name}  (n={s['n']}) ===")
        print(f"{'|z|':>4} | {'real':>5} | {'null mean [95% CI]':>22} | {'null max':>8} | {'p_emp':>8} | sig")
        for lv in Z_LEVELS:
            c = s["z_cells"][str(lv)]
            ci = f"{c['null_mean']:.1f} [{c['null_ci95'][0]},{c['null_ci95'][1]}]"
            print(f"{'>'+str(lv):>4} | {c['real']:>5} | {ci:>22} | {c['null_max']:>8} | "
                  f"{c['p_emp']:>8} | {'Y' if c['sig_bonferroni_0.01_over_12'] else 'n'}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
