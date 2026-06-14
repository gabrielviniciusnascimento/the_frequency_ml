#!/usr/bin/env python3
"""
audit_04_heteroscedastic_null.py  (HANDOFF Task 4 — heteroscedastic measurement null)

Addresses D5: "excess over a Gaussian copula" only excludes Gaussian dependence; it does
NOT exclude level-dependent (heteroscedastic) measurement error, which is real in
audiometry (worse ears are measured less reliably, in 5 dB steps near the limits). A more
hostile null adds per-channel measurement noise that GROWS with threshold level:
    noise SD(channel) = 5 + 0.1 * level_dB   (~17 dB at 120 dB HL; conservative)
applied on top of the rank-corr Gaussian copula. If the auditory far-tail excess survives
even this fattened null, the headline is not a measurement artifact.

Auditory only (the headline; the dB noise model is audiometry-specific). B=2000.
Reports side-by-side: p(Gaussian copula) | p(heteroscedastic copula).

Output: outputs/json/audit_04_heteroscedastic_null.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/json/audit_04_heteroscedastic_null.json"
B = 2000
Z_LEVELS = [2, 3, 4, 5]
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R14 = [f"thr_R_{f}" for f in FREQS]; L14 = [f"thr_L_{f}" for f in FREQS]; COLS14 = R14 + L14


def nearest_psd_corr(C):
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None)
    C2 = V @ np.diag(w) @ V.T; d = np.sqrt(np.diag(C2)); return C2 / np.outer(d, d)


def precompute(real):
    n, p = real.shape; ranks = np.zeros_like(real); sc = []
    for j in range(p):
        col = real[:, j]; sc.append(np.sort(col[~np.isnan(col)]))
        r = stats.rankdata(col, nan_policy="omit"); ranks[:, j] = np.where(np.isnan(col), np.nan, r)
    Rc = pd.DataFrame(ranks).corr().to_numpy(); Rc = np.nan_to_num(Rc, nan=0.0); np.fill_diagonal(Rc, 1.0)
    return nearest_psd_corr(Rc), sc


def draw(Rc, sc, n, seed):
    p = Rc.shape[0]
    Z = np.random.RandomState(seed).multivariate_normal(np.zeros(p), Rc, size=n)
    U = stats.norm.cdf(Z); syn = np.empty((n, p))
    for j in range(p):
        sv = sc[j]; syn[:, j] = sv[np.clip((U[:, j] * len(sv)).astype(int), 0, len(sv) - 1)]
    return syn


def deq(M, step, seed):
    return np.asarray(M, float) + np.random.RandomState(seed).uniform(-step/2, step/2, size=np.asarray(M).shape)


def add_hetero_noise(M, seed):
    """Per-channel Gaussian noise with SD = 5 + 0.1*level_dB (level clipped >=0)."""
    sd = 5.0 + 0.1 * np.clip(M, 0, None)
    return M + np.random.RandomState(seed).normal(0, 1, size=M.shape) * sd


def counts(c, sd):
    z = np.abs(c) / sd
    return np.array([int(np.nansum(z > lv)) for lv in Z_LEVELS])


def envelope(Rc, sc, n, sd, hetero):
    null = np.zeros((B, len(Z_LEVELS)), int)
    for b in range(B):
        syn = deq(draw(Rc, sc, n, seed=b), 5.0, seed=b + 100000)
        if hetero:
            syn = add_hetero_noise(syn, seed=b + 500000)
        c = np.nanmean(syn[:, :7], 1) - np.nanmean(syn[:, 7:], 1)
        null[b] = counts(c, sd)
    return null


def cells(real_counts, null):
    out = {}
    for i, lv in enumerate(Z_LEVELS):
        col = null[:, i]; r = int(real_counts[i]); nm = float(col.mean())
        out[str(lv)] = {
            "real": r, "null_mean": round(nm, 2), "null_max": int(col.max()),
            "ratio_real_over_nullmean": round(r / nm, 2) if nm > 0 else None,
            "p_emp": round((int(np.sum(col >= r)) + 1) / (B + 1), 5),
            "sig_bonf_0.01_over_8": bool((int(np.sum(col >= r)) + 1) / (B + 1) < 0.01 / 8),
        }
    return out


def main():
    df = pd.read_csv(ROOT / "data/processed/frequencia_feature_matrix_v1.csv", low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce"); df = df[(age >= 20) & (age <= 69)]
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce"); thr = thr[thr.notna().sum(axis=1) >= 10]
    thr = thr[(thr > 25).any(axis=1)]
    real = deq(thr.to_numpy(np.float64), 5.0, seed=107)
    c = np.nanmean(real[:, :7], 1) - np.nanmean(real[:, 7:], 1)
    sd = float(np.nanstd(c)); rc = counts(c, sd)
    Rc, sc = precompute(thr.to_numpy(np.float64)); n = thr.shape[0]

    gauss = cells(rc, envelope(Rc, sc, n, sd, hetero=False))
    hetero = cells(rc, envelope(Rc, sc, n, sd, hetero=True))

    OUT.write_text(json.dumps({
        "script": "audit_04_heteroscedastic_null.py",
        "task": "HANDOFF Task 4 — heteroscedastic measurement null (D5)", "B": B, "n": n,
        "noise_model": "per-channel SD = 5 + 0.1*level_dB, added on top of Gaussian copula",
        "gaussian_copula": gauss, "heteroscedastic_copula": hetero, "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"AUDITORY n={n}  SD(R-L)={sd:.2f} dB   B={B}")
    print(f"{'|z|':>4} | {'real':>5} | {'p(gauss)':>9} {'ratio':>7} | {'p(hetero)':>10} {'ratio':>7} {'nullmax':>8}")
    print("-" * 66)
    for lv in Z_LEVELS:
        g = gauss[str(lv)]; h = hetero[str(lv)]
        print(f"{'>'+str(lv):>4} | {g['real']:>5} | {g['p_emp']:>9} {str(g['ratio_real_over_nullmean']):>7} | "
              f"{h['p_emp']:>10} {str(h['ratio_real_over_nullmean']):>7} {h['null_max']:>8}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
