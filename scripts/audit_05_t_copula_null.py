#!/usr/bin/env python3
"""
audit_05_t_copula_null.py  (HANDOFF Task 5 — t-copula null with tail dependence)

A Gaussian copula has zero tail dependence: extreme values in the two channels become
asymptotically independent. A Student-t copula (nu=4) has positive tail dependence — both
sides tend to be jointly extreme — so it is a *more hostile* null for an inter-side contrast
tail (it can itself produce large joint excursions, but, being symmetric, also large
contrasts). We repeat the Monte-Carlo envelope (B=2000) against a t-copula (nu=4) with the
same rank correlation and the same empirical marginals.

Criterion (handoff): the auditory excess at |z|>4 survives the t-copula with p<0.01.

Output: outputs/json/audit_05_t_copula_null.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/json/audit_05_t_copula_null.json"
B = 2000
NU = 4
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


def draw_t(Rc, sc, n, seed):
    """Student-t copula: MVN scaled by sqrt(nu/chi2_nu) -> multivariate t; map via t.cdf."""
    p = Rc.shape[0]
    rng = np.random.RandomState(seed)
    Z = rng.multivariate_normal(np.zeros(p), Rc, size=n)
    g = rng.chisquare(NU, size=n) / NU
    T = Z / np.sqrt(g)[:, None]                  # multivariate t, nu df, corr Rc
    U = stats.t.cdf(T, df=NU)                     # uniform marginals (t-copula)
    syn = np.empty((n, p))
    for j in range(p):
        syn[:, j] = sc[j][np.clip((U[:, j] * len(sc[j])).astype(int), 0, len(sc[j]) - 1)]
    return syn


def deq(M, step, seed):
    return np.asarray(M, float) + np.random.RandomState(seed).uniform(-step/2, step/2, size=np.asarray(M).shape)


def counts(c, sd):
    z = np.abs(c) / sd
    return np.array([int(np.nansum(z > lv)) for lv in Z_LEVELS])


def envelope(Rc, sc, n, sd, is_audio):
    null = np.zeros((B, len(Z_LEVELS)), int)
    for b in range(B):
        syn = draw_t(Rc, sc, n, seed=b)
        if is_audio:
            syn = deq(syn, 5.0, seed=b + 100000)
            c = np.nanmean(syn[:, :7], 1) - np.nanmean(syn[:, 7:], 1)
        else:
            c = syn[:, 0] - syn[:, 1]
        null[b] = counts(c, sd)
    return null


def cells(real_counts, null):
    out = {}
    for i, lv in enumerate(Z_LEVELS):
        col = null[:, i]; r = int(real_counts[i]); nm = float(col.mean())
        out[str(lv)] = {
            "real": r, "null_mean": round(nm, 2), "null_ci95": [int(np.percentile(col, 2.5)), int(np.percentile(col, 97.5))],
            "null_max": int(col.max()), "ratio_real_over_nullmean": round(r / nm, 2) if nm > 0 else None,
            "p_emp": round((int(np.sum(col >= r)) + 1) / (B + 1), 5),
            "sig_bonf_0.01_over_12": bool((int(np.sum(col >= r)) + 1) / (B + 1) < 0.01 / 12),
        }
    return out


def main():
    res = {}
    # auditory
    df = pd.read_csv(ROOT / "data/processed/frequencia_feature_matrix_v1.csv", low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce"); df = df[(age >= 20) & (age <= 69)]
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce"); thr = thr[thr.notna().sum(axis=1) >= 10]
    thr = thr[(thr > 25).any(axis=1)]
    real = deq(thr.to_numpy(np.float64), 5.0, seed=107)
    c = np.nanmean(real[:, :7], 1) - np.nanmean(real[:, 7:], 1); sd = float(np.nanstd(c))
    Rc, sc = precompute(thr.to_numpy(np.float64))
    res["audicao"] = {"n": thr.shape[0], "z_cells": cells(counts(c, sd), envelope(Rc, sc, thr.shape[0], sd, True))}
    # grip / vision
    for name, path, rc, lc in [("grip", "data/processed/grip_feature_matrix.csv", "grip_R_max", "grip_L_max"),
                               ("visao", "data/processed/vis_feature_matrix.csv", "vis_R", "vis_L")]:
        d = pd.read_csv(ROOT / path, low_memory=False)
        a = pd.to_numeric(d["RIDAGEYR"], errors="coerce"); d = d[(a >= 20) & (a <= 69)]
        d = d[d[rc].notna() & d[lc].notna()]
        r2 = d[[rc, lc]].to_numpy(np.float64); sd2 = float(np.nanstd(r2[:, 0] - r2[:, 1]))
        Rc2, sc2 = precompute(r2)
        res[name] = {"n": r2.shape[0], "z_cells": cells(counts(r2[:, 0] - r2[:, 1], sd2), envelope(Rc2, sc2, r2.shape[0], sd2, False))}

    OUT.write_text(json.dumps({
        "script": "audit_05_t_copula_null.py", "task": "HANDOFF Task 5 — t-copula null (nu=4)",
        "B": B, "nu": NU, "systems": res, "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for name in ("audicao", "grip", "visao"):
        s = res[name]; print(f"\n=== {name} (n={s['n']}) — t-copula nu={NU} ===")
        print(f"{'|z|':>4} | {'real':>5} | {'null mean [95% CI]':>22} | {'p_emp':>8} | sig")
        for lv in Z_LEVELS:
            cc = s["z_cells"][str(lv)]; ci = f"{cc['null_mean']:.1f} [{cc['null_ci95'][0]},{cc['null_ci95'][1]}]"
            print(f"{'>'+str(lv):>4} | {cc['real']:>5} | {ci:>22} | {cc['p_emp']:>8} | {'Y' if cc['sig_bonf_0.01_over_12'] else 'n'}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
