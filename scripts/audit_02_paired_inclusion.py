#!/usr/bin/env python3
"""
audit_02_paired_inclusion.py  (HANDOFF Task 2 — paired inclusion across systems)

Addresses D2: the auditory system was filtered by ANY25 (abnormal audiograms only);
grip/vision were not. The cross-system gradient could be a selection artifact. We
recompute the MC envelope (B per system) under THREE comparable inclusion policies:

  (i)  none     — no abnormality filter on ANY system (the hardest test for auditory)
  (ii) abn_all  — abnormality on all: auditory ANY25; grip min-hand < P20 within
                  sex x age-decade; vision max|SE| >= 1.0 D
  (iii) current — auditory ANY25, grip/vision unfiltered (the published policy)

Criterion (handoff): ordering auditory > grip > vision preserved, AND auditory real/null
at |z|>3 stays > 2 even with auditory WITHOUT ANY25 (policy i). If it collapses/inverts,
the gradient was selection.

Output: outputs/json/audit_02_paired_inclusion.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/json/audit_02_paired_inclusion.json"
B = 1000
Z_LEVELS = [2, 3, 4, 5]
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R14 = [f"thr_R_{f}" for f in FREQS]; L14 = [f"thr_L_{f}" for f in FREQS]; COLS14 = R14 + L14


def nearest_psd_corr(C):
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None)
    C2 = V @ np.diag(w) @ V.T; d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


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


def counts(c, sd):
    z = np.abs(c) / sd
    return np.array([int(np.nansum(z > lv)) for lv in Z_LEVELS])


def run_system(real, is_audio, sd):
    real_counts = counts(
        (np.nanmean(real[:, :7], 1) - np.nanmean(real[:, 7:], 1)) if is_audio else real[:, 0] - real[:, 1], sd)
    Rc, sc = precompute(real); n = real.shape[0]
    null = np.zeros((B, len(Z_LEVELS)), int)
    for b in range(B):
        syn = draw(Rc, sc, n, seed=b)
        if is_audio:
            syn = deq(syn, 5.0, seed=b + 100000)
            c = np.nanmean(syn[:, :7], 1) - np.nanmean(syn[:, 7:], 1)
        else:
            c = syn[:, 0] - syn[:, 1]
        null[b] = counts(c, sd)
    cells = {}
    for i, lv in enumerate(Z_LEVELS):
        col = null[:, i]; r = int(real_counts[i])
        nm = float(col.mean())
        cells[str(lv)] = {
            "real": r, "null_mean": round(nm, 2),
            "ratio_real_over_nullmean": round(r / nm, 2) if nm > 0 else None,
            "p_emp": round((int(np.sum(col >= r)) + 1) / (B + 1), 5),
        }
    return {"n": n, "z_cells": cells}


# ---------- loaders with policy ----------
def load_audio(policy):
    df = pd.read_csv(ROOT / "data/processed/frequencia_feature_matrix_v1.csv", low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce"); df = df[(age >= 20) & (age <= 69)]
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce"); thr = thr[thr.notna().sum(axis=1) >= 10]
    if policy != "none":  # abn / current both apply ANY25
        thr = thr[(thr > 25).any(axis=1)]
    real = deq(thr.to_numpy(np.float64), 5.0, seed=107)
    c = np.nanmean(real[:, :7], 1) - np.nanmean(real[:, 7:], 1)
    return real, float(np.nanstd(c))


def load_grip(policy):
    d = pd.read_csv(ROOT / "data/processed/grip_feature_matrix.csv", low_memory=False)
    age = pd.to_numeric(d["RIDAGEYR"], errors="coerce"); d = d[(age >= 20) & (age <= 69)]
    d = d[d["grip_R_max"].notna() & d["grip_L_max"].notna()].copy()
    if policy == "abn_all":
        d["dec"] = (pd.to_numeric(d["RIDAGEYR"], errors="coerce") // 10).astype(int)
        d["minhand"] = d[["grip_R_max", "grip_L_max"]].min(axis=1)
        pooled = pd.concat([d[["RIAGENDR", "dec", "grip_R_max"]].rename(columns={"grip_R_max": "g"}),
                            d[["RIAGENDR", "dec", "grip_L_max"]].rename(columns={"grip_L_max": "g"})])
        p20 = pooled.groupby(["RIAGENDR", "dec"])["g"].quantile(0.20).rename("p20")
        d = d.join(p20, on=["RIAGENDR", "dec"])
        d = d[d["minhand"] < d["p20"]]
    real = d[["grip_R_max", "grip_L_max"]].to_numpy(np.float64)
    return real, float(np.nanstd(real[:, 0] - real[:, 1]))


def load_vis(policy):
    d = pd.read_csv(ROOT / "data/processed/vis_feature_matrix.csv", low_memory=False)
    age = pd.to_numeric(d["RIDAGEYR"], errors="coerce"); d = d[(age >= 20) & (age <= 69)]
    d = d[d["vis_R"].notna() & d["vis_L"].notna()].copy()
    if policy == "abn_all":
        d = d[(d["vis_R"].abs() >= 1.0) | (d["vis_L"].abs() >= 1.0)]
    real = d[["vis_R", "vis_L"]].to_numpy(np.float64)
    return real, float(np.nanstd(real[:, 0] - real[:, 1]))


def main():
    out = {}
    for policy in ("none", "abn_all", "current"):
        # loaders read the policy: auditory ANY25 unless 'none'; grip/vision filtered only if 'abn_all'
        ra, sda = load_audio(policy)
        rg, sdg = load_grip(policy)
        rv, sdv = load_vis(policy)
        out[policy] = {
            "audicao": run_system(ra, True, sda),
            "grip": run_system(rg, False, sdg),
            "visao": run_system(rv, False, sdv),
        }
        print(f"\n##### POLICY = {policy} #####")
        for name in ("audicao", "grip", "visao"):
            s = out[policy][name]; c3 = s["z_cells"]["3"]; c4 = s["z_cells"]["4"]
            print(f"  {name:7s} n={s['n']:6d} | "
                  f"|z|>3 real={c3['real']:>4} null={c3['null_mean']:>6} ratio={c3['ratio_real_over_nullmean']} p={c3['p_emp']} | "
                  f"|z|>4 real={c4['real']:>4} null={c4['null_mean']:>6} ratio={c4['ratio_real_over_nullmean']} p={c4['p_emp']}")

    OUT.write_text(json.dumps({
        "script": "audit_02_paired_inclusion.py",
        "task": "HANDOFF Task 2 — paired inclusion (D2)", "B": B,
        "policies": {"none": "no abnormality filter on any system",
                     "abn_all": "auditory ANY25; grip min-hand<P20(sex x decade); vision max|SE|>=1D",
                     "current": "auditory ANY25; grip/vision unfiltered (published)"},
        "results": out, "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
