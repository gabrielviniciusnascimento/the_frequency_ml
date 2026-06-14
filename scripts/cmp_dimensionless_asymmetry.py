#!/usr/bin/env python3
"""
Nome: cmp_dimensionless_asymmetry.py
Tarefa: Camada de comparação ADIMENSIONAL por cima dos três null models (audição/grip/visão).
        Não mexe nos thresholds clínicos nativos (dB/kg/D) — só padroniza o contraste.

Para cada sistema:
  1. contraste interaural c = R - L por indivíduo, no REAL e na CÓPULA (regenerada como no
     null model original do sistema).
  2. z = c / sd(c_real)  — MESMO sd (do real) aplicado a real e cópula.
  3. conta |z| > 2,3,4,5 para real e cópula.
  4. razão real/cópula = n_real / (n_cop + 1); reporta brutos também.

Audição: matriz 14 limiares (7695 ANY25), contraste = PTA_R - PTA_L; cópula = make_continuous
         14-dim + dequantize (idêntico a 30_null_model.py).
Grip:    grip_R_max - grip_L_max (20-69 ambas as mãos); cópula 2D (idêntico a grip_03).
Visão:   vis_R - vis_L SE (20-69 ambos olhos); cópula 2D (idêntico a vis_03).
Output: outputs/json/cmp_dimensionless_asymmetry.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RS = 42
Z_LEVELS = [2, 3, 4, 5]
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R14 = [f"thr_R_{f}" for f in FREQS]
L14 = [f"thr_L_{f}" for f in FREQS]
COLS14 = R14 + L14
OUT = Path("outputs/json/cmp_dimensionless_asymmetry.json")


# ── cópulas idênticas aos null models ────────────────────────────────
def nearest_psd_corr(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


def make_continuous_14(real14):  # idêntico a 30_null_model.make_continuous
    n, p = real14.shape
    ranks = np.zeros_like(real14)
    cols_data = []
    for j in range(p):
        col = real14[:, j]
        cols_data.append(np.sort(col[~np.isnan(col)]))
        r = stats.rankdata(col, nan_policy="omit")
        ranks[:, j] = np.where(np.isnan(col), np.nan, r)
    Rcorr = pd.DataFrame(ranks).corr().to_numpy()
    Rcorr = np.nan_to_num(Rcorr, nan=0.0); np.fill_diagonal(Rcorr, 1.0)
    Rcorr = nearest_psd_corr(Rcorr)
    Z = np.random.RandomState(RS).multivariate_normal(np.zeros(p), Rcorr, size=n)
    U = stats.norm.cdf(Z)
    syn = np.empty((n, p))
    for j in range(p):
        sv = cols_data[j]
        syn[:, j] = sv[np.clip((U[:, j] * len(sv)).astype(int), 0, len(sv) - 1)]
    return syn


def dequantize(M, step=5.0):  # idêntico a 30
    r = np.random.RandomState(RS + 7)
    return np.asarray(M, float) + r.uniform(-step / 2, step / 2, size=np.asarray(M).shape)


def copula_2d(real2):  # idêntico a grip_03 / vis_03
    n, p = real2.shape
    ranks = np.column_stack([stats.rankdata(real2[:, j]) for j in range(p)])
    Rcorr = np.corrcoef(ranks, rowvar=False)
    Z = np.random.RandomState(RS).multivariate_normal(np.zeros(p), Rcorr, size=n)
    U = stats.norm.cdf(Z)
    syn = np.empty((n, p))
    for j in range(p):
        sv = np.sort(real2[:, j])
        syn[:, j] = sv[np.clip((U[:, j] * len(sv)).astype(int), 0, len(sv) - 1)]
    return syn


def zcounts(c_real, c_cop, sd):
    zr, zc = np.abs(c_real) / sd, np.abs(c_cop) / sd
    return ({lv: int(np.nansum(zr > lv)) for lv in Z_LEVELS},
            {lv: int(np.nansum(zc > lv)) for lv in Z_LEVELS})


def load_audio():
    df = pd.read_csv("data/processed/frequencia_feature_matrix_v1.csv", low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= 20) & (age <= 69)]
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce")
    thr = thr[thr.notna().sum(axis=1) >= 10]
    thr = thr[(thr > 25).any(axis=1)]
    real14 = thr.to_numpy(np.float64)
    cop14 = dequantize(make_continuous_14(real14))
    real14 = dequantize(real14)
    cr = np.nanmean(real14[:, :7], 1) - np.nanmean(real14[:, 7:], 1)
    cc = np.nanmean(cop14[:, :7], 1) - np.nanmean(cop14[:, 7:], 1)
    return cr, cc


def load_2var(path, rcol, lcol):
    df = pd.read_csv(path, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= 20) & (age <= 69)]
    df = df[df[rcol].notna() & df[lcol].notna()]
    real2 = df[[rcol, lcol]].to_numpy(np.float64)
    cop2 = copula_2d(real2)
    return real2[:, 0] - real2[:, 1], cop2[:, 0] - cop2[:, 1]


def main():
    systems = {}
    systems["audicao"] = load_audio()
    systems["grip"] = load_2var("data/processed/grip_feature_matrix.csv", "grip_R_max", "grip_L_max")
    systems["visao"] = load_2var("data/processed/vis_feature_matrix.csv", "vis_R", "vis_L")

    results = {}
    for name, (cr, cc) in systems.items():
        sd = float(np.nanstd(cr))
        nr, nc = zcounts(cr, cc, sd)
        results[name] = {
            "n": int(np.sum(~np.isnan(cr))), "sd_native": round(sd, 4),
            "real": nr, "copula": nc,
            "ratio": {lv: round(nr[lv] / (nc[lv] + 1), 2) for lv in Z_LEVELS},
        }

    # ── Tabela ───────────────────────────────────────────────────────
    print(f"\nSD nativo do contraste R-L:  audição={results['audicao']['sd_native']} dB | "
          f"grip={results['grip']['sd_native']} kg | visão={results['visao']['sd_native']} D")
    print(f"\n{'|z|':>4} | {'AUDIÇÃO razão (real:cóp)':>26} | {'GRIP razão (real:cóp)':>24} | {'VISÃO razão (real:cóp)':>24}")
    print("-" * 86)
    for lv in Z_LEVELS:
        a, g, v = results["audicao"], results["grip"], results["visao"]
        print(f"{'>'+str(lv):>4} | {a['ratio'][lv]:>9}  ({a['real'][lv]:>4}:{a['copula'][lv]:<4}){'':>6} | "
              f"{g['ratio'][lv]:>8}  ({g['real'][lv]:>3}:{g['copula'][lv]:<3}){'':>5} | "
              f"{v['ratio'][lv]:>8}  ({v['real'][lv]:>3}:{v['copula'][lv]:<3})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "script": "cmp_dimensionless_asymmetry.py",
        "metric": "z = (R-L)/sd(R-L real); conta |z|>2,3,4,5; razão = real/(copula+1)",
        "note": "Camada adimensional; thresholds clínicos nativos preservados nos JSONs originais.",
        "systems": results, "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
