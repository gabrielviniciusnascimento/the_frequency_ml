#!/usr/bin/env python3
"""
Nome: grip_03_null_model.py
Tarefa: Espelha scripts/30_null_model.py no GRIP. Cópula gaussiana nos postos de
        grip_R_max/grip_L_max (preserva marginais + correlação de postos) + controle
        discreto. Conta assimetria interaural EXTREMA: Real vs Cópula, com SEQN.

Pergunta: há EXCESSO real de assimetria interaural de grip além do que a estrutura
de 2ª ordem (marginais+correlação, que já embute lateralidade/dominância) prevê?
O "18 vs 0" do workspace fechado é reproduzido aqui com IDs ou substituído pelo número real.

Também: overlap de SEQN entre casos extremos de grip e os 13 da audiometria.
Output: outputs/json/grip_03_null_model.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import hdbscan

RANDOM_STATE = 42
AGE_MIN, AGE_MAX = 20, 69
THRESHOLDS_KG = [10, 15, 20, 25, 30]
REL_THRESHOLDS = [0.3, 0.5]

FEATURE = Path("data/processed/grip_feature_matrix.csv")
AUDIO_13 = Path("outputs/json/27_binaural_pooling_ablation.json")
OUTPUT = Path("outputs/json/grip_03_null_model.json")
LOG = Path("outputs/logs/grip_03_null_model.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger(__name__)


def gaussian_copula(real2, seed=RANDOM_STATE):
    """Cópula gaussiana 2D: correlação de postos + marginais empíricas (R_max, L_max)."""
    n, p = real2.shape
    ranks = np.column_stack([stats.rankdata(real2[:, j]) for j in range(p)])
    Rcorr = np.corrcoef(ranks, rowvar=False)
    Z = np.random.RandomState(seed).multivariate_normal(np.zeros(p), Rcorr, size=n)
    U = stats.norm.cdf(Z)
    syn = np.empty((n, p))
    for j in range(p):
        sv = np.sort(real2[:, j])
        idx = np.clip((U[:, j] * len(sv)).astype(int), 0, len(sv) - 1)
        syn[:, j] = sv[idx]
    return syn


def discrete_control(real2, seed=RANDOM_STATE):
    """Controle: 2 grupos de FORÇA separados (fraco/forte), ambos simétricos R≈L.
    Mostra que estrutura discreta de NÍVEL não cria assimetria interaural."""
    n = len(real2)
    r = np.random.RandomState(seed)
    base = real2.mean(0)
    sd = real2.std(0)
    lab = r.randint(0, 2, size=n)
    centers = np.array([base - 1.5 * sd, base + 1.5 * sd])
    syn = centers[lab] + r.normal(0, sd * 0.15, size=(n, 2))
    return syn


def asym_counts(R, L, label):
    d = np.abs(R - L)
    rel = d / ((R + L) / 2.0)
    out = {f"n_abs_gt_{t}kg": int((d > t).sum()) for t in THRESHOLDS_KG}
    out.update({f"n_rel_gt_{int(t*100)}pct": int((rel > t).sum()) for t in REL_THRESHOLDS})
    out["max_abs_kg"] = round(float(d.max()), 1)
    out["median_abs_kg"] = round(float(np.median(d)), 1)
    log.info(f"  {label}: " + ", ".join(f"{k}={v}" for k, v in out.items()))
    return out


def main():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)].copy()
    df = df[df["grip_R_max"].notna() & df["grip_L_max"].notna()].copy()
    seqn = df["SEQN"].astype("int64").to_numpy()
    R = df["grip_R_max"].to_numpy(np.float64)
    L = df["grip_L_max"].to_numpy(np.float64)
    real2 = np.column_stack([R, L])
    log.info(f"GRIP 20-69 ambas as mãos: N={len(df)}")

    log.info("Contagens de assimetria interaural extrema:")
    real_c = asym_counts(R, L, "REAL")
    cop = gaussian_copula(real2)
    cop_c = asym_counts(cop[:, 0], cop[:, 1], "CÓPULA")
    disc = discrete_control(real2)
    disc_c = asym_counts(disc[:, 0], disc[:, 1], "DISCRETO(nível)")

    # SEQN rastreáveis dos casos extremos reais (por threshold de referência)
    extreme_by_thr = {}
    for t in THRESHOLDS_KG:
        mask = np.abs(R - L) > t
        extreme_by_thr[f"gt_{t}kg"] = {
            "n_real": int(mask.sum()), "n_copula": cop_c[f"n_abs_gt_{t}kg"],
            "seqn_real": sorted(int(s) for s in seqn[mask]),
        }

    # Overlap com os 13 da audiometria
    audio13 = json.loads(AUDIO_13.read_text(encoding="utf-8"))[
        "arm_A_14d_separate_ears"]["asymmetry_cluster"]["seqn"]
    grip_seqn_set = set(int(s) for s in seqn)
    audio13_in_grip = sorted(s for s in audio13 if s in grip_seqn_set)
    # extremos de grip (usar 20 kg como referência) ∩ 13 audiometria
    grip_extreme_20 = set(extreme_by_thr["gt_20kg"]["seqn_real"])
    overlap = sorted(grip_extreme_20 & set(audio13))

    pearson = float(stats.pearsonr(R, L)[0])
    result = {
        "script": "grip_03_null_model.py",
        "n_samples": int(len(df)),
        "rl_pearson_r": round(pearson, 4),
        "asymmetry_counts": {"real": real_c, "copula": cop_c, "discrete_level_control": disc_c},
        "extreme_cases_real_vs_copula_by_threshold": extreme_by_thr,
        "audiometry_overlap": {
            "audio13_seqn": audio13,
            "audio13_present_in_grip_dataset": audio13_in_grip,
            "n_audio13_in_grip": len(audio13_in_grip),
            "grip_extreme_gt20kg_AND_audio13": overlap,
        },
        "interpretation": (
            f"R-L grip Pearson r={pearson:.2f} (alta correlação por lateralidade/dominância). "
            f"Real |R-L|>20kg: {real_c['n_abs_gt_20kg']} casos; Cópula: {cop_c['n_abs_gt_20kg']}. "
            "Se Real≈Cópula, a assimetria de grip é explicada por estrutura de 2ª ordem (sem "
            "excesso real) — DIVERGE da audiometria (69 vs 0). Se Real≫Cópula, há cauda real."
        ),
        "status": "EXECUTED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nGRIP null model: N={len(df)}, R-L Pearson r={pearson:.3f}")
    print(f"{'threshold':>12} | {'REAL':>6} | {'CÓPULA':>7} | {'DISCRETO':>8}")
    for t in THRESHOLDS_KG:
        print(f"  |R-L|>{t:>2}kg   | {real_c[f'n_abs_gt_{t}kg']:>6} | {cop_c[f'n_abs_gt_{t}kg']:>7} | {disc_c[f'n_abs_gt_{t}kg']:>8}")
    for t in REL_THRESHOLDS:
        k = f"n_rel_gt_{int(t*100)}pct"
        print(f"  |R-L|>{int(t*100)}%   | {real_c[k]:>6} | {cop_c[k]:>7} | {disc_c[k]:>8}")
    print(f"\n13 da audiometria presentes no dataset de grip: {len(audio13_in_grip)} {audio13_in_grip}")
    print(f"Overlap (grip |R-L|>20kg ∩ audio13): {overlap if overlap else 'nenhum'}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
