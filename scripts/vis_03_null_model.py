#!/usr/bin/env python3
"""
Nome: vis_03_null_model.py
Tarefa: Espelha 30/grip_03 na VISÃO (controle negativo da tese). Cópula gaussiana nos
        postos de (vis_R, vis_L) + controle discreto. Conta anisometropia interaural
        EXTREMA (asym = OD - OS) Real vs Cópula, com SEQN. Inclui o OVERLAP TRIPLO.

Threshold de "extremo" (documentado): escala em dioptrias -> lógica clínica de
anisometropia. Reporta |OD-OS| > 1,2,3,5 D; referência = >3 D (anisometropia
clinicamente significativa); cauda = >5 D (severa). Relativo >50% reportado com
ressalva: perto da emetropia (SE≈0) o relativo explode e não é informativo.

Output: outputs/json/vis_03_null_model.json
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RANDOM_STATE = 42
AGE_MIN, AGE_MAX = 20, 69
THRESHOLDS_D = [1, 2, 3, 5, 8, 10]
REF_D = 3
FEATURE = Path("data/processed/vis_feature_matrix.csv")
AUDIO = Path("outputs/json/27_binaural_pooling_ablation.json")
GRIP = Path("outputs/json/grip_03_null_model.json")
GRIP_MATRIX = Path("data/processed/grip_feature_matrix.csv")
OUTPUT = Path("outputs/json/vis_03_null_model.json")
LOG = Path("outputs/logs/vis_03_null_model.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger(__name__)


def gaussian_copula(real2, seed=RANDOM_STATE):
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
    n = len(real2)
    r = np.random.RandomState(seed)
    base, sd = real2.mean(0), real2.std(0)
    lab = r.randint(0, 2, size=n)
    centers = np.array([base - 1.5 * sd, base + 1.5 * sd])
    return centers[lab] + r.normal(0, sd * 0.15, size=(n, 2))


def asym_counts(R, L, label):
    d = np.abs(R - L)
    rel = d / (np.abs((R + L) / 2.0) + 1e-9)
    out = {f"n_abs_gt_{t}D": int((d > t).sum()) for t in THRESHOLDS_D}
    out["n_rel_gt_50pct"] = int((rel > 0.5).sum())
    out["max_abs_D"] = round(float(d.max()), 2)
    out["median_abs_D"] = round(float(np.median(d)), 2)
    log.info(f"  {label}: " + ", ".join(f"{k}={v}" for k, v in out.items()))
    return out


def main():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)].copy()
    df = df[df["vis_R"].notna() & df["vis_L"].notna()].copy()
    seqn = df["SEQN"].astype("int64").to_numpy()
    R = df["vis_R"].to_numpy(np.float64)
    L = df["vis_L"].to_numpy(np.float64)
    real2 = np.column_stack([R, L])
    log.info(f"VISÃO 20-69 ambos olhos: N={len(df)}")

    log.info("Contagens de anisometropia extrema:")
    real_c = asym_counts(R, L, "REAL")
    cop = gaussian_copula(real2); cop_c = asym_counts(cop[:, 0], cop[:, 1], "CÓPULA")
    disc = discrete_control(real2); disc_c = asym_counts(disc[:, 0], disc[:, 1], "DISCRETO(nível)")

    extreme_by_thr = {}
    for t in THRESHOLDS_D:
        mask = np.abs(R - L) > t
        extreme_by_thr[f"gt_{t}D"] = {"n_real": int(mask.sum()), "n_copula": cop_c[f"n_abs_gt_{t}D"],
                                      "seqn_real": sorted(int(s) for s in seqn[mask])}
    pearson = float(stats.pearsonr(R, L)[0])

    # ── PASSO 5: OVERLAP TRIPLO ──────────────────────────────────────
    audio13 = json.loads(AUDIO.read_text(encoding="utf-8"))[
        "arm_A_14d_separate_ears"]["asymmetry_cluster"]["seqn"]
    grip15 = json.loads(GRIP.read_text(encoding="utf-8"))[
        "extreme_cases_real_vs_copula_by_threshold"]["gt_20kg"]["seqn_real"]
    vis_valid = set(int(s) for s in seqn)
    vis_extreme = set(extreme_by_thr[f"gt_{REF_D}D"]["seqn_real"])  # referência >3D

    def overlap(name, seqn_list):
        has_vis = sorted(s for s in seqn_list if s in vis_valid)
        also_ext = sorted(s for s in has_vis if s in vis_extreme)
        return {"n_source": len(seqn_list), "n_with_valid_vision": len(has_vis),
                "seqn_with_vision": has_vis, "n_also_vision_extreme_gt3D": len(also_ext),
                "seqn_also_extreme": also_ext}

    ov_audio = overlap("audio13", audio13)
    ov_grip = overlap("grip15", grip15)

    result = {
        "script": "vis_03_null_model.py", "n_samples": int(len(df)),
        "threshold_choice": "anisometropia: ref >3D (clinicamente significativa), cauda >5D (severa); "
                            "relativo >50% com ressalva (emetropia infla).",
        "rl_pearson_r": round(pearson, 4),
        "asymmetry_counts": {"real": real_c, "copula": cop_c, "discrete_level_control": disc_c},
        "extreme_cases_real_vs_copula_by_threshold": extreme_by_thr,
        "triple_overlap": {"audio13": ov_audio, "grip15": ov_grip, "vision_extreme_threshold_D": REF_D},
        "status": "EXECUTED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nVISÃO null model: N={len(df)}, OD-OS Pearson r={pearson:.3f}")
    print(f"{'threshold':>14} | {'REAL':>6} | {'CÓPULA':>7} | {'DISCRETO':>8}")
    for t in THRESHOLDS_D:
        print(f"  |OD-OS|>{t}D    | {real_c[f'n_abs_gt_{t}D']:>6} | {cop_c[f'n_abs_gt_{t}D']:>7} | {disc_c[f'n_abs_gt_{t}D']:>8}")
    print(f"  |OD-OS|>50%   | {real_c['n_rel_gt_50pct']:>6} | {cop_c['n_rel_gt_50pct']:>7} | {disc_c['n_rel_gt_50pct']:>8}")
    print(f"\n=== OVERLAP TRIPLO (visão extrema = >{REF_D}D) ===")
    print(f"  audio13: {ov_audio['n_with_valid_vision']}/13 com visão válida; "
          f"também extremos em visão: {ov_audio['n_also_vision_extreme_gt3D']} {ov_audio['seqn_also_extreme'] or ''}")
    print(f"  grip15:  {ov_grip['n_with_valid_vision']}/15 com visão válida; "
          f"também extremos em visão: {ov_grip['n_also_vision_extreme_gt3D']} {ov_grip['seqn_also_extreme'] or ''}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
