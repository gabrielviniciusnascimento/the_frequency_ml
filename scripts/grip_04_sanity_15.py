#!/usr/bin/env python3
"""
Nome: grip_04_sanity_15.py
Tarefa: (1) Sanity-check dos SEQN com |R-L|>20kg em grip (análogo ao Bloco 1 da audição):
            trials brutos por mão + flags *E de exclusão; assimetria consistente entre
            trials vs. outlier de trial único.
        (2) Overlap audição×grip: dos 13 SEQN auditivos, quantos têm grip válido, e
            desses quantos são ALSO assimétricos em grip (|R-L|>20kg).
Lê do disco: outputs/json/grip_03_null_model.json, outputs/json/27_*.json,
data/processed/grip_feature_matrix.csv, e os XPT brutos (para as flags *E).
Output: outputs/json/grip_04_sanity_15.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/nhanes")
NULL = Path("outputs/json/grip_03_null_model.json")
AUDIO = Path("outputs/json/27_binaural_pooling_ablation.json")
MATRIX = Path("data/processed/grip_feature_matrix.csv")
OUT = Path("outputs/json/grip_04_sanity_15.json")

H1 = ["MGXH1T1", "MGXH1T2", "MGXH1T3"]
H2 = ["MGXH2T1", "MGXH2T2", "MGXH2T3"]
H1E = ["MGXH1T1E", "MGXH1T2E", "MGXH1T3E"]
H2E = ["MGXH2T1E", "MGXH2T2E", "MGXH2T3E"]
XPTS = [("2011", "MGX_G"), ("2013", "MGX_H")]


def load_raw_mgx():
    parts = []
    for year, f in XPTS:
        d = pd.read_sas(RAW / year / f"{f}.xpt", format="xport")
        d["SEQN"] = d["SEQN"].astype("int64")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    seqn15 = json.loads(NULL.read_text(encoding="utf-8"))[
        "extreme_cases_real_vs_copula_by_threshold"]["gt_20kg"]["seqn_real"]
    raw = load_raw_mgx().set_index("SEQN")

    print(f"=== SANITY CHECK: {len(seqn15)} SEQN com |R-L|>20kg ===\n")
    cases = []
    n_consistent = 0
    n_single_outlier = 0
    n_with_excl_flag = 0
    for s in seqn15:
        r = raw.loc[s]
        mgathand = float(r["MGATHAND"])
        h1 = [float(r[c]) if pd.notna(r[c]) else None for c in H1]
        h2 = [float(r[c]) if pd.notna(r[c]) else None for c in H2]
        h1e = [None if pd.isna(r[c]) else int(r[c]) for c in H1E]
        h2e = [None if pd.isna(r[c]) else int(r[c]) for c in H2E]
        # mapeamento anatômico
        if mgathand == 1.0:
            R, L, RE, LE = h1, h2, h1e, h2e
        else:
            R, L, RE, LE = h2, h1, h2e, h1e
        Rv = [x for x in R if x is not None]
        Lv = [x for x in L if x is not None]
        r_max, l_max = max(Rv), max(Lv)
        weak, strong = ("R", "L") if r_max < l_max else ("L", "R")
        weak_trials = Rv if weak == "R" else Lv
        strong_trials = Rv if strong == "R" else Lv
        weak_max, strong_min = max(weak_trials), min(strong_trials)
        weak_spread = round(max(weak_trials) - min(weak_trials), 1)
        # consistência: mão fraca no seu MELHOR ainda abaixo da forte no PIOR -> separação total
        fully_separated = weak_max < strong_min
        # outlier de trial único: 1 trial destoa muito (>10kg) dos outros 2 na mão fraca
        wt = sorted(weak_trials)
        single_outlier = (len(wt) == 3) and ((wt[1] - wt[0]) > 10 and (wt[2] - wt[1]) < 5)
        # *E é flag de ESFORÇO: 1=máximo (bom), 2=questionável. Contar só os ==2.
        questionable = int(sum(1 for e in (RE + LE) if e == 2))
        if questionable:
            n_with_excl_flag += 1
        if single_outlier:
            n_single_outlier += 1
        else:
            n_consistent += 1
        verdict = "single_trial_outlier" if single_outlier else "consistente"
        cases.append({
            "seqn": int(s), "MGATHAND": int(mgathand),
            "R_trials": [round(x, 1) for x in Rv], "L_trials": [round(x, 1) for x in Lv],
            "R_effort_flags": RE, "L_effort_flags": LE,
            "R_max": round(r_max, 1), "L_max": round(l_max, 1),
            "abs_diff": round(abs(r_max - l_max), 1),
            "weak_hand": weak, "weak_hand_spread": weak_spread,
            "weak_max_lt_strong_min": bool(fully_separated),
            "n_questionable_effort_trials": questionable,
            "verdict": verdict,
        })
        print(f"SEQN {s} (MGATHAND={int(mgathand)}): R={[round(x,1) for x in Rv]} L={[round(x,1) for x in Lv]} "
              f"| diff={abs(r_max-l_max):.1f} weak={weak} spread={weak_spread} "
              f"sep={'TOTAL' if fully_separated else 'overlap'} questionable_effort={questionable} -> {verdict}")

    print(f"\nConsistentes: {n_consistent}/{len(seqn15)} | single-trial-outlier: {n_single_outlier} "
          f"| com >=1 trial de esforço questionável (*E==2): {n_with_excl_flag}")
    fully_sep = sum(c["weak_max_lt_strong_min"] for c in cases)
    print(f"Separação total (weak_max < strong_min): {fully_sep}/{len(seqn15)}")

    # ── (2) Overlap audição × grip ───────────────────────────────────
    audio13 = json.loads(AUDIO.read_text(encoding="utf-8"))[
        "arm_A_14d_separate_ears"]["asymmetry_cluster"]["seqn"]
    m = pd.read_csv(MATRIX, low_memory=False)
    m["SEQN"] = pd.to_numeric(m["SEQN"], errors="coerce").astype("Int64")
    grip_valid = m[m["grip_R_max"].notna() & m["grip_L_max"].notna()].set_index("SEQN")
    has_grip, also_asym = [], []
    detail = []
    for s in audio13:
        if s in grip_valid.index:
            rr = grip_valid.loc[s]
            diff = abs(float(rr["grip_R_max"]) - float(rr["grip_L_max"]))
            has_grip.append(s)
            detail.append({"seqn": int(s), "grip_R_max": round(float(rr["grip_R_max"]), 1),
                           "grip_L_max": round(float(rr["grip_L_max"]), 1), "abs_diff_kg": round(diff, 1),
                           "grip_asym_gt20kg": bool(diff > 20)})
            if diff > 20:
                also_asym.append(s)

    print(f"\n=== OVERLAP AUDIÇÃO × GRIP ===")
    print(f"13 SEQN auditivos -> com grip válido (R_max+L_max): {len(has_grip)} {has_grip}")
    for d in detail:
        print(f"  SEQN {d['seqn']}: gripR={d['grip_R_max']} gripL={d['grip_L_max']} diff={d['abs_diff_kg']}kg asym>20={d['grip_asym_gt20kg']}")
    print(f"Desses, ALSO assimétricos em grip (|R-L|>20kg): {len(also_asym)} {also_asym or 'nenhum'}")

    OUT.write_text(json.dumps({
        "script": "grip_04_sanity_15.py",
        "sanity_15": {"n": len(seqn15), "n_consistent": n_consistent,
                      "n_single_trial_outlier": n_single_outlier,
                      "n_fully_separated_weak_lt_strong": fully_sep,
                      "effort_flag_note": "*E: 1=esforço máximo (bom), 2=questionável",
                      "n_with_questionable_effort_trial": n_with_excl_flag, "cases": cases},
        "overlap": {"audio13": audio13, "n_with_valid_grip": len(has_grip),
                    "seqn_with_valid_grip": has_grip, "grip_detail": detail,
                    "n_also_grip_asymmetric_gt20kg": len(also_asym), "seqn_also_asym": also_asym},
        "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
