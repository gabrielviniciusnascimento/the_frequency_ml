#!/usr/bin/env python3
"""
Nome: 29_sanity_check_13.py
Tarefa: Sanity check clínico dos 13 SEQN do cluster de assimetria. Confirmar que
        o R alto é medição real (não ceiling-pin nem código de não-testado vazado),
        per-frequency, e contar flags 666/888 por linha.

Lê SEQN de outputs/json/27_binaural_pooling_ablation.json.
Lê data/processed/frequencia_feature_matrix_v1.csv (colunas LIMPAS thr_* já têm
666/888/out-of-range -> NaN, ver scripts/01_ingest_aux.py:96-101).

Output: outputs/json/29_sanity_check_13.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R_COLS = [f"thr_R_{f}" for f in FREQS]
L_COLS = [f"thr_L_{f}" for f in FREQS]
R_FLAGS_NR = [f"thr_R_{f}_no_response" for f in FREQS]
R_FLAGS_CNO = [f"thr_R_{f}_could_not_obtain" for f in FREQS]
L_FLAGS_NR = [f"thr_L_{f}_no_response" for f in FREQS]
L_FLAGS_CNO = [f"thr_L_{f}_could_not_obtain" for f in FREQS]

FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
ABLATION = Path("outputs/json/27_binaural_pooling_ablation.json")
OUTPUT = Path("outputs/json/29_sanity_check_13.json")

INSTRUMENT_CEILING_DB = 120.0  # teto típico do audiômetro NHANES


def main():
    seqn13 = json.loads(ABLATION.read_text(encoding="utf-8"))[
        "arm_A_14d_separate_ears"]["asymmetry_cluster"]["seqn"]
    df = pd.read_csv(FEATURE, low_memory=False)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
    sub = df[df["SEQN"].isin(seqn13)].copy()
    print(f"Casados {len(sub)}/{len(seqn13)} SEQN")

    have_flags = all(c in df.columns for c in R_FLAGS_NR + R_FLAGS_CNO)

    per_case = []
    n_genuine = 0
    for _, row in sub.iterrows():
        r_vals = [None if pd.isna(row[c]) else round(float(row[c]), 0) for c in R_COLS]
        l_vals = [None if pd.isna(row[c]) else round(float(row[c]), 0) for c in L_COLS]
        r_valid = [v for v in r_vals if v is not None]
        l_valid = [v for v in l_vals if v is not None]

        nr_count = int(sum(int(row[c]) for c in R_FLAGS_NR if c in df.columns)) if have_flags else None
        cno_count = int(sum(int(row[c]) for c in R_FLAGS_CNO if c in df.columns)) if have_flags else None
        l_nr = int(sum(int(row[c]) for c in L_FLAGS_NR if c in df.columns)) if have_flags else None

        # ceiling-pin: >=5 das freqs R válidas no mesmo valor no teto?
        ceiling_pinned = False
        if r_valid:
            at_ceiling = [v for v in r_valid if v >= INSTRUMENT_CEILING_DB]
            ceiling_pinned = len(at_ceiling) >= 5
        # variação fisiológica do R
        r_spread = round(float(np.ptp(r_valid)), 0) if len(r_valid) >= 2 else 0.0

        # critério genuíno: R varia, >=4 freqs R válidas, não ceiling-pinned,
        # e não predominantemente sustentado por 666 (>=3 freqs perdidas para código)
        coded_lost_R = (nr_count or 0) + (cno_count or 0)
        genuine = (len(r_valid) >= 4 and not ceiling_pinned and r_spread >= 10
                   and coded_lost_R < 3)
        n_genuine += int(genuine)

        per_case.append({
            "seqn": int(row["SEQN"]),
            "R_per_freq_db": dict(zip([str(f) for f in FREQS], r_vals)),
            "L_per_freq_db": dict(zip([str(f) for f in FREQS], l_vals)),
            "R_pta": round(float(np.mean(r_valid)), 1) if r_valid else None,
            "L_pta": round(float(np.mean(l_valid)), 1) if l_valid else None,
            "R_minus_L": round(float(np.mean(r_valid) - np.mean(l_valid)), 1) if (r_valid and l_valid) else None,
            "n_valid_R": len(r_valid),
            "n_valid_L": len(l_valid),
            "n_valid_total": len(r_valid) + len(l_valid),
            "R_666_no_response_flags": nr_count,
            "R_888_could_not_obtain_flags": cno_count,
            "L_666_no_response_flags": l_nr,
            "R_spread_db": r_spread,
            "R_ceiling_pinned": ceiling_pinned,
            "verdict_genuine": genuine,
        })

    for c in per_case:
        print(f"SEQN {c['seqn']}: R={list(c['R_per_freq_db'].values())} "
              f"(PTA {c['R_pta']}, spread {c['R_spread_db']}) | "
              f"L PTA {c['L_pta']} | validR={c['n_valid_R']} validTot={c['n_valid_total']} | "
              f"666R={c['R_666_no_response_flags']} 888R={c['R_888_could_not_obtain_flags']} | "
              f"genuine={c['verdict_genuine']}")

    summary = {
        "n_cases": len(per_case),
        "n_genuine": n_genuine,
        "n_flagged_artifact": len(per_case) - n_genuine,
        "any_ceiling_pinned": any(c["R_ceiling_pinned"] for c in per_case),
        "max_R_666_flags": max((c["R_666_no_response_flags"] or 0) for c in per_case),
        "max_R_888_flags": max((c["R_888_could_not_obtain_flags"] or 0) for c in per_case),
        "min_n_valid_total": min(c["n_valid_total"] for c in per_case),
    }
    verdict = (
        "13/13 GENUÍNOS — perda interaural real no instrumento; nenhum artifact de coleta"
        if n_genuine == len(per_case) else
        f"{n_genuine}/{len(per_case)} genuínos; {len(per_case)-n_genuine} sinalizados — checar se modo sobrevive"
    )
    print(f"\nRESUMO: {summary}")
    print(f"VEREDITO: {verdict}")

    OUTPUT.write_text(json.dumps({
        "script": "29_sanity_check_13.py",
        "seqn_checked": seqn13,
        "instrument_ceiling_db": INSTRUMENT_CEILING_DB,
        "per_case": per_case,
        "summary": summary,
        "verdict": verdict,
        "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
