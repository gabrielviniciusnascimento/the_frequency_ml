#!/usr/bin/env python3
"""
Nome: 36_triplet_overlap_audit.py
Tarefa: Formalizar — como artefato commitado, não mais como prosa — o overlap de SEQN
        entre os EXTREMOS dos três sistemas (audição, grip, visão), e a disjunção de
        calendário (NHANES cycles) que torna o overlap triplo impossível por desenho.
        Sustenta a afirmação do v6/meta_analysis: "não há confundimento intra-indivíduo;
        o gradiente é populacional, sistema-a-sistema, não dentro da mesma pessoa."

Input: data/processed/{frequencia_feature_matrix_v1.csv, grip_feature_matrix.csv,
       vis_feature_matrix.csv}
Output: outputs/json/36_triplet_overlap_audit.json
Run: .venv/Scripts/python.exe scripts/36_triplet_overlap_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _shape_space import load_cohort, R_COLS, L_COLS, lib_versions  # noqa: E402

OUT = ROOT / "outputs" / "json" / "36_triplet_overlap_audit.json"
Z_CUT = 4.0


def extreme_seqns(df: pd.DataFrame, contrast: np.ndarray) -> tuple[set, list]:
    z = np.abs(contrast) / np.nanstd(contrast)
    mask = z > Z_CUT
    seqns = set(int(s) for s in df.loc[mask, "SEQN"].tolist())
    cycles = sorted(set(str(c) for c in df.loc[mask, "cycle"].tolist()))
    return seqns, cycles


def main() -> None:
    # auditory (canonical cohort)
    da, thr = load_cohort()
    ca = thr[R_COLS].mean(axis=1).to_numpy() - thr[L_COLS].mean(axis=1).to_numpy()
    audio_seqn, audio_cycles = extreme_seqns(da, ca)

    # grip
    dg = pd.read_csv(ROOT / "data/processed/grip_feature_matrix.csv", low_memory=False)
    age = pd.to_numeric(dg["RIDAGEYR"], errors="coerce")
    dg = dg[(age >= 20) & (age <= 69)]
    dg = dg[dg["grip_R_max"].notna() & dg["grip_L_max"].notna()].copy()
    cg = dg["grip_R_max"].to_numpy(float) - dg["grip_L_max"].to_numpy(float)
    grip_seqn, grip_cycles = extreme_seqns(dg, cg)

    # vision
    dv = pd.read_csv(ROOT / "data/processed/vis_feature_matrix.csv", low_memory=False)
    age = pd.to_numeric(dv["RIDAGEYR"], errors="coerce")
    dv = dv[(age >= 20) & (age <= 69)]
    dv = dv[dv["vis_R"].notna() & dv["vis_L"].notna()].copy()
    cv = dv["vis_R"].to_numpy(float) - dv["vis_L"].to_numpy(float)
    vis_seqn, vis_cycles = extreme_seqns(dv, cv)

    # cohort membership overlap (who COULD be compared at all) vs extreme overlap
    audio_all = set(int(s) for s in da["SEQN"].tolist())
    grip_all = set(int(s) for s in dg["SEQN"].tolist())
    vis_all = set(int(s) for s in dv["SEQN"].tolist())

    out = {
        "script": "36_triplet_overlap_audit.py",
        "purpose": "Formalize cross-system SEQN overlap of extremes + calendar disjointness.",
        "z_cut": Z_CUT,
        "extremes": {
            "auditory": {"n": len(audio_seqn), "cycles": audio_cycles},
            "grip": {"n": len(grip_seqn), "cycles": grip_cycles},
            "vision": {"n": len(vis_seqn), "cycles": vis_cycles},
        },
        "cohort_membership_overlap": {
            "audio_and_grip": len(audio_all & grip_all),
            "audio_and_vision": len(audio_all & vis_all),
            "grip_and_vision": len(grip_all & vis_all),
            "all_three": len(audio_all & grip_all & vis_all),
        },
        "extreme_overlap": {
            "audio_and_grip": len(audio_seqn & grip_seqn),
            "audio_and_vision": len(audio_seqn & vis_seqn),
            "grip_and_vision": len(grip_seqn & vis_seqn),
            "all_three": len(audio_seqn & grip_seqn & vis_seqn),
        },
        "interpretation": (
            "If cohort_membership_overlap between a pair (or all three) is 0, the systems "
            "were measured in non-overlapping NHANES cycles, so a within-person cross-system "
            "test is impossible by design — the cross-system gradient is a population-level, "
            "system-level statement, not a within-individual claim."),
        "lib_versions": lib_versions(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"extremes |z|>4: audio={len(audio_seqn)} ({audio_cycles})")
    print(f"               grip={len(grip_seqn)} ({grip_cycles})")
    print(f"               vision={len(vis_seqn)} ({vis_cycles})")
    print(f"cohort membership overlap (all three) = {out['cohort_membership_overlap']['all_three']}")
    print(f"extreme overlap: {out['extreme_overlap']}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
