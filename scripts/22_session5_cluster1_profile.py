#!/usr/bin/env python3
"""
Nome: 22_session5_cluster1_profile.py
Tarefa: Profile individual das 12 pessoas do cluster 1.
Input: outputs/json/session4_assignments_any25.csv, data/processed/frequencia_feature_matrix_v1.csv
Output: outputs/json/22_cluster1_individual_profiles.json, outputs/json/22_cluster1_individual_profiles.csv
"""

import logging, json
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
FREQ_COLS = [
    "thr_R_500","thr_R_1000","thr_R_2000","thr_R_3000","thr_R_4000","thr_R_6000","thr_R_8000",
    "thr_L_500","thr_L_1000","thr_L_2000","thr_L_3000","thr_L_4000","thr_L_6000","thr_L_8000",
]
ASSIGN = Path("outputs/json/session4_assignments_any25.csv")
FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/22_cluster1_individual_profiles.json")
CSV_OUT = Path("outputs/json/22_cluster1_individual_profiles.csv")
LOG = Path("outputs/logs/22_cluster1_profile.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT.exists():
    log.info(f"Output já existe: {OUTPUT}. Pulando.")
    raise SystemExit(0)


def main():
    log.info("=" * 60)
    log.info("SESSÃO 5 — TAREFA 4: Profile do Cluster 1")
    log.info("=" * 60)

    assign = pd.read_csv(ASSIGN)
    df = pd.read_csv(FEATURE, low_memory=False)
    c1 = assign[assign["cluster_id"] == 1].copy()
    df = df[df["SEQN"].isin(c1["SEQN"])].copy()
    df = df.merge(c1, on="SEQN", how="inner", validate="one_to_one", suffixes=("", "_assign"))
    log.info(f"Cluster 1: {len(df)} pessoas")

    profiles = []
    csv_rows = []

    for _, row in df.iterrows():
        seqn = int(row["SEQN"])
        age = float(row["RIDAGEYR"]) if pd.notna(row["RIDAGEYR"]) else None
        sex = float(row["RIAGENDR"]) if pd.notna(row["RIAGENDR"]) else None
        cycle = str(row["cycle"])
        dist = float(row["distance_to_centroid"]) if pd.notna(row.get("distance_to_centroid")) else None
        out_score = float(row["outlier_score"]) if pd.notna(row.get("outlier_score")) else None
        mem_prob = float(row["membership_probability"]) if pd.notna(row.get("membership_probability")) else None

        # Thresholds
        thr = {}
        for col in FREQ_COLS:
            val = pd.to_numeric(row.get(col), errors="coerce")
            thr[col] = round(float(val), 1) if pd.notna(val) else None

        # Assimetria por frequência
        asym = {}
        for freq in [500, 1000, 2000, 3000, 4000, 6000, 8000]:
            r = thr.get(f"thr_R_{freq}")
            l = thr.get(f"thr_L_{freq}")
            if r is not None and l is not None:
                asym[f"asym_{freq}"] = round(abs(r - l), 1)

        # PTA
        r_high_vals = [thr.get(f"thr_R_{f}") for f in [3000, 4000, 6000, 8000]]
        r_low_vals = [thr.get(f"thr_R_{f}") for f in [500, 1000, 2000]]
        l_high_vals = [thr.get(f"thr_L_{f}") for f in [3000, 4000, 6000, 8000]]
        l_low_vals = [thr.get(f"thr_L_{f}") for f in [500, 1000, 2000]]
        r_high = round(float(np.nanmean(r_high_vals)), 1)
        r_low = round(float(np.nanmean(r_low_vals)), 1)
        l_high = round(float(np.nanmean(l_high_vals)), 1)
        l_low = round(float(np.nanmean(l_low_vals)), 1)
        asym_mean = round(float(np.nanmean(list(asym.values()))), 1) if asym else None

        # Flags
        n666 = int(pd.to_numeric(row.get("n_no_response_666_thresholds", 0), errors="coerce") or 0)
        n888 = int(pd.to_numeric(row.get("n_could_not_obtain_888_thresholds", 0), errors="coerce") or 0)

        # Tinnitus
        tin = pd.to_numeric(row.get("AUQ191"), errors="coerce") if "AUQ191" in row.index else None
        tin_label = "sim" if tin == 1 else ("não" if tin == 2 else None)

        profile = {
            "SEQN": seqn, "cycle": cycle, "age": age, "sex": sex,
            "distance_to_centroid": dist, "outlier_score": out_score, "membership_probability": mem_prob,
            "thresholds_db": thr, "asymmetry_by_freq": asym,
            "pta_low_R": r_low, "pta_low_L": l_low,
            "pta_high_R": r_high, "pta_high_L": l_high,
            "hf_lf_contrast_R": round(r_high - r_low, 1), "hf_lf_contrast_L": round(l_high - l_low, 1),
            "asym_mean_abs": asym_mean,
            "n_no_response_666": n666, "n_could_not_obtain_888": n888,
            "tinnitus_AUQ191": tin_label,
        }
        profiles.append(profile)

        # CSV row
        csv_row = {
            "SEQN": seqn, "cycle": cycle, "age": age, "sex": sex,
            "pta_low_R": r_low, "pta_low_L": l_low,
            "pta_high_R": r_high, "pta_high_L": l_high,
            "hf_lf_contrast_R": round(r_high - r_low, 1), "hf_lf_contrast_L": round(l_high - l_low, 1),
            "asym_mean": asym_mean,
        }
        for col in FREQ_COLS:
            csv_row[col] = thr[col]
        csv_row["tinnitus"] = tin_label
        csv_rows.append(csv_row)

        log.info(f"  SEQN={seqn}, age={age}, cycle={cycle}, R_high={r_high}, L_high={l_high}, asym={asym_mean}")

    # Summary
    ages = [p["age"] for p in profiles if p["age"] is not None]
    asym_vals = [p["asym_mean_abs"] for p in profiles if p["asym_mean_abs"] is not None]
    r_highs = [p["pta_high_R"] for p in profiles]
    l_highs = [p["pta_high_L"] for p in profiles]
    sexes = [p["sex"] for p in profiles if p["sex"] is not None]
    tins = [p["tinnitus_AUQ191"] for p in profiles if p["tinnitus_AUQ191"] is not None]

    summary = {
        "n": len(profiles),
        "age_median": round(float(np.median(ages)), 1) if ages else None,
        "age_range": [round(float(min(ages)), 1), round(float(max(ages)), 1)] if ages else None,
        "sex_distribution": {str(k): int(v) for k, v in pd.Series(sexes).value_counts().items()},
        "pta_high_R_mean": round(float(np.mean(r_highs)), 1),
        "pta_high_L_mean": round(float(np.mean(l_highs)), 1),
        "asym_mean_median": round(float(np.median(asym_vals)), 1) if asym_vals else None,
        "asym_mean_range": [round(float(min(asym_vals)), 1), round(float(max(asym_vals)), 1)] if asym_vals else None,
        "tinnitus_rate": round(sum(1 for t in tins if t == "sim") / len(tins), 4) if tins else None,
        "cycles": {str(k): int(v) for k, v in pd.Series([p["cycle"] for p in profiles]).value_counts().items()},
    }

    log.info(f"\nFINDING #CLUSTER1-PROFILE")
    log.info(f"n={summary['n']}, idade_mediana={summary['age_median']}")
    log.info(f"PTA_high_R_mean={summary['pta_high_R_mean']}, PTA_high_L_mean={summary['pta_high_L_mean']}")
    log.info(f"Assimetria mediana={summary['asym_mean_median']} dB")
    log.info(f"Status: PRELIMINAR — sem rótulo clínico")

    # Salvar
    result = {
        "script": "22_session5_cluster1_profile.py",
        "random_state": RANDOM_STATE,
        "summary": summary,
        "individual_profiles": profiles,
        "status": "EXECUTED — sem rótulo clínico",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(CSV_OUT, index=False)

    log.info(f"Concluído. Output: {OUTPUT}")
    log.info(f"CSV: {CSV_OUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
