#!/usr/bin/env python3
"""
Nome: vis_01_ingest.py
Tarefa: Ingerir VIX/VIX_B (autorefração) + DEMO e construir equivalente esférico
        bilateral OD/OS. Espelha grip_01 (clean de código especial; OD/OS são
        anatômicos diretos, SEM ambiguidade tipo MGATHAND).

Equivalente esférico (SE) = esfera + cilindro/2:
  vis_R (OD) = VIXORSM + VIXORCM/2 ; vis_L (OS) = VIXOLSM + VIXOLCM/2
Códigos especiais (codebook): 88 = could not obtain -> NaN; fora de faixa clínica -> NaN.
Confiança VIDORFM/VIDOLFM (5-9; maior=melhor) preservada para referência.
Output: data/processed/vis_feature_matrix.csv; outputs/json/vis_01_ingest.json
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/nhanes")
OUT_CSV = Path("data/processed/vis_feature_matrix.csv")
OUT_JSON = Path("outputs/json/vis_01_ingest.json")
LOG = Path("outputs/logs/vis_01_ingest.log")

CYCLES = {
    "1999-2000": {"year": "1999", "vix": "VIX", "demo": "DEMO"},
    "2001-2002": {"year": "2001", "vix": "VIX_B", "demo": "DEMO_B"},
}
COULD_NOT_OBTAIN = 88.0
SPH_MIN, SPH_MAX = -25.0, 25.0   # dioptrias plausíveis
CYL_MIN, CYL_MAX = -15.0, 15.0

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger(__name__)


def clean(s, lo, hi):
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    x = x.mask(x == COULD_NOT_OBTAIN, np.nan)
    return x.mask((x < lo) | (x > hi), np.nan)


def main():
    frames = []
    per_cycle = {}
    for cycle, spec in CYCLES.items():
        vix = pd.read_sas(RAW / spec["year"] / f"{spec['vix']}.xpt", format="xport")
        demo = pd.read_sas(RAW / spec["year"] / f"{spec['demo']}.xpt", format="xport")
        vix["SEQN"] = vix["SEQN"].astype("int64")
        demo["SEQN"] = demo["SEQN"].astype("int64")
        df = vix.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR"]], on="SEQN", how="left")
        df["cycle"] = cycle

        r_sph = clean(df["VIXORSM"], SPH_MIN, SPH_MAX)
        r_cyl = clean(df["VIXORCM"], CYL_MIN, CYL_MAX)
        l_sph = clean(df["VIXOLSM"], SPH_MIN, SPH_MAX)
        l_cyl = clean(df["VIXOLCM"], CYL_MIN, CYL_MAX)
        n88 = int(((pd.to_numeric(df["VIXORSM"], errors="coerce") == 88) |
                   (pd.to_numeric(df["VIXOLSM"], errors="coerce") == 88)).sum())

        out = pd.DataFrame({
            "SEQN": df["SEQN"], "cycle": cycle,
            "RIDAGEYR": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
            "RIAGENDR": pd.to_numeric(df["RIAGENDR"], errors="coerce"),
            "OD_sphere": r_sph, "OD_cyl": r_cyl, "OS_sphere": l_sph, "OS_cyl": l_cyl,
            "VIDORFM": pd.to_numeric(df["VIDORFM"], errors="coerce"),
            "VIDOLFM": pd.to_numeric(df["VIDOLFM"], errors="coerce"),
        })
        out["vis_R"] = out["OD_sphere"] + out["OD_cyl"] / 2.0   # SE OD
        out["vis_L"] = out["OS_sphere"] + out["OS_cyl"] / 2.0   # SE OS
        out["asym_RL"] = out["vis_R"] - out["vis_L"]
        frames.append(out)

        both = out["vis_R"].notna() & out["vis_L"].notna()
        per_cycle[cycle] = {
            "n_total": int(len(vix)),
            "n_both_eyes_valid": int(both.sum()),
            "n_could_not_obtain_88": n88,
            "median_SE_OD": round(float(out["vis_R"].median()), 2),
            "median_SE_OS": round(float(out["vis_L"].median()), 2),
        }
        log.info(f"{cycle}: N={len(vix)}, ambos olhos válidos={int(both.sum())}, 88(could not obtain)={n88}")

    full = pd.concat(frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_CSV, index=False)

    # 5 linhas de exemplo (ambos olhos válidos, mix de ciclos)
    valid = full[full["vis_R"].notna() & full["vis_L"].notna()]
    examples = pd.concat([valid.head(3), valid.tail(2)])
    ex_records = []
    print("\n=== 5 LINHAS DE EXEMPLO (SE = esfera + cilindro/2) ===")
    print(f"{'SEQN':>7} {'cycle':>10} | {'OD(sph,cyl)->SE':>22} {'OS(sph,cyl)->SE':>22} | {'asym R-L':>9}")
    for _, r in examples.iterrows():
        print(f"{int(r['SEQN']):>7} {r['cycle']:>10} | "
              f"({r['OD_sphere']:.2f},{r['OD_cyl']:.2f})->{r['vis_R']:.2f}".rjust(22) + " " +
              f"({r['OS_sphere']:.2f},{r['OS_cyl']:.2f})->{r['vis_L']:.2f}".rjust(22) +
              f" | {r['asym_RL']:>9.2f}")
        ex_records.append({"SEQN": int(r["SEQN"]), "cycle": r["cycle"],
                           "OD_sphere": round(float(r["OD_sphere"]), 2), "OD_cyl": round(float(r["OD_cyl"]), 2),
                           "OS_sphere": round(float(r["OS_sphere"]), 2), "OS_cyl": round(float(r["OS_cyl"]), 2),
                           "vis_R_SE": round(float(r["vis_R"]), 2), "vis_L_SE": round(float(r["vis_L"]), 2),
                           "asym_RL": round(float(r["asym_RL"]), 2)})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "script": "vis_01_ingest.py",
        "spherical_equivalent_rule": "SE = sphere + cylinder/2; vis_R=OD, vis_L=OS (anatômicos diretos)",
        "special_codes": "88=could not obtain -> NaN; fora de faixa -> NaN",
        "n_total_rows": int(len(full)), "per_cycle": per_cycle, "example_rows": ex_records,
        "output_csv": str(OUT_CSV),
        "status": "EXECUTED — STOP-GATE: revisar 5 linhas antes dos passos 3-5",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal: N={len(full)} | por ciclo:")
    for c, s in per_cycle.items():
        print(f"  {c}: ambos olhos válidos={s['n_both_eyes_valid']}, 88={s['n_could_not_obtain_88']}, "
              f"mediana SE OD={s['median_SE_OD']}, OS={s['median_SE_OS']}")


if __name__ == "__main__":
    main()
