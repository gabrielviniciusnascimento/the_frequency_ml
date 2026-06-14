#!/usr/bin/env python3
"""
Nome: grip_01_ingest_anatomical.py
Tarefa: Ingerir MGX (grip) + DEMO e reconstruir DIREITA/ESQUERDA ANATÔMICA real
        via MGATHAND. Espelha scripts/01_ingest_aux.py (tratamento de código/missing).

PONTO QUE DECIDE TUDO — mapeamento anatômico (verificado no codebook MGX_H):
  MGXH1* = Hand 1 (1ª mão testada);  MGXH2* = Hand 2 (2ª mão testada) — ORDEM, não lado.
  MGATHAND = "Begin the test with this hand": 1=Right, 2=Left.
  => MGATHAND==1: Hand1=DIREITA, Hand2=ESQUERDA
     MGATHAND==2: Hand1=ESQUERDA, Hand2=DIREITA   (inversão)
  Se MGATHAND não ∈ {1,2}: NÃO atribuir lado — excluir + contar (não improvisar).

Força por mão = máximo dos 3 trials válidos (convenção NHANES; MGDCGSZ soma os máximos).
Output: data/processed/grip_feature_matrix.csv; outputs/json/grip_01_ingest_anatomical.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

RAW_ROOT = Path("data/raw/nhanes")
OUT_CSV = Path("data/processed/grip_feature_matrix.csv")
OUT_JSON = Path("outputs/json/grip_01_ingest_anatomical.json")
LOG = Path("outputs/logs/grip_01_ingest_anatomical.log")

CYCLES = {
    "2011-2012": {"year": "2011", "mgx": "MGX_G", "demo": "DEMO_G"},
    "2013-2014": {"year": "2013", "mgx": "MGX_H", "demo": "DEMO_H"},
}
H1 = ["MGXH1T1", "MGXH1T2", "MGXH1T3"]   # Hand 1 = primeira mão testada
H2 = ["MGXH2T1", "MGXH2T2", "MGXH2T3"]   # Hand 2 = segunda
GRIP_MIN_KG, GRIP_MAX_KG = 0.0, 150.0    # faixa plausível; codebook ~4-82 por trial

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def clean_kg(s):
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    return x.mask((x < GRIP_MIN_KG) | (x > GRIP_MAX_KG), np.nan)


def main():
    frames = []
    per_cycle = {}
    for cycle, spec in CYCLES.items():
        mgx = pd.read_sas(RAW_ROOT / spec["year"] / f"{spec['mgx']}.xpt", format="xport")
        demo = pd.read_sas(RAW_ROOT / spec["year"] / f"{spec['demo']}.xpt", format="xport")
        mgx["SEQN"] = mgx["SEQN"].astype("int64")
        demo["SEQN"] = demo["SEQN"].astype("int64")
        df = mgx.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR"]], on="SEQN", how="left")
        df["cycle"] = cycle

        for c in H1 + H2:
            df[c] = clean_kg(df[c])
        df["MGATHAND"] = pd.to_numeric(df["MGATHAND"], errors="coerce")

        # ── Mapeamento anatômico via MGATHAND ────────────────────────
        valid = df["MGATHAND"].isin([1.0, 2.0])
        n_excluded_mgathand = int((~valid).sum())
        mgathand_dist = df["MGATHAND"].value_counts(dropna=False).to_dict()
        log.info(f"{cycle}: N={len(df)}; MGATHAND válido={int(valid.sum())}; "
                 f"excluídos (MGATHAND ausente/inválido)={n_excluded_mgathand}; dist={mgathand_dist}")

        df = df[valid].copy()
        right_first = df["MGATHAND"] == 1.0   # Hand1=R; senão Hand1=L
        # Direita anatômica = H1 se right_first, senão H2 ; Esquerda = o oposto
        for i, (h1c, h2c) in enumerate(zip(H1, H2), start=1):
            df[f"grip_R_t{i}"] = np.where(right_first, df[h1c], df[h2c])
            df[f"grip_L_t{i}"] = np.where(right_first, df[h2c], df[h1c])

        rcols = [f"grip_R_t{i}" for i in (1, 2, 3)]
        lcols = [f"grip_L_t{i}" for i in (1, 2, 3)]
        df["grip_R_max"] = df[rcols].max(axis=1, skipna=True)
        df["grip_L_max"] = df[lcols].max(axis=1, skipna=True)
        df["asym_RL"] = df["grip_R_max"] - df["grip_L_max"]

        keep_cols = (["SEQN", "cycle", "RIDAGEYR", "RIAGENDR", "MGATHAND", "MGAPHAND",
                      "MGDCGSZ"] + H1 + H2 + rcols + lcols
                     + ["grip_R_max", "grip_L_max", "asym_RL"])
        keep_cols = [c for c in keep_cols if c in df.columns]
        frames.append(df[keep_cols])
        per_cycle[cycle] = {
            "n_total": int(len(mgx)),
            "n_mgathand_valid": int(valid.sum()),
            "n_excluded_mgathand": n_excluded_mgathand,
            "mgathand_distribution": {str(k): int(v) for k, v in mgathand_dist.items()},
            "n_right_first": int((df["MGATHAND"] == 1.0).sum()),
            "n_left_first": int((df["MGATHAND"] == 2.0).sum()),
            "n_both_hands_valid": int((df["grip_R_max"].notna() & df["grip_L_max"].notna()).sum()),
        }

    full = pd.concat(frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_CSV, index=False)

    # ── 5 linhas de exemplo: misturar MGATHAND==1 e ==2 p/ checar inversão ──
    ex1 = full[full["MGATHAND"] == 1.0].head(3)
    ex2 = full[full["MGATHAND"] == 2.0].head(2)
    examples = pd.concat([ex1, ex2])
    ex_records = []
    print("\n=== 5 LINHAS DE EXEMPLO (conferir mapeamento anatômico) ===")
    print(f"{'SEQN':>7} {'MGATHAND':>8} | {'H1(t1,t2,t3)':>22} {'H2(t1,t2,t3)':>22} | "
          f"{'lado de H1':>10} | {'R_max':>6} {'L_max':>6}")
    for _, r in examples.iterrows():
        h1 = [None if pd.isna(r[c]) else round(float(r[c]), 1) for c in H1]
        h2 = [None if pd.isna(r[c]) else round(float(r[c]), 1) for c in H2]
        side_h1 = "DIREITA" if r["MGATHAND"] == 1.0 else "ESQUERDA"
        print(f"{int(r['SEQN']):>7} {int(r['MGATHAND']):>8} | {str(h1):>22} {str(h2):>22} | "
              f"{side_h1:>10} | {r['grip_R_max']:>6.1f} {r['grip_L_max']:>6.1f}")
        ex_records.append({
            "SEQN": int(r["SEQN"]), "MGATHAND": int(r["MGATHAND"]),
            "MGXH1_t1t2t3": h1, "MGXH2_t1t2t3": h2,
            "hand1_anatomical_side": side_h1,
            "grip_R_max": round(float(r["grip_R_max"]), 1),
            "grip_L_max": round(float(r["grip_L_max"]), 1),
        })

    result = {
        "script": "grip_01_ingest_anatomical.py",
        "mapping_rule": "MGATHAND==1 -> Hand1=Right,Hand2=Left; MGATHAND==2 -> inverted. "
                        "Per-hand strength = max of 3 trials. Source: MGX_H codebook.",
        "n_total_rows": int(len(full)),
        "per_cycle": per_cycle,
        "example_rows": ex_records,
        "output_csv": str(OUT_CSV),
        "status": "EXECUTED — STOP-GATE: aguardar revisão do mapeamento antes dos passos 3-4",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Output: {OUT_JSON} | CSV: {OUT_CSV} | N={len(full)}")
    print(f"\nTotal com lado anatômico atribuído: N={len(full)}")
    for c, s in per_cycle.items():
        print(f"  {c}: válidos={s['n_mgathand_valid']}, excluídos MGATHAND={s['n_excluded_mgathand']}, "
              f"R-first={s['n_right_first']}, L-first={s['n_left_first']}, ambas-mãos={s['n_both_hands_valid']}")


if __name__ == "__main__":
    main()
