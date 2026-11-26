

#!/usr/bin/env python3
"""
Nome: vis_01_ingest.py
Tarefa: Ingerir VIX/VIX_B/VIX_C (1999-2004: autorefração + acuidade visual) + DEMO,
        zerando underflow IBM float na leitura, construindo equivalente esférico
        bilateral OD/OS e convertendo VIDRVA/VIDLVA/VIDROVA/VIDLOVA para logMAR.

Equivalente esférico (SE) = esfera + cilindro/2:
  vis_R (OD) = VIXORSM + VIXORCM/2 ; vis_L (OS) = VIXOLSM + VIXOLCM/2
Acuidade visual (codebooks VIX/VIX_B/VIX_C verificados via script 38):
  Denominadores Snellen válidos: {20, 25, 30, 40, 50, 60, 80, 200} -> log10(code / 20)
  (Em VIDROVA/VIDLOVA o menor código é 25 = 20/25, medido apenas se VIDxVA >= 30).
  Códigos especiais de acuidade: 666 = 20/200+ (censura à direita -> NaN + flag);
                                 888 = Could not obtain (-> NaN + flag).
Confiança VIDORFM/VIDOLFM (5-9; maior=melhor) preservada para referência.
Output: data/processed/vis_feature_matrix.csv; outputs/json/vis_01_ingest.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/nhanes")
OUT_CSV = Path("data/processed/vis_feature_matrix.csv")          # congelado: só 1999-2002 (coorte do paper v6, N=7.057)
OUT_CSV_V2 = Path("data/processed/vis_feature_matrix_v2.csv")    # completo: 1999-2004 + acuidade
PAPER_V6_CYCLES = ("1999-2000", "2001-2002")
OUT_JSON = Path("outputs/json/vis_01_ingest.json")
LOG = Path("outputs/logs/vis_01_ingest.log")

CYCLES = {
    "1999-2000": {"year": "1999", "vix": "VIX", "demo": "DEMO"},
    "2001-2002": {"year": "2001", "vix": "VIX_B", "demo": "DEMO_B"},
    "2003-2004": {"year": "2003", "vix": "VIX_C", "demo": "DEMO_C"},
}
COULD_NOT_OBTAIN = 88.0
SPH_MIN, SPH_MAX = -25.0, 25.0   # dioptrias plausíveis
CYL_MIN, CYL_MAX = -15.0, 15.0
SAS_ZERO_EPS = 1e-12

VALID_SNELLEN_PRESENTING = frozenset({20, 25, 30, 40, 50, 60, 80, 200})
VALID_SNELLEN_OBJ_REFR = frozenset({25, 30, 40, 50, 60, 80, 200})
VA_WORSE_THAN_20_200_CODE = 666
VA_COULD_NOT_OBTAIN_CODE = 888


def scrub_ibm_underflow(df: pd.DataFrame, eps: float = SAS_ZERO_EPS) -> pd.DataFrame:
    """Zera resíduos de ponto flutuante IBM XPORT (~5.3976e-79) em todas as colunas numéricas."""
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        vals = out[num_cols].to_numpy(copy=True)
        vals[np.isfinite(vals) & (np.abs(vals) < eps)] = 0.0
        out[num_cols] = vals
    return out


def read_sas_xpt(path: Path) -> pd.DataFrame:
    return scrub_ibm_underflow(pd.read_sas(path, format="xport"))


def snellen_code_to_logmar(
    series: pd.Series,
    *,
    allow_20: bool = True,
    code_666_logmar: float | None = None,
    strict: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Converte códigos inteiros de denominador Snellen do NHANES VIX para logMAR = log10(code / 20).
    Retorna (logmar_float64, flag_gt200_666_int8, flag_could_not_obtain_888_int8).
    """
    raw = pd.to_numeric(series, errors="coerce")
    valid_codes = VALID_SNELLEN_PRESENTING if allow_20 else VALID_SNELLEN_OBJ_REFR

    non_na = raw.dropna()
    int_mask = np.isclose(non_na, np.round(non_na), atol=1e-6)
    if strict and not bool(np.all(int_mask)):
        bad_float = sorted(set(non_na[~int_mask].tolist()))
        raise ValueError(f"Códigos de acuidade não-inteiros encontrados: {bad_float}")

    rounded = raw.round()
    allowed_all = valid_codes | {VA_WORSE_THAN_20_200_CODE, VA_COULD_NOT_OBTAIN_CODE}
    unexpected = rounded.notna() & ~rounded.isin(allowed_all)
    if strict and bool(unexpected.any()):
        bad_codes = sorted(set(rounded[unexpected].astype(int).tolist()))
        raise ValueError(f"Códigos de acuidade fora do codebook (allow_20={allow_20}): {bad_codes}")

    is_666 = rounded == VA_WORSE_THAN_20_200_CODE
    is_888 = rounded == VA_COULD_NOT_OBTAIN_CODE
    is_snellen = rounded.isin(valid_codes)

    logmar = pd.Series(np.nan, index=series.index, dtype="float64")
    logmar.loc[is_snellen] = np.log10(rounded.loc[is_snellen].astype("float64") / 20.0)
    if code_666_logmar is not None:
        logmar.loc[is_666] = float(code_666_logmar)

    return logmar, is_666.astype("int8"), is_888.astype("int8")


def clean(s: pd.Series, lo: float, hi: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    x = x.mask(x == COULD_NOT_OBTAIN, np.nan)
    return x.mask((x < lo) | (x > hi), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingestão VIX 1999-2004 (autorefração + acuidade em logMAR).")
    ap.add_argument("--force", action="store_true", help="Reexecutar mesmo se o JSON de saída existir.")
    args = ap.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    if OUT_JSON.exists() and not args.force:
        log.info(f"Output já existe em {OUT_JSON}. Use --force para regerar. Pulando.")
        raise SystemExit(0)

    frames = []
    per_cycle = {}
    for cycle, spec in CYCLES.items():
        vix = read_sas_xpt(RAW / spec["year"] / f"{spec['vix']}.xpt")
        demo = read_sas_xpt(RAW / spec["year"] / f"{spec['demo']}.xpt")
        vix["SEQN"] = vix["SEQN"].astype("int64")
        demo["SEQN"] = demo["SEQN"].astype("int64")
        df = vix.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR"]], on="SEQN", how="left")
        df["cycle"] = cycle

        r_sph = clean(df["VIXORSM"], SPH_MIN, SPH_MAX)
        r_cyl = clean(df["VIXORCM"], CYL_MIN, CYL_MAX)
        l_sph = clean(df["VIXOLSM"], SPH_MIN, SPH_MAX)
        l_cyl = clean(df["VIXOLCM"], CYL_MIN, CYL_MAX)
        n88 = int(
            (
                (pd.to_numeric(df["VIXORSM"], errors="coerce") == COULD_NOT_OBTAIN)
                | (pd.to_numeric(df["VIXOLSM"], errors="coerce") == COULD_NOT_OBTAIN)
            ).sum()
        )

        r_va_logmar, r_va_666, r_va_888 = snellen_code_to_logmar(df["VIDRVA"], allow_20=True)
        l_va_logmar, l_va_666, l_va_888 = snellen_code_to_logmar(df["VIDLVA"], allow_20=True)
        ro_va_logmar, ro_va_666, ro_va_888 = snellen_code_to_logmar(df["VIDROVA"], allow_20=False)
        lo_va_logmar, lo_va_666, lo_va_888 = snellen_code_to_logmar(df["VIDLOVA"], allow_20=False)

        vidrva_raw = pd.to_numeric(df["VIDRVA"], errors="coerce")
        vidlva_raw = pd.to_numeric(df["VIDLVA"], errors="coerce")
        vidrova_raw = pd.to_numeric(df["VIDROVA"], errors="coerce")
        vidlova_raw = pd.to_numeric(df["VIDLOVA"], errors="coerce")

        out = pd.DataFrame({
            "SEQN": df["SEQN"],
            "cycle": cycle,
            "RIDAGEYR": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
            "RIAGENDR": pd.to_numeric(df["RIAGENDR"], errors="coerce"),
            "OD_sphere": r_sph,
            "OD_cyl": r_cyl,
            "OS_sphere": l_sph,
            "OS_cyl": l_cyl,
            "VIDORFM": pd.to_numeric(df["VIDORFM"], errors="coerce"),
            "VIDOLFM": pd.to_numeric(df["VIDOLFM"], errors="coerce"),
            "vis_R": r_sph + r_cyl / 2.0,
            "vis_L": l_sph + l_cyl / 2.0,
            "asym_RL": (r_sph + r_cyl / 2.0) - (l_sph + l_cyl / 2.0),
            "VIDRVA": vidrva_raw,
            "VIDLVA": vidlva_raw,
            "VIDROVA": vidrova_raw,
            "VIDLOVA": vidlova_raw,
            "VIDRVA_logmar": r_va_logmar,
            "VIDLVA_logmar": l_va_logmar,
            "VIDROVA_logmar": ro_va_logmar,
            "VIDLOVA_logmar": lo_va_logmar,
            "VIDRVA_gt200_666": r_va_666,
            "VIDLVA_gt200_666": l_va_666,
            "VIDROVA_gt200_666": ro_va_666,
            "VIDLOVA_gt200_666": lo_va_666,
            "VIDRVA_could_not_obtain_888": r_va_888,
            "VIDLVA_could_not_obtain_888": l_va_888,
            "VIDROVA_could_not_obtain_888": ro_va_888,
            "VIDLOVA_could_not_obtain_888": lo_va_888,
            "VIDROVA_skipped_20_25": (vidrva_raw.round().isin({20, 25}) & vidrova_raw.isna()).astype("int8"),
            "VIDLOVA_skipped_20_25": (vidlva_raw.round().isin({20, 25}) & vidlova_raw.isna()).astype("int8"),
        })
        frames.append(out)

        both_se = out["vis_R"].notna() & out["vis_L"].notna()
        per_cycle[cycle] = {
            "n_total": int(len(vix)),
            "n_both_eyes_SE_valid": int(both_se.sum()),
            "n_could_not_obtain_88_SE": n88,
            "median_SE_OD": round(float(out["vis_R"].median()), 2),
            "median_SE_OS": round(float(out["vis_L"].median()), 2),
            "n_VIDRVA_logmar_valid": int(out["VIDRVA_logmar"].notna().sum()),
            "n_VIDLVA_logmar_valid": int(out["VIDLVA_logmar"].notna().sum()),
            "n_VIDRVA_gt200_666": int(out["VIDRVA_gt200_666"].sum()),
            "n_VIDLVA_gt200_666": int(out["VIDLVA_gt200_666"].sum()),
            "n_VIDROVA_logmar_valid": int(out["VIDROVA_logmar"].notna().sum()),
            "n_VIDLOVA_logmar_valid": int(out["VIDLOVA_logmar"].notna().sum()),
            "n_VIDROVA_gt200_666": int(out["VIDROVA_gt200_666"].sum()),
            "n_VIDLOVA_gt200_666": int(out["VIDLOVA_gt200_666"].sum()),
            "n_VIDROVA_skipped_20_25": int(out["VIDROVA_skipped_20_25"].sum()),
            "n_VIDLOVA_skipped_20_25": int(out["VIDLOVA_skipped_20_25"].sum()),
        }
        log.info(
            f"{cycle}: N={len(vix)}, SE ambos olhos={int(both_se.sum())}, "
            f"VIDRVA valid={per_cycle[cycle]['n_VIDRVA_logmar_valid']}, "
            f"VIDROVA valid={per_cycle[cycle]['n_VIDROVA_logmar_valid']}"
        )

    full = pd.concat(frames, ignore_index=True)

    # Acuidade melhor corrigida (variável da Lei 14.126/2021): refração objetiva quando medida;
    # olhos pulados por protocolo (apresentada 20/20-20/25) herdam a acuidade apresentada.
    for eye, pres, obj in (("R", "VIDRVA", "VIDROVA"), ("L", "VIDLVA", "VIDLOVA")):
        skipped = full[f"{obj}_skipped_20_25"] == 1
        best = full[f"{obj}_logmar"].copy()
        best.loc[skipped] = full.loc[skipped, f"{pres}_logmar"]
        full[f"VA_best_{eye}_logmar"] = best.astype("float64")
        full[f"VA_best_{eye}_gt200_666"] = full[f"{obj}_gt200_666"].astype("int8")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Arquivo congelado do paper v6: mesmas linhas de antes (1999-2002); 9 scripts downstream leem
    # este arquivo sem filtro de ciclo, então ele NÃO pode ganhar 2003-2004.
    full[full["cycle"].isin(PAPER_V6_CYCLES)].to_csv(OUT_CSV, index=False)
    full.to_csv(OUT_CSV_V2, index=False)

    valid = full[full["vis_R"].notna() & full["vis_L"].notna()]
    examples = valid.iloc[:: max(len(valid) // 5, 1)][:5]
    ex_records = [
        {
            "SEQN": int(r["SEQN"]),
            "cycle": r["cycle"],
            "OD_sphere": float(r["OD_sphere"]),
            "OD_cyl": float(r["OD_cyl"]),
            "OS_sphere": float(r["OS_sphere"]),
            "OS_cyl": float(r["OS_cyl"]),
            "vis_R_SE": round(float(r["vis_R"]), 2),
            "vis_L_SE": round(float(r["vis_L"]), 2),
            "asym_RL": round(float(r["asym_RL"]), 2),
            "VIDRVA_logmar": round(float(r["VIDRVA_logmar"]), 4) if pd.notna(r["VIDRVA_logmar"]) else None,
            "VIDLVA_logmar": round(float(r["VIDLVA_logmar"]), 4) if pd.notna(r["VIDLVA_logmar"]) else None,
        }
        for _, r in examples.iterrows()
    ]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "script": "vis_01_ingest.py",
                "cycles_ingested": list(CYCLES.keys()),
                "ibm_underflow_eps": SAS_ZERO_EPS,
                "spherical_equivalent_rule": "SE = sphere + cylinder/2; vis_R=OD, vis_L=OS (anatômicos diretos)",
                "visual_acuity_logmar_rule": (
                    "logMAR = log10(code / 20) para denominadores Snellen {20,25,30,40,50,60,80,200}; "
                    "666 (20/200+) -> NaN + flag *_gt200_666; 888 (Could not obtain) -> NaN + flag *_could_not_obtain_888; "
                    "VIDROVA/VIDLOVA medidos apenas quando VIDxVA >= 30 (flag *_skipped_20_25)."
                ),
                "n_total_rows": int(len(full)),
                "per_cycle": per_cycle,
                "example_rows": ex_records,
                "output_csv_frozen_paper_v6": str(OUT_CSV.as_posix()),
                "output_csv_v2_full": str(OUT_CSV_V2.as_posix()),
                "status": "EXECUTED",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info(f"Concluído. CSV={OUT_CSV} | JSON={OUT_JSON}")


if __name__ == "__main__":
    main()

