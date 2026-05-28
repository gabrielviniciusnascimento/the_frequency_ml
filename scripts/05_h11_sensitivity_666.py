#!/usr/bin/env python3
"""
Nome: 05_h11_sensitivity_666.py
Tarefa: Tornar explícita a análise de sensibilidade H11 para 666→NaN vs 666→125dB+flag.
Input: data/processed/frequencia_bruto.csv; data/processed/frequencia_feature_matrix_v1.csv.
Output: data/processed/frequencia_feature_matrix_v1_666cap125.csv; outputs/json/05_h11_sensitivity_666.json.
Dependências: 03_features_v1.py; 04_qa_report.py recomendado.
"""

import logging
import json
from pathlib import Path

# Núcleo científico — sempre presente
import numpy as np
import pandas as pd
from scipy import stats, spatial, linalg
from scipy.spatial.distance import jensenshannon

# ML — scikit-learn para tudo modelável
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
try:
    import shap  # se disponível
except ImportError:
    shap = None

# Clustering
from sklearn.cluster import KMeans, DBSCAN
try:
    import hdbscan
except ImportError:
    hdbscan = None

# Paralelismo
from joblib import Parallel, delayed
import multiprocessing as mp
N_JOBS = max(mp.cpu_count() - 1, 1)

# ── Constantes ───────────────────────────────────────────────────────
RANDOM_STATE = 42
BRUTO_CSV = Path("data/processed/frequencia_bruto.csv")
FEATURE_NAN_CSV = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT_PATH = Path("outputs/json/05_h11_sensitivity_666.json")
LOG_PATH = Path("outputs/logs/05_h11_sensitivity_666.log")
FEATURE_CAP_CSV = Path("data/processed/frequencia_feature_matrix_v1_666cap125.csv")
FEATURE_CAP_PARQUET = Path("data/processed/frequencia_feature_matrix_v1_666cap125.parquet")
DELTA_CSV = Path("outputs/json/h11_feature_deltas_666cap125_vs_nan.csv")
NO_RESPONSE_CODE = 666.0
CAP_VALUE_DB = 125.0
FREQS = np.array([500, 1000, 2000, 3000, 4000, 6000, 8000], dtype=np.int32)
LOW_FREQS = np.array([500, 1000, 2000], dtype=np.int32)
SPEECH_FREQS = np.array([500, 1000, 2000, 4000], dtype=np.int32)
MID_FREQS = np.array([1000, 2000, 3000, 4000], dtype=np.int32)
HIGH_FREQS = np.array([3000, 4000, 6000, 8000], dtype=np.int32)
ULTRA_PROXY_FREQS = np.array([6000, 8000], dtype=np.int32)
SENSITIVITY_FEATURES = [
    "pta_low_R", "pta_low_L", "pta_speech_R", "pta_speech_L", "pta_high_R", "pta_high_L",
    "pta_6_8_R", "pta_6_8_L", "hf_lf_contrast_R", "hf_lf_contrast_L",
    "slope_500_8000_R", "slope_500_8000_L", "slope_2000_8000_R", "slope_2000_8000_L",
    "asym_mean", "asym_high", "pta_high_mean_binaural", "hf_lf_contrast_mean"
]

# ── Logging padronizado ──────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Checkpointing ────────────────────────────────────────────────────
if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def mean_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    return df[existing].mean(axis=1, skipna=True).astype("float32")


def build_cap125_features(bruto: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    out = template.copy()
    # Substituir apenas onde raw == 666; manter demais features/contextos do template.
    for ear in ["R", "L"]:
        for freq in FREQS:
            clean = f"thr_{ear}_{int(freq)}"
            raw = f"thr_{ear}_{int(freq)}_raw"
            if clean in out.columns and raw in bruto.columns:
                raw_s = pd.to_numeric(bruto[raw], errors="coerce")
                cap_s = pd.to_numeric(out[clean], errors="coerce").astype("float32")
                cap_s = cap_s.mask(raw_s == NO_RESPONSE_CODE, CAP_VALUE_DB)
                out[clean] = cap_s.astype("float32")

    for ear in ["R", "L"]:
        out[f"pta_low_{ear}"] = mean_cols(out, [f"thr_{ear}_{int(f)}" for f in LOW_FREQS])
        out[f"pta_speech_{ear}"] = mean_cols(out, [f"thr_{ear}_{int(f)}" for f in SPEECH_FREQS])
        out[f"pta_mid_{ear}"] = mean_cols(out, [f"thr_{ear}_{int(f)}" for f in MID_FREQS])
        out[f"pta_high_{ear}"] = mean_cols(out, [f"thr_{ear}_{int(f)}" for f in HIGH_FREQS])
        out[f"pta_6_8_{ear}"] = mean_cols(out, [f"thr_{ear}_{int(f)}" for f in ULTRA_PROXY_FREQS])
        out[f"hf_lf_contrast_{ear}"] = (out[f"pta_high_{ear}"] - out[f"pta_low_{ear}"]).astype("float32")
        if f"thr_{ear}_8000" in out.columns and f"thr_{ear}_500" in out.columns:
            out[f"slope_500_8000_{ear}"] = ((out[f"thr_{ear}_8000"] - out[f"thr_{ear}_500"]) / np.log2(8000 / 500)).astype("float32")
        if f"thr_{ear}_8000" in out.columns and f"thr_{ear}_2000" in out.columns:
            out[f"slope_2000_8000_{ear}"] = ((out[f"thr_{ear}_8000"] - out[f"thr_{ear}_2000"]) / np.log2(8000 / 2000)).astype("float32")
        if all(f"thr_{ear}_{f}" in out.columns for f in [3000, 4000, 6000]):
            out[f"notch_4k_{ear}"] = (out[f"thr_{ear}_4000"] - out[[f"thr_{ear}_3000", f"thr_{ear}_6000"]].mean(axis=1)).astype("float32")
        if all(f"thr_{ear}_{f}" in out.columns for f in [6000, 8000]):
            out[f"recovery_8k_{ear}"] = (out[f"thr_{ear}_8000"] - out[f"thr_{ear}_6000"]).astype("float32")
        if all(f"thr_{ear}_{f}" in out.columns for f in [4000, 6000, 8000]):
            out[f"curvature_high_{ear}"] = (out[f"thr_{ear}_8000"] - 2 * out[f"thr_{ear}_6000"] + out[f"thr_{ear}_4000"]).astype("float32")

    for freq in FREQS:
        r = f"thr_R_{int(freq)}"
        l = f"thr_L_{int(freq)}"
        if r in out.columns and l in out.columns:
            out[f"better_thr_{int(freq)}"] = out[[r, l]].min(axis=1).astype("float32")
            out[f"worse_thr_{int(freq)}"] = out[[r, l]].max(axis=1).astype("float32")
            out[f"asym_thr_{int(freq)}"] = (out[r] - out[l]).abs().astype("float32")

    out["asym_mean"] = mean_cols(out, [f"asym_thr_{int(f)}" for f in FREQS])
    out["asym_high"] = mean_cols(out, [f"asym_thr_{int(f)}" for f in [4000, 6000, 8000]])
    out["pta_high_mean_binaural"] = mean_cols(out, ["pta_high_R", "pta_high_L"])
    out["hf_lf_contrast_mean"] = mean_cols(out, ["hf_lf_contrast_R", "hf_lf_contrast_L"])
    return out


def save_with_parquet_fallback(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> dict:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    status = {"csv": str(csv_path), "parquet": None, "parquet_error": None}
    try:
        df.to_parquet(parquet_path, index=False)
        status["parquet"] = str(parquet_path)
    except Exception as exc:
        log.warning(f"Não foi possível salvar parquet {parquet_path}: {exc}")
        status["parquet_error"] = repr(exc)
    return status


def main():
    log.info("Iniciando sensibilidade H11: 666→NaN vs 666→125dB+flag...")
    if not BRUTO_CSV.exists() or not FEATURE_NAN_CSV.exists():
        raise FileNotFoundError("Dependências ausentes para H11")
    bruto = pd.read_csv(BRUTO_CSV, low_memory=False)
    features_nan = pd.read_csv(FEATURE_NAN_CSV, low_memory=False)
    log.info(f"Shape bruto: {bruto.shape}")
    log.info(f"Shape features_nan: {features_nan.shape}")

    if not bruto[["SEQN", "cycle"]].equals(features_nan[["SEQN", "cycle"]]):
        raise ValueError("Ordem/chaves entre bruto e feature matrix não coincidem; abortando H11")

    features_cap = build_cap125_features(bruto, features_nan)
    affected_mask = features_nan.get("n_no_response_666_thresholds", pd.Series(0, index=features_nan.index)) > 0
    n_affected = int(affected_mask.sum())
    log.info(f"Linhas afetadas por 666: {n_affected}")

    rows = []
    for col in SENSITIVITY_FEATURES:
        if col not in features_nan.columns or col not in features_cap.columns:
            continue
        nan_s = pd.to_numeric(features_nan.loc[affected_mask, col], errors="coerce")
        cap_s = pd.to_numeric(features_cap.loc[affected_mask, col], errors="coerce")
        delta = cap_s - nan_s
        rows.append({
            "feature": col,
            "n_affected_nonnull_delta": int(delta.notna().sum()),
            "mean_delta": float(delta.mean(skipna=True)) if delta.notna().any() else np.nan,
            "median_delta": float(delta.median(skipna=True)) if delta.notna().any() else np.nan,
            "max_abs_delta": float(delta.abs().max(skipna=True)) if delta.notna().any() else np.nan,
            "n_changed": int((delta.fillna(0).abs() > 0).sum()),
        })
    deltas = pd.DataFrame(rows)
    DELTA_CSV.parent.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(DELTA_CSV, index=False)
    output_status = save_with_parquet_fallback(features_cap, FEATURE_CAP_CSV, FEATURE_CAP_PARQUET)

    no_response_by_cycle = features_nan.groupby("cycle")["n_no_response_666_thresholds"].agg(
        n_rows="count",
        n_rows_any_666=lambda s: int((s > 0).sum()),
        total_666="sum"
    ).reset_index()

    finding_status = "PRELIMINAR — sem clustering; sensibilidade geométrica preparada antes de interpretação"
    log.info(f"""
FINDING #H11-SETUP
Descrição: 666/no response altera a matriz de features; versão alternativa 666→{CAP_VALUE_DB}dB+flag foi computada.
Métrica: n_linhas_afetadas = {n_affected}
N: {features_nan.shape[0]}
Output salvo: {OUTPUT_PATH}
Status: {finding_status}
""")

    result = {
        "script": "05_h11_sensitivity_666.py",
        "random_state": RANDOM_STATE,
        "cap_value_db": CAP_VALUE_DB,
        "input_shape": list(features_nan.shape),
        "n_rows_affected_by_666": n_affected,
        "feature_cap_outputs": output_status,
        "delta_csv": str(DELTA_CSV),
        "no_response_by_cycle": no_response_by_cycle.to_dict(orient="records"),
        "delta_summary": deltas.to_dict(orient="records"),
        "status": finding_status,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
