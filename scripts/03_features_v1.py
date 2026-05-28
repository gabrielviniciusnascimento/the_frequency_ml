#!/usr/bin/env python3
"""
Nome: 03_features_v1.py
Tarefa: Construir FEATS_THE_FREQUENCY_V1 a partir de frequencia_bruto.
Input: data/processed/frequencia_bruto.csv.
Output: data/processed/frequencia_feature_matrix_v1.csv; outputs/json/03_features_v1.json.
Dependências: 02_merge_context.py.
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
INPUT_CSV = Path("data/processed/frequencia_bruto.csv")
OUTPUT_PATH = Path("outputs/json/03_features_v1.json")
LOG_PATH = Path("outputs/logs/03_features_v1.log")
FEATURE_CSV = Path("data/processed/frequencia_feature_matrix_v1.csv")
FEATURE_PARQUET = Path("data/processed/frequencia_feature_matrix_v1.parquet")
FEATURE_DICTIONARY_JSON = Path("outputs/json/feature_dictionary_v1.json")
FREQS = np.array([500, 1000, 2000, 3000, 4000, 6000, 8000], dtype=np.int32)
HIGH_FREQS = np.array([3000, 4000, 6000, 8000], dtype=np.int32)
LOW_FREQS = np.array([500, 1000, 2000], dtype=np.int32)
SPEECH_FREQS = np.array([500, 1000, 2000, 4000], dtype=np.int32)
MID_FREQS = np.array([1000, 2000, 3000, 4000], dtype=np.int32)
ULTRA_PROXY_FREQS = np.array([6000, 8000], dtype=np.int32)
RELIABILITY_WARNING_DB = 10.0

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


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame({"SEQN": df["SEQN"].astype("int64"), "cycle": df["cycle"].astype("string")})
    feature_dict: dict[str, str] = {}

    # IDs/contexto mínimo
    for col in ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "INDFMPIR", "SDMVSTRA", "SDMVPSU", "WTMEC2YR", "WTSAU2YR", "WTMECPRP"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            feature_dict[col] = f"Contexto NHANES original: {col}"

    # Thresholds limpos e flags brutas
    for ear in ["R", "L"]:
        for freq in FREQS:
            clean = f"thr_{ear}_{int(freq)}"
            if clean in df.columns:
                out[clean] = pd.to_numeric(df[clean], errors="coerce").astype("float32")
                feature_dict[clean] = f"Limiar {ear} {int(freq)} Hz, dB HL, 666/888 tratados como NaN"
            for suffix in ["no_response", "could_not_obtain"]:
                flag = f"thr_{ear}_{int(freq)}_{suffix}"
                if flag in df.columns:
                    out[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype("int8")
                    feature_dict[flag] = f"Flag {suffix} para {ear} {int(freq)} Hz"

    # Features por ouvido
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

        for feat in ["pta_low", "pta_speech", "pta_mid", "pta_high", "pta_6_8", "hf_lf_contrast", "slope_500_8000", "slope_2000_8000", "notch_4k", "recovery_8k", "curvature_high"]:
            name = f"{feat}_{ear}"
            if name in out.columns:
                feature_dict[name] = f"Feature audiométrica derivada: {name}"

    # Binaural / assimetria
    for freq in FREQS:
        r = f"thr_R_{int(freq)}"
        l = f"thr_L_{int(freq)}"
        if r in out.columns and l in out.columns:
            out[f"better_thr_{int(freq)}"] = out[[r, l]].min(axis=1).astype("float32")
            out[f"worse_thr_{int(freq)}"] = out[[r, l]].max(axis=1).astype("float32")
            out[f"asym_thr_{int(freq)}"] = (out[r] - out[l]).abs().astype("float32")
            feature_dict[f"asym_thr_{int(freq)}"] = f"Assimetria absoluta R-L em {int(freq)} Hz"

    asym_cols = [f"asym_thr_{int(f)}" for f in FREQS if f"asym_thr_{int(f)}" in out.columns]
    asym_high_cols = [f"asym_thr_{int(f)}" for f in [4000, 6000, 8000] if f"asym_thr_{int(f)}" in out.columns]
    out["asym_mean"] = mean_cols(out, asym_cols)
    out["asym_high"] = mean_cols(out, asym_high_cols)
    out["pta_high_mean_binaural"] = mean_cols(out, ["pta_high_R", "pta_high_L"])
    out["hf_lf_contrast_mean"] = mean_cols(out, ["hf_lf_contrast_R", "hf_lf_contrast_L"])

    # Reliability 1000 Hz
    for ear in ["R", "L"]:
        col = f"retest1000_absdiff_{ear}"
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            out[f"retest1000_gt_{int(RELIABILITY_WARNING_DB)}db_{ear}"] = (out[col] > RELIABILITY_WARNING_DB).fillna(False).astype("int8")
            feature_dict[col] = f"Diferença absoluta 1000 Hz primeira vs segunda leitura, {ear}"

    # Middle-ear / otoscopia flags conservadoras: só codifica presença se coluna existe e valor positivo comum.
    for col in ["AUXLOEXC", "AUXLOIMC", "AUXLOCOL", "AUXLOABN", "AUXROEXC", "AUXROIMC", "AUXROCOL", "AUXROABN"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            feature_dict[col] = f"Flag/medida original AUX para otoscopia: {col}"
    for col in ["AUXTMEPR", "AUXTPVR", "AUXTWIDR", "AUXTCOMR", "AUXTMEPL", "AUXTPVL", "AUXTWIDL", "AUXTCOML", "AUAREQC", "AUALEQC", "AUATYMTR", "AUATYMTL"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            feature_dict[col] = f"Timpanometria/qualidade original: {col}"

    # AUQ sintomas/contexto — preservar código original, sem inverter sim/não sem codebook por ciclo.
    for col in ["AUQ054", "AUQ100", "AUQ101", "AUQ110", "AUQ191", "AUQ250", "AUQ255", "AUQ260", "AUQ270", "AUQ280", "AUQ300", "AUQ310", "AUQ320", "AUQ330", "AUQ340", "AUQ350", "AUQ360", "AUQ370", "AUQ380", "AUQ400", "AUQ410E", "AUQ410e", "AUQ480", "AUQ490", "AUQ500"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            feature_dict[col] = f"Código original AUQ preservado para interpretação pós-cluster: {col}"

    # Contagem por linha de 666/888 em frequências principais
    no_response_cols = [c for c in out.columns if c.endswith("_no_response")]
    could_not_cols = [c for c in out.columns if c.endswith("_could_not_obtain")]
    if no_response_cols:
        out["n_no_response_666_thresholds"] = out[no_response_cols].sum(axis=1).astype("int16")
    if could_not_cols:
        out["n_could_not_obtain_888_thresholds"] = out[could_not_cols].sum(axis=1).astype("int16")

    feature_dict["n_no_response_666_thresholds"] = "Número de thresholds com código 666; H11 sensibilidade obrigatória"
    feature_dict["n_could_not_obtain_888_thresholds"] = "Número de thresholds com código 888"
    return out, feature_dict


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
    log.info("Iniciando feature engineering V1...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Dependência ausente: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    log.info(f"Shape frequencia_bruto: {df.shape}")
    features, feature_dict = add_features(df)
    log.info(f"Shape feature_matrix_v1: {features.shape}")
    if "n_no_response_666_thresholds" in features.columns:
        affected = int((features["n_no_response_666_thresholds"] > 0).sum())
        log.info(f"H11 checkpoint: linhas com ao menos um 666 = {affected}")
        if affected > 0:
            log.warning("H11 obrigatório: rodar 05_h11_sensitivity_666.py antes de interpretar cluster de alta perda.")

    output_status = save_with_parquet_fallback(features, FEATURE_CSV, FEATURE_PARQUET)
    FEATURE_DICTIONARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with FEATURE_DICTIONARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(feature_dict, f, indent=2, ensure_ascii=False)

    result = {
        "script": "03_features_v1.py",
        "random_state": RANDOM_STATE,
        "input_shape": list(df.shape),
        "output_shape": list(features.shape),
        "outputs": output_status,
        "feature_dictionary_json": str(FEATURE_DICTIONARY_JSON),
        "h11_lines_with_666": int((features.get("n_no_response_666_thresholds", pd.Series(dtype=int)) > 0).sum()) if "n_no_response_666_thresholds" in features.columns else None,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
