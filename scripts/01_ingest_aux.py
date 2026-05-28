#!/usr/bin/env python3
"""
Nome: 01_ingest_aux.py
Tarefa: Ingerir NHANES AUX XPT, padronizar limiares auditivos e gerar tabelas wide/long.
Input: outputs/json/00_download_nhanes.json; data/raw/nhanes/*/AUX*.xpt.
Output: data/interim/nhanes_aux_wide.csv; data/interim/nhanes_aux_long.csv; outputs/json/01_ingest_aux.json.
Dependências: 00_download_nhanes.py.
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
DOWNLOAD_MANIFEST = Path("outputs/json/00_download_nhanes.json")
OUTPUT_PATH = Path("outputs/json/01_ingest_aux.json")
LOG_PATH = Path("outputs/logs/01_ingest_aux.log")
INTERIM_DIR = Path("data/interim")
AUX_WIDE_CSV = INTERIM_DIR / "nhanes_aux_wide.csv"
AUX_LONG_CSV = INTERIM_DIR / "nhanes_aux_long.csv"
AUX_WIDE_PARQUET = INTERIM_DIR / "nhanes_aux_wide.parquet"
AUX_LONG_PARQUET = INTERIM_DIR / "nhanes_aux_long.parquet"
MISSINGNESS_CSV = Path("outputs/json/threshold_missingness_by_cycle_v1.csv")
VALID_THRESHOLD_MIN_DB = -10.0
VALID_THRESHOLD_MAX_DB = 130.0
NO_RESPONSE_CODE = 666.0
COULD_NOT_OBTAIN_CODE = 888.0

FREQS = np.array([500, 1000, 2000, 3000, 4000, 6000, 8000], dtype=np.int32)
RIGHT_COLS = {
    500: "AUXU500R", 1000: "AUXU1K1R", 2000: "AUXU2KR", 3000: "AUXU3KR",
    4000: "AUXU4KR", 6000: "AUXU6KR", 8000: "AUXU8KR", "1000_2": "AUXU1K2R"
}
LEFT_COLS = {
    500: "AUXU500L", 1000: "AUXU1K1L", 2000: "AUXU2KL", 3000: "AUXU3KL",
    4000: "AUXU4KL", 6000: "AUXU6KL", 8000: "AUXU8KL", "1000_2": "AUXU1K2L"
}
QA_COLS = [
    "AUAEXSTS", "AUAEXCMT", "AUAEAR", "AUAMODE", "AUAFMANL", "AUAFMANR",
    "AUXOTSPL", "AUXLOEXC", "AUXLOIMC", "AUXLOCOL", "AUXLOABN",
    "AUXROTSP", "AUXROEXC", "AUXROIMC", "AUXROCOL", "AUXROABN",
    "AUXTMEPR", "AUXTPVR", "AUXTWIDR", "AUXTCOMR",
    "AUXTMEPL", "AUXTPVL", "AUXTWIDL", "AUXTCOML",
    "AUAREQC", "AUALEQC", "AUATYMTR", "AUATYMTL",
    "AUQ010", "AUQ011", "AUQ020", "AUQ020A", "AUQ020B", "AUQ020C", "AUQ020D", "AUQ020E",
    "AUQ030", "AUQ031", "AUQ040", "AUQ041", "AUQ050", "AUQ051",
    "AUQ520", "AUQ530", "AUQ540", "AUQ550", "AUQ550A", "AUQ550B", "AUQ550C", "AUQ550D",
    "AUQ550E", "AUQ560", "AUQ570", "AUQ580", "AUQ590", "AUQ600", "AUQ610"
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


def clean_threshold_nan(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype("float32")
    x = x.mask(x == COULD_NOT_OBTAIN_CODE, np.nan)
    x = x.mask(x == NO_RESPONSE_CODE, np.nan)
    x = x.mask((x < VALID_THRESHOLD_MIN_DB) | (x > VALID_THRESHOLD_MAX_DB), np.nan)
    return x.astype("float32")


def read_sas_xpt(path: Path) -> pd.DataFrame:
    log.info(f"Lendo XPT: {path}")
    df = pd.read_sas(path, format="xport")
    log.info(f"Shape bruto {path.name}: {df.shape}")
    return df


def build_wide_one(record: dict) -> tuple[pd.DataFrame | None, list[dict]]:
    cycle = record["cycle"]
    path = Path(record["path"])
    missing_rows: list[dict] = []
    try:
        aux = read_sas_xpt(path)
        if "SEQN" not in aux.columns:
            raise ValueError(f"SEQN ausente em {path}")
        out = pd.DataFrame({"SEQN": aux["SEQN"].astype("int64"), "cycle": cycle})
        for ear, mapping in [("R", RIGHT_COLS), ("L", LEFT_COLS)]:
            for freq in FREQS:
                source = mapping[int(freq)]
                if source not in aux.columns:
                    log.warning(f"ANOMALIA DETECTADA — coluna ausente: {source} em {cycle}")
                    continue
                raw_name = f"thr_{ear}_{int(freq)}_raw"
                clean_name = f"thr_{ear}_{int(freq)}"
                out[raw_name] = pd.to_numeric(aux[source], errors="coerce").astype("float32")
                out[clean_name] = clean_threshold_nan(aux[source])
                out[f"thr_{ear}_{int(freq)}_no_response"] = (out[raw_name] == NO_RESPONSE_CODE).astype("int8")
                out[f"thr_{ear}_{int(freq)}_could_not_obtain"] = (out[raw_name] == COULD_NOT_OBTAIN_CODE).astype("int8")

                missing_rows.append({
                    "cycle": cycle,
                    "ear": ear,
                    "frequency_hz": int(freq),
                    "column": source,
                    "n_total": int(aux.shape[0]),
                    "n_valid_model_nan": int(out[clean_name].notna().sum()),
                    "n_no_response_666": int((out[raw_name] == NO_RESPONSE_CODE).sum()),
                    "n_could_not_obtain_888": int((out[raw_name] == COULD_NOT_OBTAIN_CODE).sum()),
                    "n_missing_raw": int(out[raw_name].isna().sum()),
                })

        for ear, mapping in [("R", RIGHT_COLS), ("L", LEFT_COLS)]:
            c1 = mapping[1000]
            c2 = mapping["1000_2"]
            if c1 in aux.columns and c2 in aux.columns:
                t1 = clean_threshold_nan(aux[c1])
                t2 = clean_threshold_nan(aux[c2])
                out[f"retest1000_absdiff_{ear}"] = (t1 - t2).abs().astype("float32")

        for col in QA_COLS:
            if col in aux.columns:
                out[col] = aux[col]

        log.info(f"Shape wide harmonizado {cycle}: {out.shape}")
        return out, missing_rows
    except Exception as exc:
        log.exception(f"Falha ao ingerir AUX: {cycle} path={path}")
        missing_rows.append({"cycle": cycle, "error": repr(exc)})
        return None, missing_rows


def build_long_from_wide(wide: pd.DataFrame) -> pd.DataFrame:
    value_cols = [f"thr_{ear}_{freq}" for ear in ["R", "L"] for freq in FREQS if f"thr_{ear}_{freq}" in wide.columns]
    raw_cols = [f"{c}_raw" for c in value_cols if f"{c}_raw" in wide.columns]
    clean_long = wide[["SEQN", "cycle"] + value_cols].melt(
        id_vars=["SEQN", "cycle"], var_name="ear_frequency", value_name="threshold_db_hl_model_nan"
    )
    raw_long = wide[["SEQN", "cycle"] + raw_cols].rename(columns={c: c.replace("_raw", "") for c in raw_cols}).melt(
        id_vars=["SEQN", "cycle"], var_name="ear_frequency", value_name="threshold_db_hl_raw"
    )
    long = clean_long.merge(raw_long, on=["SEQN", "cycle", "ear_frequency"], how="left")
    parts = long["ear_frequency"].str.extract(r"thr_([RL])_(\d+)")
    long["ear"] = parts[0]
    long["frequency_hz"] = pd.to_numeric(parts[1], errors="coerce").astype("Int32")
    long["no_response_666"] = (long["threshold_db_hl_raw"] == NO_RESPONSE_CODE).astype("int8")
    long["could_not_obtain_888"] = (long["threshold_db_hl_raw"] == COULD_NOT_OBTAIN_CODE).astype("int8")
    long = long[["SEQN", "cycle", "ear", "frequency_hz", "threshold_db_hl_raw", "threshold_db_hl_model_nan", "no_response_666", "could_not_obtain_888"]]
    log.info(f"Shape long harmonizado: {long.shape}")
    return long


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
    log.info("Iniciando ingestão AUX...")
    if not DOWNLOAD_MANIFEST.exists():
        raise FileNotFoundError(f"Dependência ausente: {DOWNLOAD_MANIFEST}")
    manifest = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    aux_records = [
        r for r in manifest["results"]
        if r["status"] == "ok" and r["component"] == "aux"
    ]
    log.info(f"Arquivos AUX principais encontrados: {len(aux_records)}")

    processed = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(build_wide_one)(r) for r in aux_records)
    frames = [item[0] for item in processed if item[0] is not None]
    missing_rows = [row for item in processed for row in item[1]]
    if not frames:
        raise RuntimeError("Nenhum AUX ingerido com sucesso")

    wide = pd.concat(frames, ignore_index=True, sort=False)
    log.info(f"Shape wide concatenado: {wide.shape}")
    long = build_long_from_wide(wide)

    wide_status = save_with_parquet_fallback(wide, AUX_WIDE_CSV, AUX_WIDE_PARQUET)
    long_status = save_with_parquet_fallback(long, AUX_LONG_CSV, AUX_LONG_PARQUET)
    miss = pd.DataFrame(missing_rows)
    MISSINGNESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    miss.to_csv(MISSINGNESS_CSV, index=False)

    result = {
        "script": "01_ingest_aux.py",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "n_aux_files": len(aux_records),
        "wide_shape": list(wide.shape),
        "long_shape": list(long.shape),
        "wide_outputs": wide_status,
        "long_outputs": long_status,
        "missingness_csv": str(MISSINGNESS_CSV),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
