#!/usr/bin/env python3
"""
Nome: 02_merge_context.py
Tarefa: Unir NHANES AUX harmonizado com DEMO e AUQ por ciclo/SEQN.
Input: data/interim/nhanes_aux_wide.csv; outputs/json/00_download_nhanes.json.
Output: data/processed/frequencia_bruto.csv; outputs/json/02_merge_context.json.
Dependências: 00_download_nhanes.py; 01_ingest_aux.py.
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
AUX_WIDE_CSV = Path("data/interim/nhanes_aux_wide.csv")
OUTPUT_PATH = Path("outputs/json/02_merge_context.json")
LOG_PATH = Path("outputs/logs/02_merge_context.log")
PROCESSED_DIR = Path("data/processed")
FREQUENCIA_BRUTO_CSV = PROCESSED_DIR / "frequencia_bruto.csv"
FREQUENCIA_BRUTO_PARQUET = PROCESSED_DIR / "frequencia_bruto.parquet"
COLUMN_PRESENCE_CSV = Path("outputs/json/nhanes_column_presence_v1.csv")

DEMO_KEEP = [
    "SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3", "DMDEDUC2", "DMDEDUC3",
    "INDFMPIR", "SDMVSTRA", "SDMVPSU", "WTMEC2YR", "WTINT2YR", "WTSAU2YR", "WTSAU4YR",
    "WTMECPRP", "WTINTPRP"
]
AUQ_KEEP = [
    "SEQN", "AUQ054", "AUQ060", "AUQ070", "AUQ080", "AUQ090", "AUQ100", "AUQ101", "AUQ110",
    "AUQ136", "AUQ138", "AUQ139", "AUQ144", "AUQ146", "AUD148", "AUQ147", "AUQ149A", "AUQ149B",
    "AUQ149C", "AUQ149a", "AUQ149b", "AUQ149c", "AUQ152", "AUQ153", "AUQ154", "AUQ156", "AUQ191",
    "AUQ250", "AUQ255", "AUQ260", "AUQ270", "AUQ280", "AUQ300", "AUQ310", "AUQ320", "AUQ330",
    "AUQ340", "AUQ350", "AUQ360", "AUQ370", "AUQ380", "AUQ395", "AUQ400", "AUQ410A", "AUQ410B",
    "AUQ410C", "AUQ410D", "AUQ410E", "AUQ410F", "AUQ410G", "AUQ410H", "AUQ410I", "AUQ410J",
    "AUQ410a", "AUQ410b", "AUQ410c", "AUQ410d", "AUQ410e", "AUQ410f", "AUQ410g", "AUQ410h",
    "AUQ410i", "AUQ410j", "AUQ420", "AUQ430", "AUQ440", "AUQ450A", "AUQ450B", "AUQ450C",
    "AUQ450D", "AUQ450E", "AUQ450F", "AUQ460", "AUQ470", "AUQ480", "AUQ490", "AUQ500", "AUQ510"
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


def read_sas_xpt(path: Path) -> pd.DataFrame:
    log.info(f"Lendo XPT contexto: {path}")
    df = pd.read_sas(path, format="xport")
    log.info(f"Shape {path.name}: {df.shape}")
    return df


def load_download_lookup() -> dict:
    manifest = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], Path] = {}
    for r in manifest["results"]:
        if r["status"] == "ok":
            lookup[(r["cycle"], r["component"])] = Path(r["path"])
    return lookup


def select_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def merge_one_cycle(cycle: str, base: pd.DataFrame, lookup: dict) -> tuple[pd.DataFrame, list[dict]]:
    events: list[dict] = []
    out = base.copy()
    log.info(f"Merge ciclo={cycle}; base shape={out.shape}")

    demo_key = (cycle, "demo")
    if demo_key in lookup:
        demo = read_sas_xpt(lookup[demo_key])
        keep = select_existing(demo, DEMO_KEEP)
        events.append({"cycle": cycle, "component": "demo", "path": str(lookup[demo_key]), "n_cols_kept": len(keep), "cols_kept": keep})
        out = out.merge(demo[keep], on="SEQN", how="left", validate="many_to_one")
        log.info(f"Após DEMO {cycle}: {out.shape}")
    else:
        log.warning(f"ANOMALIA DETECTADA — DEMO ausente para ciclo={cycle}")
        events.append({"cycle": cycle, "component": "demo", "error": "missing_download"})

    auq_key = (cycle, "auq")
    if auq_key in lookup:
        auq = read_sas_xpt(lookup[auq_key])
        keep = select_existing(auq, AUQ_KEEP)
        # Evitar colisão com colunas de pre-exam já vindas de AUX.
        keep_no_collision = ["SEQN"] + [c for c in keep if c == "SEQN" or c not in out.columns]
        keep_no_collision = list(dict.fromkeys(keep_no_collision))
        dropped_collision = sorted(set(keep) - set(keep_no_collision))
        events.append({
            "cycle": cycle,
            "component": "auq",
            "path": str(lookup[auq_key]),
            "n_cols_kept": len(keep_no_collision),
            "cols_kept": keep_no_collision,
            "dropped_collision": dropped_collision,
        })
        out = out.merge(auq[keep_no_collision], on="SEQN", how="left", validate="many_to_one")
        log.info(f"Após AUQ {cycle}: {out.shape}")
    else:
        log.warning(f"ANOMALIA DETECTADA — AUQ ausente para ciclo={cycle}")
        events.append({"cycle": cycle, "component": "auq", "error": "missing_download"})

    return out, events


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
    log.info("Iniciando merge AUX + DEMO + AUQ...")
    if not AUX_WIDE_CSV.exists():
        raise FileNotFoundError(f"Dependência ausente: {AUX_WIDE_CSV}")
    if not DOWNLOAD_MANIFEST.exists():
        raise FileNotFoundError(f"Dependência ausente: {DOWNLOAD_MANIFEST}")

    wide = pd.read_csv(AUX_WIDE_CSV, low_memory=False)
    log.info(f"Shape AUX wide carregado: {wide.shape}")
    lookup = load_download_lookup()

    cycles = sorted(wide["cycle"].dropna().unique().tolist())
    merged = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(merge_one_cycle)(cycle, wide.loc[wide["cycle"] == cycle].copy(), lookup)
        for cycle in cycles
    )
    frames = [m[0] for m in merged]
    events = [e for m in merged for e in m[1]]
    bruto = pd.concat(frames, ignore_index=True, sort=False)
    log.info(f"Shape frequencia_bruto concatenado: {bruto.shape}")

    output_status = save_with_parquet_fallback(bruto, FREQUENCIA_BRUTO_CSV, FREQUENCIA_BRUTO_PARQUET)
    presence = pd.DataFrame(events)
    COLUMN_PRESENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    presence.to_csv(COLUMN_PRESENCE_CSV, index=False)

    result = {
        "script": "02_merge_context.py",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "input_shape": list(wide.shape),
        "output_shape": list(bruto.shape),
        "cycles": cycles,
        "outputs": output_status,
        "column_presence_csv": str(COLUMN_PRESENCE_CSV),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
