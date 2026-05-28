#!/usr/bin/env python3
"""
Nome: 00_download_nhanes.py
Tarefa: Baixar com cache os arquivos públicos NHANES necessários para The Frequency ML.
Input: Manifest embutido de ciclos NHANES públicos.
Output: outputs/json/00_download_nhanes.json; arquivos .xpt em data/raw/nhanes/.
Dependências: nenhuma.
"""

import logging
import json
from pathlib import Path
import time
import hashlib
import requests

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
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_SLEEP_SECONDS = 1
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}.xpt"
RAW_ROOT = Path("data/raw/nhanes")
OUTPUT_PATH = Path("outputs/json/00_download_nhanes.json")
LOG_PATH = Path("outputs/logs/00_download_nhanes.log")

NHANES_MANIFEST = {
    "1999-2000": {"year": "1999", "files": {"aux": "AUX1", "demo": "DEMO", "auq": "AUQ"}},
    "2001-2002": {"year": "2001", "files": {"aux": "AUX_B", "demo": "DEMO_B", "auq": "AUQ_B"}},
    "2003-2004": {"year": "2003", "files": {"aux": "AUX_C", "demo": "DEMO_C", "auq": "AUQ_C"}},
    "2005-2006": {"year": "2005", "files": {"aux": "AUX_D", "demo": "DEMO_D", "auq": "AUQ_D"}},
    "2007-2008": {"year": "2007", "files": {"aux": "AUX_E", "demo": "DEMO_E", "auq": "AUQ_E"}},
    "2009-2010": {"year": "2009", "files": {"aux": "AUX_F", "demo": "DEMO_F", "auq": "AUQ_F"}},
    "2011-2012": {"year": "2011", "files": {"aux": "AUX_G", "demo": "DEMO_G", "auq": "AUQ_G"}},
    # 2013-2014: sem componente AUX público.
    "2015-2016": {"year": "2015", "files": {"aux": "AUX_I", "demo": "DEMO_I", "auq": "AUQ_I"}},
    # Não concatenar AUX_J com P_AUX numa mesma análise principal.
    "2017-2018": {"year": "2017", "files": {"aux_j_reference": "AUX_J", "demo_j_reference": "DEMO_J", "auq_j_reference": "AUQ_J"}},
    "2017-Mar2020": {"year": "2017", "files": {"aux": "P_AUX", "demo": "P_DEMO", "auq": "P_AUQ", "rx": "P_RXQ_RX"}},
}

# ── Logging padronizado ──────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Checkpointing ────────────────────────────────────────────────────
if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def validar_conteudo(path: Path, esperado: str = "xpt") -> None:
    """Checar se o que chegou não é HTML/página de erro."""
    content = path.read_bytes()[:512]
    lowered = content.lower()
    if b"<html" in lowered or b"<!doctype" in lowered:
        raise ValueError(f"ERRO: {path} contém HTML — provável página de erro/login")
    if esperado == "xpt" and len(content) < 80:
        raise ValueError(f"ERRO: {path} pequeno demais para XPT válido")
    log.info(f"Conteúdo validado: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_com_cache(url: str, path_local: Path, force: bool = False) -> Path:
    """Nunca baixar o que já foi baixado."""
    path = Path(path_local)
    if path.exists() and not force:
        log.info(f"Cache hit: {path_local} ({path.stat().st_size / 1024:.1f} KB)")
        validar_conteudo(path, esperado="xpt")
        return path

    log.info(f"Fetch: {url}")
    r = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    log.info(f"Salvo: {path_local} ({len(r.content) / 1024:.1f} KB)")
    validar_conteudo(path, esperado="xpt")
    time.sleep(REQUEST_SLEEP_SECONDS)
    return path


def download_one(cycle: str, year: str, component: str, file_stem: str) -> dict:
    url = BASE_URL.format(year=year, file=file_stem)
    local_path = RAW_ROOT / year / f"{file_stem}.xpt"
    result = {
        "cycle": cycle,
        "year": year,
        "component": component,
        "file": file_stem,
        "url": url,
        "path": str(local_path),
        "status": "pending",
        "size_bytes": None,
        "sha256": None,
        "error": None,
    }
    try:
        path = fetch_com_cache(url, local_path)
        result.update({
            "status": "ok",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    except Exception as exc:
        log.exception(f"Falha no download: cycle={cycle}, component={component}, file={file_stem}, url={url}")
        result.update({"status": "error", "error": repr(exc)})
    return result


def main():
    log.info("Iniciando downloads NHANES com cache...")
    jobs = []
    for cycle, spec in NHANES_MANIFEST.items():
        year = spec["year"]
        for component, file_stem in spec["files"].items():
            jobs.append((cycle, year, component, file_stem))

    resultados = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(download_one)(cycle, year, component, file_stem)
        for cycle, year, component, file_stem in jobs
    )

    n_ok = int(np.sum([r["status"] == "ok" for r in resultados]))
    n_error = int(np.sum([r["status"] == "error" for r in resultados]))
    output = {
        "script": "00_download_nhanes.py",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "n_files": len(resultados),
        "n_ok": n_ok,
        "n_error": n_error,
        "results": resultados,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"Concluído. OK={n_ok}, ERROS={n_error}. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
