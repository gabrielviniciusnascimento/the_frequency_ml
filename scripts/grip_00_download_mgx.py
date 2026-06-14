#!/usr/bin/env python3
"""
Nome: grip_00_download_mgx.py
Tarefa: Baixar com cache os arquivos NHANES de Muscle Strength - Grip Test (MGX)
        e DEMO (idade/sexo), via HTTP real ao CDC. Espelha scripts/00_download_nhanes.py.
Output: outputs/json/grip_00_download_mgx.json; .xpt em data/raw/nhanes/.

Ciclos adultos com grip test: 2011-2012 (MGX_G) e 2013-2014 (MGX_H). Não há P_MGX.
"""

import logging
import json
from pathlib import Path
import time
import hashlib
import requests

import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp

N_JOBS = max(mp.cpu_count() - 1, 1)
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_SLEEP_SECONDS = 1
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}.xpt"
RAW_ROOT = Path("data/raw/nhanes")
OUTPUT_PATH = Path("outputs/json/grip_00_download_mgx.json")
LOG_PATH = Path("outputs/logs/grip_00_download_mgx.log")

# Apenas os dois ciclos adultos com o teste de grip + DEMO para idade/sexo.
MGX_MANIFEST = {
    "2011-2012": {"year": "2011", "files": {"mgx": "MGX_G", "demo": "DEMO_G"}},
    "2013-2014": {"year": "2013", "files": {"mgx": "MGX_H", "demo": "DEMO_H"}},
}

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def validar_conteudo(path: Path) -> None:
    """Rejeitar HTML / página de erro disfarçada de .xpt."""
    content = path.read_bytes()[:512]
    lowered = content.lower()
    if b"<html" in lowered or b"<!doctype" in lowered:
        raise ValueError(f"ERRO: {path} contém HTML — provável página de erro/404")
    if len(content) < 80:
        raise ValueError(f"ERRO: {path} pequeno demais para XPT válido")
    # XPT SAS transport começa com 'HEADER RECORD*******LIBRARY HEADER RECORD'
    if not content.startswith(b"HEADER RECORD"):
        raise ValueError(f"ERRO: {path} não começa com header XPT SAS (conteúdo inesperado)")
    log.info(f"Conteúdo validado (XPT): {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_com_cache(url: str, path_local: Path) -> Path:
    path = Path(path_local)
    if path.exists():
        log.info(f"Cache hit: {path_local} ({path.stat().st_size / 1024:.1f} KB)")
        validar_conteudo(path)
        return path
    log.info(f"Fetch: {url}")
    r = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    log.info(f"Salvo: {path_local} ({len(r.content) / 1024:.1f} KB)")
    validar_conteudo(path)
    time.sleep(REQUEST_SLEEP_SECONDS)
    return path


def download_one(cycle: str, year: str, component: str, file_stem: str) -> dict:
    url = BASE_URL.format(year=year, file=file_stem)
    local_path = RAW_ROOT / year / f"{file_stem}.xpt"
    result = {
        "cycle": cycle, "year": year, "component": component, "file": file_stem,
        "url": url, "path": str(local_path), "status": "pending",
        "size_bytes": None, "sha256": None, "error": None,
    }
    try:
        path = fetch_com_cache(url, local_path)
        result.update({"status": "ok", "size_bytes": path.stat().st_size,
                       "sha256": sha256_file(path)})
    except Exception as exc:
        log.exception(f"Falha no download: {cycle}/{component}/{file_stem}")
        result.update({"status": "error", "error": repr(exc)})
    return result


def main():
    log.info("Iniciando downloads MGX/DEMO com cache...")
    jobs = [(cycle, spec["year"], comp, stem)
            for cycle, spec in MGX_MANIFEST.items()
            for comp, stem in spec["files"].items()]
    resultados = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(download_one)(c, y, comp, stem) for c, y, comp, stem in jobs)

    n_ok = int(np.sum([r["status"] == "ok" for r in resultados]))
    n_error = int(np.sum([r["status"] == "error" for r in resultados]))
    output = {
        "script": "grip_00_download_mgx.py", "n_jobs": N_JOBS,
        "n_files": len(resultados), "n_ok": n_ok, "n_error": n_error,
        "results": resultados,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info(f"Concluído. OK={n_ok}, ERROS={n_error}. Output: {OUTPUT_PATH}")
    for r in resultados:
        print(f"  [{r['status']}] {r['cycle']} {r['component']:5} {r['file']:8} "
              f"{(r['size_bytes'] or 0)/1024:8.1f} KB  sha256={str(r['sha256'])[:16]}...")


if __name__ == "__main__":
    main()
