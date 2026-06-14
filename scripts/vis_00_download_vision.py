#!/usr/bin/env python3
"""
Nome: vis_00_download_vision.py
Tarefa: Baixar com cache os arquivos NHANES de Vision (autorefração objetiva) + DEMO,
        via HTTP real ao CDC. Espelha scripts/00_download_nhanes.py / grip_00.

Ciclos com autorefração objetiva e nomes de variável idênticos (verificado no codebook):
  1999-2000 (VIX, N=6758) e 2001-2002 (VIX_B, N=7445).
Variáveis: VIXORSM/VIXORCM (OD esfera/cilindro), VIXOLSM/VIXOLCM (OS). 88=could not obtain.
Output: outputs/json/vis_00_download_vision.json; .xpt em data/raw/nhanes/.
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
OUTPUT_PATH = Path("outputs/json/vis_00_download_vision.json")
LOG_PATH = Path("outputs/logs/vis_00_download_vision.log")

VISION_MANIFEST = {
    "1999-2000": {"year": "1999", "files": {"vix": "VIX", "demo": "DEMO"}},
    "2001-2002": {"year": "2001", "files": {"vix": "VIX_B", "demo": "DEMO_B"}},
}

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def validar_conteudo(path: Path) -> None:
    content = path.read_bytes()[:512]
    low = content.lower()
    if b"<html" in low or b"<!doctype" in low:
        raise ValueError(f"ERRO: {path} contém HTML — provável página de erro/404")
    if len(content) < 80:
        raise ValueError(f"ERRO: {path} pequeno demais para XPT válido")
    if not content.startswith(b"HEADER RECORD"):
        raise ValueError(f"ERRO: {path} não começa com header XPT SAS")
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
        log.info(f"Cache hit: {path_local} ({path.stat().st_size/1024:.1f} KB)")
        validar_conteudo(path)
        return path
    log.info(f"Fetch: {url}")
    r = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    log.info(f"Salvo: {path_local} ({len(r.content)/1024:.1f} KB)")
    validar_conteudo(path)
    time.sleep(REQUEST_SLEEP_SECONDS)
    return path


def download_one(cycle, year, component, file_stem) -> dict:
    url = BASE_URL.format(year=year, file=file_stem)
    local_path = RAW_ROOT / year / f"{file_stem}.xpt"
    result = {"cycle": cycle, "year": year, "component": component, "file": file_stem,
              "url": url, "path": str(local_path), "status": "pending",
              "size_bytes": None, "sha256": None, "error": None}
    try:
        path = fetch_com_cache(url, local_path)
        result.update({"status": "ok", "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    except Exception as exc:
        log.exception(f"Falha: {cycle}/{component}/{file_stem}")
        result.update({"status": "error", "error": repr(exc)})
    return result


def main():
    jobs = [(c, s["year"], comp, stem) for c, s in VISION_MANIFEST.items()
            for comp, stem in s["files"].items()]
    resultados = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(download_one)(c, y, comp, stem) for c, y, comp, stem in jobs)
    n_ok = int(np.sum([r["status"] == "ok" for r in resultados]))
    n_error = int(np.sum([r["status"] == "error" for r in resultados]))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(
        {"script": "vis_00_download_vision.py", "n_files": len(resultados),
         "n_ok": n_ok, "n_error": n_error, "results": resultados}, indent=2, default=str), encoding="utf-8")
    log.info(f"Concluído. OK={n_ok}, ERROS={n_error}.")
    for r in resultados:
        print(f"  [{r['status']}] {r['cycle']} {r['component']:5} {r['file']:8} "
              f"{(r['size_bytes'] or 0)/1024:8.1f} KB  sha256={str(r['sha256'])[:16]}...")


if __name__ == "__main__":
    main()
