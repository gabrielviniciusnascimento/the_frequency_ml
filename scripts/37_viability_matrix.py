#!/usr/bin/env python3
"""
Nome: 37_viability_matrix.py
Tarefa: Construir empiricamente a matriz de viabilidade centrada em (cycle, SEQN)
        para todos os componentes candidatos do NHANES cruzados com a audiometria
        adulta 20-69 anos nos 5 ciclos elegíveis (1999-2000, 2001-2002, 2003-2004,
        2011-2012, 2015-2016), preservando o cofre de replicação (zero cálculo de
        associação X <-> Y; apenas cobertura, cardinalidade, pesos amostrais NCHS,
        graus de liberdade de desenho e % de ausência).

Input:  Cache local em data/raw/nhanes/ (baixa do CDC apenas o que faltar).
Output: outputs/json/37_viability_matrix.json
        outputs/json/37_viability_matrix_summary.csv
        outputs/logs/37_viability_matrix.log
Run:    python scripts/37_viability_matrix.py [--force] [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from _shape_space import lib_versions  # type: ignore
except Exception:
    import platform
    import scipy
    import sklearn

    def lib_versions() -> dict:
        return {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        }


# ── Constantes ───────────────────────────────────────────────────────
RANDOM_STATE = 42
AGE_MIN_BASE = 20
AGE_MAX_BASE = 69
MIN_COMPLETENESS = 10
ANY25_DB = 25.0
SAS_ZERO_EPS = 1e-12

RAW_ROOT = ROOT / "data" / "raw" / "nhanes"
MORT_ROOT = RAW_ROOT / "mortality"
FEATURE_CSV = ROOT / "data" / "processed" / "frequencia_feature_matrix_v1.csv"
BRUTO_CSV = ROOT / "data" / "processed" / "frequencia_bruto.csv"
OUTPUT_JSON = ROOT / "outputs" / "json" / "37_viability_matrix.json"
OUTPUT_CSV = ROOT / "outputs" / "json" / "37_viability_matrix_summary.csv"
LOG_PATH = ROOT / "outputs" / "logs" / "37_viability_matrix.log"

NHANES_XPT_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}.xpt"
NCHS_MORT_URLS = [
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{y1}_{y2}_MORT_2019_PUBLIC.dat",
]

REQUEST_TIMEOUT = 60
REQUEST_SLEEP = 0.4

ADULT_20_69_CYCLES = [
    {"cycle": "1999-2000", "year": "1999", "y1": "1999", "y2": "2000", "aux": "AUX1", "demo": "DEMO", "auq": "AUQ", "partition": "discovery"},
    {"cycle": "2001-2002", "year": "2001", "y1": "2001", "y2": "2002", "aux": "AUX_B", "demo": "DEMO_B", "auq": "AUQ_B", "partition": "discovery"},
    {"cycle": "2003-2004", "year": "2003", "y1": "2003", "y2": "2004", "aux": "AUX_C", "demo": "DEMO_C", "auq": "AUQ_C", "partition": "replication"},
    {"cycle": "2011-2012", "year": "2011", "y1": "2011", "y2": "2012", "aux": "AUX_G", "demo": "DEMO_G", "auq": "AUQ_G", "partition": "discovery"},
    {"cycle": "2015-2016", "year": "2015", "y1": "2015", "y2": "2016", "aux": "AUX_I", "demo": "DEMO_I", "auq": "AUQ_I", "partition": "replication"},
]

FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
AUX_R_MAP = {500: "AUXU500R", 1000: "AUXU1K1R", 2000: "AUXU2KR", 3000: "AUXU3KR", 4000: "AUXU4KR", 6000: "AUXU6KR", 8000: "AUXU8KR"}
AUX_L_MAP = {500: "AUXU500L", 1000: "AUXU1K1L", 2000: "AUXU2KL", 3000: "AUXU3KL", 4000: "AUXU4KL", 6000: "AUXU6KL", 8000: "AUXU8KL"}
AUX_THR_RAW_COLS = list(AUX_R_MAP.values()) + list(AUX_L_MAP.values())


# ── Especificação declarativa dos componentes a auditar ──────────────
@dataclass
class ComponentSpec:
    key: str
    description: str
    lockbox_regime: str  # CONTAMINATED_POSTHOC | UNTOUCHED_LOCKBOX_VALID | SINGLE_CYCLE_EXPLORATORY
    age_min: int = 20
    age_max: int = 69
    # Mapa ciclo -> lista ordenada de stems candidatos no CDC (tenta em ordem até achar 200 OK)
    candidates_by_cycle: dict[str, list[str]] = field(default_factory=dict)
    # Tipo de fonte: "xpt", "aux_embedded", "auq_embedded", "mortality_dat"
    source_kind: str = "xpt"
    # Colunas primárias esperadas (qualquer/todas conforme rule)
    target_cols: list[str] = field(default_factory=list)
    validity_mode: str = "any"  # "any" | "all" | "custom"
    invalid_codes: set[float] = field(default_factory=set)
    # Overrides específicos de partição (ex.: CFQ não tem 03-04 nem 15-16, logo 11-12 vira replicação)
    partition_override: dict[str, str] = field(default_factory=dict)
    notes: str = ""


COMPONENT_CATALOG: list[ComponentSpec] = [
    ComponentSpec(
        key="auq_self_report",
        description="Questionário Auditivo (AUQ054/AUQ010 geral, AUQ101/AUQ100 ruído, AUQ191/AUQ190 zumbido)",
        lockbox_regime="CONTAMINATED_POSTHOC",
        source_kind="auq_embedded",
        notes="Já explorado em 16, 23, 34 e virada prévia; audita harmonização 1999-2004 vs 2011-2016.",
    ),
    ComponentSpec(
        key="auxtym_summary_in_aux",
        description="Timpanometria — variáveis escalares já embutidas em AUX (complacência, pressão, volume, largura)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        source_kind="aux_embedded",
        target_cols=["AUXTCOMR", "AUXTCOML", "AUXTMEPR", "AUXTMEPL", "AUXTPVR", "AUXTPVL", "AUXTWIDR", "AUXTWIDL"],
        validity_mode="any",
        invalid_codes={888.0, 999.0, 8888.0, 9999.0, 88888.0, 99999.0},
        notes="Ingeridas em 01/03 mas nunca cruzadas com desfechos; filtro para perda condutiva.",
    ),
    ComponentSpec(
        key="auxtym_curves",
        description="Timpanometria — séries brutas de pressão-complacência (.xpt separado)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["AUXTYM", "AUXTYM1"],
            "2001-2002": ["AUXTYM_B"],
            "2003-2004": ["AUXTYM_C"],
            "2011-2012": ["AUXTYM_G"],
            "2015-2016": ["AUXTYM_I"],
        },
        validity_mode="custom",
        invalid_codes={88888.0, 99999.0},
    ),
    ComponentSpec(
        key="auxar_waveform",
        description="Reflexo Acústico Estapediano — séries temporais brutas 1 kHz / 2 kHz (.xpt separado)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["AUXAR", "AUXAR1"],
            "2001-2002": ["AUXAR_B"],
            "2003-2004": ["AUXAR_C"],
            "2011-2012": ["AUXAR_G"],
            "2015-2016": ["AUXAR_I"],
        },
        validity_mode="custom",
        invalid_codes={88888.0, 99999.0},
        notes="Triagem em nível único (105 dB HL, pulso curto); pista fisiológica, não reflex decay de 10 s.",
    ),
    ComponentSpec(
        key="auxwbr_wideband",
        description="Reflectância/absorvância de banda larga (Titan, só 2015-2016) — AUXWBR_I",
        lockbox_regime="SINGLE_CYCLE_EXPLORATORY",
        candidates_by_cycle={"2015-2016": ["AUXWBR_I"]},
        validity_mode="custom",
        notes="Adicionado na revisão: medida de orelha média de banda larga, só em 2015-16.",
    ),
    ComponentSpec(
        key="bax_balance",
        description="Exame de Equilíbrio / Romberg modificado incl. Condição 4 em espuma (BAX)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        age_min=40,
        age_max=69,
        candidates_by_cycle={
            "1999-2000": ["BAX"],
            "2001-2002": ["BAX_B"],
            "2003-2004": ["BAX_C"],
            "2011-2012": ["BAX_G"],
            "2015-2016": ["BAX_I"],
        },
        target_cols=["BAXPFC41", "BAXPFC42"],  # FIX revisão: nomes reais (BAX/BAX_B/BAX_C), condição 4 espuma olhos fechados
        validity_mode="any",
        invalid_codes={8.0, 9.0, 88.0, 99.0},
        partition_override={"1999-2000": "discovery", "2001-2002": "discovery", "2003-2004": "replication"},
    ),
    ComponentSpec(
        key="mgx_grip",
        description="Força de Preensão Manual com lado anatômico via MGATHAND (MGX)",
        lockbox_regime="SINGLE_CYCLE_EXPLORATORY",
        candidates_by_cycle={
            "1999-2000": ["MGX"],
            "2001-2002": ["MGX_B"],
            "2003-2004": ["MGX_C"],
            "2011-2012": ["MGX_G"],
            "2015-2016": ["MGX_I"],
        },
        target_cols=["MGATHAND", "MGXH1T1", "MGXH2T1"],
        validity_mode="all",
        invalid_codes={8.0, 9.0, 999.0, 9999.0},
        notes="Marginais auditadas em grip_00-04; apenas 2011-2012 cruza com AUX 20-69.",
    ),
    ComponentSpec(
        key="vix_vision",
        description="Refração Ocular Objetiva / Acuidade Visual (VIX)",
        lockbox_regime="CONTAMINATED_POSTHOC",
        candidates_by_cycle={
            "1999-2000": ["VIX"],
            "2001-2002": ["VIX_B"],
            "2003-2004": ["VIX_C"],
            "2011-2012": ["VIX_G"],
            "2015-2016": ["VIX_I"],
        },
        target_cols=["VIXORSM", "VIXOLSM"],
        validity_mode="all",
        invalid_codes={88.0, 99.0, 888.0, 999.0},
        notes="1999-2002 já auditados em vis_00-04; VIX_C (2003-2004) nunca havia sido baixado.",
    ),
    ComponentSpec(
        key="rxq_rx_meds",
        description="Medicamentos Prescritos nos últimos 30 dias — tabela 1:N (RXQ_RX)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["RXQ_RX"],
            "2001-2002": ["RXQ_RX_B"],
            "2003-2004": ["RXQ_RX_C"],
            "2011-2012": ["RXQ_RX_G"],
            "2015-2016": ["RXQ_RX_I"],
        },
        validity_mode="custom",
        notes="Reduzido 1:N -> 1:1 por SEQN; audita N marginal de diuréticos de alça, aminoglicosídeos e platinas.",
    ),
    ComponentSpec(
        key="mcq_cancer",
        description="Condições Médicas — Histórico de Câncer MCQ220/MCQ230 (MCQ)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["MCQ"],
            "2001-2002": ["MCQ_B"],
            "2003-2004": ["MCQ_C"],
            "2011-2012": ["MCQ_G"],
            "2015-2016": ["MCQ_I"],
        },
        target_cols=["MCQ220"],
        validity_mode="all",
        invalid_codes={7.0, 9.0, 77.0, 99.0},
    ),
    ComponentSpec(
        key="dpq_phq9",
        description="Sintomas Depressivos PHQ-9 e Fadiga DPQ040 (DPQ)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["DPQ"],
            "2001-2002": ["DPQ_B"],
            "2003-2004": ["DPQ_C"],
            "2011-2012": ["DPQ_G"],
            "2015-2016": ["DPQ_I"],
        },
        target_cols=["DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090"],
        validity_mode="any",
        invalid_codes={7.0, 9.0, 77.0, 99.0},
        partition_override={"2011-2012": "discovery", "2015-2016": "replication"},
    ),
    ComponentSpec(
        key="cfq_cognition",
        description="Função Cognitiva — Digit Symbol Substitution Test CFDDS (CFQ)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        age_min=60,
        age_max=69,
        candidates_by_cycle={
            "1999-2000": ["CFQ"],
            "2001-2002": ["CFQ_B"],
            "2003-2004": ["CFQ_C"],
            "2011-2012": ["CFQ_G"],
            "2015-2016": ["CFQ_I"],
        },
        target_cols=["CFDDS", "CFDRIGHT"],  # FIX revisão: DSST = CFDRIGHT em 1999-2002
        validity_mode="any",
        invalid_codes={777.0, 888.0, 999.0},
        partition_override={"1999-2000": "discovery", "2001-2002": "discovery", "2011-2012": "replication"},
    ),
    ComponentSpec(
        key="ocq_occupation",
        description="Ocupação e Situação Laboral (OCQ)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["OCQ"],
            "2001-2002": ["OCQ_B"],
            "2003-2004": ["OCQ_C"],
            "2011-2012": ["OCQ_G"],
            "2015-2016": ["OCQ_I"],
        },
        target_cols=["OCD150", "OCQ150"],
        validity_mode="any",
        invalid_codes={7.0, 9.0, 77.0, 99.0},
    ),
    ComponentSpec(
        key="lab_ghb",
        description="Laboratório — Hemoglobina Glicada HbA1c LBXGH (LAB10 / L10 / GHB)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["LAB10", "GHB"],
            "2001-2002": ["L10_B", "GHB_B"],
            "2003-2004": ["L10_C", "GHB_C"],
            "2011-2012": ["GHB_G"],
            "2015-2016": ["GHB_I"],
        },
        target_cols=["LBXGH"],
        validity_mode="any",
    ),
    ComponentSpec(
        key="lab_biopro_creatinine",
        description="Laboratório — Bioquímica Padrão / Creatinina Sérica LBXSCR/LBDSCR (LAB18 / L40 / BIOPRO)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["LAB18", "BIOPRO"],
            "2001-2002": ["L40_B", "BIOPRO_B"],
            "2003-2004": ["L40_C", "BIOPRO_C"],
            "2011-2012": ["BIOPRO_G"],
            "2015-2016": ["BIOPRO_I"],
        },
        target_cols=["LBXSCR", "LBDSCR"],
        validity_mode="any",
    ),
    ComponentSpec(
        key="lab_alb_cr",
        description="Laboratório — Albumina e Creatinina Urinárias URXUMA/URXUCR (LAB16 / L16 / ALB_CR)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["LAB16", "ALB_CR"],
            "2001-2002": ["L16_B", "ALB_CR_B"],
            "2003-2004": ["L16_C", "ALB_CR_C"],
            "2011-2012": ["ALB_CR_G"],
            "2015-2016": ["ALB_CR_I"],
        },
        target_cols=["URXUMA", "URXUCR"],
        validity_mode="all",
    ),
    ComponentSpec(
        key="exam_bpx",
        description="Exame Físico — Pressão Arterial Sistólica/Diastólica (BPX)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["BPX"],
            "2001-2002": ["BPX_B"],
            "2003-2004": ["BPX_C"],
            "2011-2012": ["BPX_G"],
            "2015-2016": ["BPX_I"],
        },
        target_cols=["BPXSY1", "BPXSY2", "BPXSY3"],
        validity_mode="any",
    ),
    ComponentSpec(
        key="exam_bmx",
        description="Exame Físico — Antropometria / IMC BMXBMI (BMX)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["BMX"],
            "2001-2002": ["BMX_B"],
            "2003-2004": ["BMX_C"],
            "2011-2012": ["BMX_G"],
            "2015-2016": ["BMX_I"],
        },
        target_cols=["BMXBMI"],
        validity_mode="any",
    ),
    ComponentSpec(
        key="lab_pbcd_metals",
        description="Laboratório — Chumbo e Cádmio Sanguíneos LBXBPB/LBXBCD (LAB06 / L06 / L06BMT_C / PBCD)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        candidates_by_cycle={
            "1999-2000": ["LAB06", "PBCD"],
            "2001-2002": ["L06_B", "PBCD_B"],
            "2003-2004": ["L06BMT_C", "L06_C", "PBCD_C"],
            "2011-2012": ["PBCD_G"],
            "2015-2016": ["PBCD_I"],
        },
        target_cols=["LBXBPB", "LBXBCD"],
        validity_mode="all",
    ),
    ComponentSpec(
        key="lmf_mortality",
        description="NCHS Public-Use Linked Mortality Files (.dat largura fixa — seguimento de óbito)",
        lockbox_regime="UNTOUCHED_LOCKBOX_VALID",
        source_kind="mortality_dat",
        notes="Reporta apenas elegibilidade e total marginal de óbitos na coorte (sem estratificar por AUX).",
    ),
]


# ── Utilitários HTTP, Cache, Hash e Leitura SAS Segura ───────────────
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "the-frequency-ml-viability-audit/1.0"})
    return s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_non_html(path: Path, min_bytes: int = 80) -> None:
    head = path.read_bytes()[:512].lower()
    if b"<html" in head or b"<!doctype" in head:
        raise ValueError(f"Arquivo contém HTML (erro/redirect): {path}")
    if path.stat().st_size < min_bytes:
        raise ValueError(f"Arquivo truncado (<{min_bytes} bytes): {path}")


def find_existing_xpt_cache(year: str, stem: str) -> Path | None:
    search_dirs = [
        RAW_ROOT / year,
        ROOT / "data" / "raw" / "mgx",
        ROOT / "data" / "raw" / "vision",
        ROOT / "data" / "raw",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in (".xpt", ".XPT"):
            cand = d / f"{stem}{ext}"
            if cand.exists():
                return cand
    return None


def resolve_or_download_xpt(
    year: str,
    candidates: list[str],
    log: logging.Logger,
) -> dict:
    """Tenta resolver no cache local ou baixar do CDC o primeiro stem válido da lista."""
    for stem in candidates:
        cached = find_existing_xpt_cache(year, stem)
        if cached is not None:
            try:
                validate_non_html(cached, min_bytes=80)
                return {
                    "status": "ok_cache",
                    "stem": stem,
                    "path": str(cached),
                    "url": NHANES_XPT_URL.format(year=year, stem=stem),
                    "size_bytes": cached.stat().st_size,
                    "sha256": sha256_file(cached),
                }
            except Exception as exc:
                log.warning(f"Cache inválido descartado ({cached}): {exc}")

    session = make_session()
    tried_errors: list[str] = []
    for stem in candidates:
        url = NHANES_XPT_URL.format(year=year, stem=stem)
        dest = RAW_ROOT / year / f"{stem}.xpt"
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                tried_errors.append(f"{stem}:404")
                continue
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            validate_non_html(dest, min_bytes=80)
            time.sleep(REQUEST_SLEEP)
            log.info(f"Baixado XPT: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
            return {
                "status": "ok_downloaded",
                "stem": stem,
                "path": str(dest),
                "url": url,
                "size_bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        except Exception as exc:
            if dest.exists():
                dest.unlink(missing_ok=True)
            tried_errors.append(f"{stem}:{exc!r}")

    all_404 = bool(tried_errors) and all(err.endswith(":404") for err in tried_errors)
    return {
        "status": "not_published_404" if all_404 else "download_error",
        "stem": None,
        "path": None,
        "url": None,
        "errors": tried_errors,
    }


def resolve_or_download_mortality(y1: str, y2: str, log: logging.Logger) -> dict:
    fname = f"NHANES_{y1}_{y2}_MORT_2019_PUBLIC.dat"
    dest = MORT_ROOT / fname
    if dest.exists():
        try:
            validate_non_html(dest, min_bytes=500)
            return {
                "status": "ok_cache",
                "path": str(dest),
                "url": NCHS_MORT_URLS[0].format(y1=y1, y2=y2),
                "size_bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        except Exception:
            dest.unlink(missing_ok=True)

    session = make_session()
    errors: list[str] = []
    for tmpl in NCHS_MORT_URLS:
        url = tmpl.format(y1=y1, y2=y2)
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                errors.append(f"{url}:404")
                continue
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            validate_non_html(dest, min_bytes=500)
            time.sleep(REQUEST_SLEEP)
            log.info(f"Baixado LMF: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
            return {
                "status": "ok_downloaded",
                "path": str(dest),
                "url": url,
                "size_bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        except Exception as exc:
            dest.unlink(missing_ok=True)
            errors.append(f"{url}:{exc!r}")

    return {"status": "download_error", "path": None, "url": None, "errors": errors}


def read_xpt_clean(path: Path) -> pd.DataFrame:
    """Lê SAS XPORT padronizando colunas UPPER e zerando underflow IBM float (~5.3976e-79)."""
    df = pd.read_sas(path, format="xport")
    df.columns = [str(c).strip().upper() for c in df.columns]
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        vals = df[num_cols].to_numpy(copy=True)
        tiny_mask = np.isfinite(vals) & (np.abs(vals) < SAS_ZERO_EPS)
        vals[tiny_mask] = 0.0
        df[num_cols] = vals
    if "SEQN" in df.columns:
        df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").round().astype("Int64")
    return df


def parse_nchs_mortality_dat(path: Path) -> pd.DataFrame:
    """Layout público NCHS 2019: SEQN (1-14), ELIGSTAT (15), MORTSTAT (16), PERMTH_INT (43-45), PERMTH_EXM (46-48)."""
    rows = []
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if len(line) < 16:
                continue
            seqn_str = line[0:14].strip()
            if not seqn_str.isdigit():
                continue
            elig_str = line[14:15].strip()
            mort_str = line[15:16].strip()
            permth_exm_str = line[45:48].strip() if len(line) >= 48 else ""
            rows.append({
                "SEQN": int(seqn_str),
                "ELIGSTAT": int(elig_str) if elig_str.isdigit() else np.nan,
                "MORTSTAT": int(mort_str) if mort_str.isdigit() else np.nan,
                "PERMTH_EXM": int(permth_exm_str) if permth_exm_str.isdigit() else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["SEQN"] = out["SEQN"].astype("Int64")
    return out


# ── Construção da Base de Referência AUX + DEMO + AUQ (5 Ciclos) ─────
def build_base_reference(log: logging.Logger) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    """Carrega/baixa AUX, DEMO e AUQ dos 5 ciclos 20-69 e audita onde moram os pesos WTSAU*."""
    base_frames: list[pd.DataFrame] = []
    cycle_raw_tables: dict[str, dict] = {}
    weight_audit_by_cycle: dict[str, dict] = {}

    for spec in ADULT_20_69_CYCLES:
        cyc = spec["cycle"]
        yr = spec["year"]
        aux_meta = resolve_or_download_xpt(yr, [spec["aux"]], log)
        demo_meta = resolve_or_download_xpt(yr, [spec["demo"]], log)
        auq_meta = resolve_or_download_xpt(yr, [spec["auq"]], log)

        if not aux_meta["path"] or not demo_meta["path"]:
            raise RuntimeError(f"Falha ao obter AUX/DEMO de base para {cyc}: {aux_meta} / {demo_meta}")

        aux_df = read_xpt_clean(Path(aux_meta["path"]))
        demo_df = read_xpt_clean(Path(demo_meta["path"]))
        auq_df = read_xpt_clean(Path(auq_meta["path"])) if auq_meta["path"] else pd.DataFrame()

        cycle_raw_tables[cyc] = {"aux": aux_df, "demo": demo_df, "auq": auq_df}

        aux_wt_cols = sorted([c for c in aux_df.columns if c.startswith("WT")])
        demo_wt_cols = sorted([c for c in demo_df.columns if c.startswith("WT")])
        weight_audit_by_cycle[cyc] = {
            "aux_wt_cols": aux_wt_cols,
            "demo_wt_cols": demo_wt_cols,
        }

        # Limiares limpos para reproduzir exatamente os 3 denominadores do repo
        thr_mat = pd.DataFrame(index=aux_df.index)
        for col in AUX_THR_RAW_COLS:
            if col in aux_df.columns:
                s = pd.to_numeric(aux_df[col], errors="coerce")
                s = s.mask(s.isin([666.0, 888.0]) | (s < -10.0) | (s > 130.0), np.nan)
                thr_mat[col] = s
            else:
                thr_mat[col] = np.nan

        n_valid_thr = thr_mat.notna().sum(axis=1)
        any25_flag = (thr_mat > ANY25_DB).any(axis=1)

        sub = pd.DataFrame({
            "cycle": cyc,
            "partition_default": spec["partition"],
            "SEQN": aux_df["SEQN"],
            "n_valid_thresholds_14": n_valid_thr.astype("int16"),
            "is_complete10": (n_valid_thr >= MIN_COMPLETENESS),
            "is_any25": (n_valid_thr >= MIN_COMPLETENESS) & any25_flag,
        })

        # Trazer pesos de AUX se existirem lá
        for wcol in ["WTSAU2YR", "WTSAU4YR"]:
            if wcol in aux_df.columns:
                sub[f"{wcol}_FROM_AUX"] = pd.to_numeric(aux_df[wcol], errors="coerce")

        # Merge DEMO (idade, PSU, estrato e pesos DEMO)
        demo_cols = [c for c in ["SEQN", "RIDAGEYR", "RIAGENDR", "SDMVSTRA", "SDMVPSU", "WTMEC2YR", "WTINT2YR", "WTSAU2YR", "WTSAU4YR"] if c in demo_df.columns]
        sub = sub.merge(demo_df[demo_cols], on="SEQN", how="left", validate="one_to_one")

        # Consolidar WTSAU2YR e WTSAU4YR de DEMO ou AUX
        for wcol in ["WTSAU2YR", "WTSAU4YR"]:
            c_demo = sub[wcol] if wcol in sub.columns else pd.Series(np.nan, index=sub.index)
            c_aux = sub[f"{wcol}_FROM_AUX"] if f"{wcol}_FROM_AUX" in sub.columns else pd.Series(np.nan, index=sub.index)
            sub[wcol] = c_demo.combine_first(c_aux)

        age = pd.to_numeric(sub["RIDAGEYR"], errors="coerce")
        sub = sub[(age >= AGE_MIN_BASE) & (age <= AGE_MAX_BASE)].copy()
        base_frames.append(sub)

    base = pd.concat(base_frames, ignore_index=True)

    # Checagem contra arquivos já processados no repo (se existirem)
    repo_cross_check: dict = {"feature_csv_exists": FEATURE_CSV.exists(), "bruto_csv_exists": BRUTO_CSV.exists()}
    if FEATURE_CSV.exists():
        feat_cols = pd.read_csv(FEATURE_CSV, nrows=1).columns.tolist()
        repo_cross_check["feature_csv_has_WTSAU2YR"] = "WTSAU2YR" in feat_cols
        repo_cross_check["feature_csv_has_WTSAU4YR"] = "WTSAU4YR" in feat_cols
    if BRUTO_CSV.exists():
        b_df = pd.read_csv(BRUTO_CSV, usecols=["cycle", "RIDAGEYR"], low_memory=False)
        b_age = pd.to_numeric(b_df["RIDAGEYR"], errors="coerce")
        b_20_69 = b_df[(b_age >= 20) & (b_age <= 69)]
        repo_cross_check["bruto_20_69_counts_by_cycle"] = {
            str(k): int(v) for k, v in b_20_69["cycle"].value_counts().to_dict().items()
        }

    audit_info = {
        "n_raw_20_69_5cycles": int(len(base)),
        "n_complete10_20_69_5cycles": int(base["is_complete10"].sum()),
        "n_any25_20_69_5cycles": int(base["is_any25"].sum()),
        "counts_by_cycle_raw_20_69": {str(k): int(v) for k, v in base["cycle"].value_counts().sort_index().items()},
        "weight_columns_location_by_cycle": weight_audit_by_cycle,
        "repo_existing_files_check": repo_cross_check,
    }
    return base, cycle_raw_tables, audit_info


# ── Redutores e Validadores por Componente (Zero leitura de Y vs AUX) ─
def extract_person_validity(
    comp: ComponentSpec,
    cyc: str,
    df_comp: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Retorna DataFrame 1:1 ['SEQN', 'comp_valid', ...pesos próprios...] + metadados estruturais."""
    meta: dict = {
        "raw_rows": int(len(df_comp)),
        "unique_seqn": int(df_comp["SEQN"].dropna().nunique()) if "SEQN" in df_comp.columns else 0,
    }
    if df_comp.empty or "SEQN" not in df_comp.columns:
        return pd.DataFrame(columns=["SEQN", "comp_valid"]), meta

    df_comp = df_comp[df_comp["SEQN"].notna()].copy()
    is_1_to_n = bool(df_comp["SEQN"].duplicated().any())
    meta["cardinality_observed"] = "1:N" if is_1_to_n else "1:1"

    # Detectar colunas de peso de subamostra próprias no componente (ex. WTSH2YR, WTSBA2YR)
    own_wt_cols = sorted([
        c for c in df_comp.columns
        if c.startswith("WTS") and (c.endswith("2YR") or c.endswith("4YR"))
    ])
    meta["component_subsample_wt_cols"] = own_wt_cols
    # FIX revisão: WTSAU* repetidos no componente colidem com os da base no merge; descartar aqui
    df_comp = df_comp.drop(columns=[c for c in df_comp.columns if c.startswith("WTSAU")], errors="ignore")
    own_wt_cols = [c for c in own_wt_cols if not c.startswith("WTSAU")]

    # Caso especial 1: RXQ_RX (tabela 1:N de medicamentos)
    if comp.key == "rxq_rx_meds":
        # FIX revisão: 1999-2002 usa RXD030 (uso no mês) e RXD240B (nome genérico) em vez de RXDUSE/RXDDRUG
        use_col = "RXDUSE" if "RXDUSE" in df_comp.columns else ("RXD030" if "RXD030" in df_comp.columns else None)
        rxduse = pd.to_numeric(df_comp[use_col], errors="coerce") if use_col else pd.Series(np.nan, index=df_comp.index)
        valid_use = rxduse.isin([1.0, 2.0])
        drug_str = pd.Series("", index=df_comp.index, dtype="object")
        drug_col = "RXDDRUG" if "RXDDRUG" in df_comp.columns else ("RXD240B" if "RXD240B" in df_comp.columns else None)
        if drug_col:
            drug_str = df_comp[drug_col].apply(
                lambda v: v.decode("latin-1", errors="ignore").strip().upper()
                if isinstance(v, (bytes, bytearray))
                else ("" if pd.isna(v) else str(v).strip().upper())
            )
        is_loop = drug_str.str.contains(r"FUROSEMIDE|BUMETANIDE|TORSEMIDE|ETHACRYNIC", regex=True, na=False)
        is_amino = drug_str.str.contains(r"GENTAMICIN|TOBRAMYCIN|AMIKACIN|STREPTOMYCIN|NEOMYCIN", regex=True, na=False)
        is_plat = drug_str.str.contains(r"CISPLATIN|CARBOPLATIN|OXALIPLATIN", regex=True, na=False)

        tmp = pd.DataFrame({
            "SEQN": df_comp["SEQN"],
            "comp_valid": valid_use,
            "any_rx_30d": rxduse == 1.0,
            "loop_diuretic_30d": is_loop,
            "aminoglycoside_30d": is_amino,
            "platinum_30d": is_plat,
        })
        person = tmp.groupby("SEQN", as_index=False).max()
        return person, meta

    # Caso especial 2: Séries brutas AUXAR / AUXTYM (1:1 ou 1:N)
    if comp.key in {"auxar_waveform", "auxtym_curves", "auxwbr_wideband"}:
        ignore_prefixes = ("SEQN", "WT")
        num_cols = [
            c for c in df_comp.select_dtypes(include=[np.number]).columns
            if not c.startswith(ignore_prefixes)
        ]
        # Identificar colunas de pontos da curva (>= 20 colunas numéricas de série ou colunas de sinal)
        mat = df_comp[num_cols].copy() if num_cols else pd.DataFrame(index=df_comp.index)
        for code in comp.invalid_codes:
            mat = mat.mask(mat == code, np.nan)
        row_valid = mat.notna().sum(axis=1) >= 10 if not mat.empty else pd.Series(False, index=df_comp.index)

        counts_per_seqn = df_comp.groupby("SEQN").size()
        meta["rows_per_seqn_median"] = float(counts_per_seqn.median()) if len(counts_per_seqn) else 0.0
        meta["rows_per_seqn_max"] = int(counts_per_seqn.max()) if len(counts_per_seqn) else 0
        meta["n_numeric_signal_cols"] = len(num_cols)
        meta["sample_cols_head"] = num_cols[:10]
        if "RFXLEVEL" in df_comp.columns:  # FIX revisão: 2015-16 Titan = 3 níveis x 2 freq x 2 orelhas
            lv = df_comp.groupby("SEQN")["RFXLEVEL"].nunique()
            meta["rfx_levels_per_seqn_dist"] = {str(k): int(v) for k, v in lv.value_counts().sort_index().items()}
            meta["n_seqn_with_12_curves"] = int((df_comp.groupby("SEQN").size() == 12).sum())

        tmp = pd.DataFrame({"SEQN": df_comp["SEQN"], "comp_valid": row_valid})
        for w in own_wt_cols:
            tmp[w] = pd.to_numeric(df_comp[w], errors="coerce")
        person = tmp.groupby("SEQN", as_index=False).max()
        return person, meta

    # Caso especial 3: Questionário AUQ harmonizado (1999-2004 vs 2011-2016)
    if comp.key == "auq_self_report":
        cand_general = [c for c in ["AUQ054", "AUQ130"] if c in df_comp.columns]  # FIX revisão: 1999-2004 = AUQ130
        cand_noise = [c for c in ["AUQ100"] if c in df_comp.columns]  # FIX revisão: AUQ101 não existe; item é AUQ100 (só 2011-2016)
        cand_tinn = [c for c in ["AUQ191", "AUQ190"] if c in df_comp.columns]
        meta["auq_cols_found"] = {"general": cand_general, "speech_in_noise": cand_noise, "tinnitus": cand_tinn}

        def clean_auq_col(cols: list[str]) -> pd.Series:
            s = pd.Series(np.nan, index=df_comp.index, dtype="float64")
            for col in cols:
                v = pd.to_numeric(df_comp[col], errors="coerce")
                v = v.mask(v.isin([7.0, 9.0, 77.0, 99.0]), np.nan)
                s = s.combine_first(v)
            return s.notna()

        g_ok = clean_auq_col(cand_general)
        n_ok = clean_auq_col(cand_noise)
        t_ok = clean_auq_col(cand_tinn)
        person = pd.DataFrame({
            "SEQN": df_comp["SEQN"],
            "comp_valid": g_ok | n_ok | t_ok,
            "auq_general_valid": g_ok,
            "auq_noise_valid": n_ok,
            "auq_tinnitus_valid": t_ok,
        }).drop_duplicates(subset=["SEQN"])
        return person, meta

    # Caso especial 4: Linked Mortality Files (.dat)
    if comp.key == "lmf_mortality":
        elig = df_comp["ELIGSTAT"] == 1
        mort_ok = elig & df_comp["MORTSTAT"].isin([0, 1])
        person = pd.DataFrame({
            "SEQN": df_comp["SEQN"],
            "comp_valid": mort_ok,
            "is_deceased_marginal": mort_ok & (df_comp["MORTSTAT"] == 1),
        }).drop_duplicates(subset=["SEQN"])
        return person, meta

    # Caso padrão: colunas-alvo explícitas em tabela 1:1
    present_targets = [c for c in comp.target_cols if c in df_comp.columns]
    meta["target_cols_found"] = present_targets
    if not present_targets:
        return pd.DataFrame({"SEQN": df_comp["SEQN"].drop_duplicates(), "comp_valid": False}), meta

    mat = pd.DataFrame(index=df_comp.index)
    for col in present_targets:
        v = pd.to_numeric(df_comp[col], errors="coerce")
        if comp.invalid_codes:
            v = v.mask(v.isin(comp.invalid_codes), np.nan)
        mat[col] = v

    if comp.validity_mode == "all":
        valid_mask = mat.notna().all(axis=1)
    else:
        valid_mask = mat.notna().any(axis=1)

    out = pd.DataFrame({"SEQN": df_comp["SEQN"], "comp_valid": valid_mask})
    if comp.key == "mcq_cancer" and "MCQ220" in mat.columns:
        out["cancer_yes_marginal"] = mat["MCQ220"] == 1.0
    for w in own_wt_cols:
        out[w] = pd.to_numeric(df_comp[w], errors="coerce")

    if is_1_to_n:
        out = out.groupby("SEQN", as_index=False).max()
    return out, meta


# ── Seleção e Reescalonamento de Pesos Amostrais NCHS ────────────────
def compute_pooled_weights(merged: pd.DataFrame, present_cycles: list[str], own_wt_by_cycle: dict[str, list[str]]) -> tuple[pd.Series, dict, str]:
    """
    Aplica a hierarquia NCHS de subamostra nos K ciclos de 2 anos onde o componente existe:
      - Se 1999-2000 e 2001-2002 estão ambos presentes, usa WTSAU4YR * (2 / K).
      - Se apenas um deles está presente, usa WTSAU2YR * (1 / K).
      - Em 2003-2004, usa WTSAU2YR * (1 / K).
      - Em 2011-2012 e 2015-2016 (onde AUX é MEC completo), usa o peso de subamostra do
        componente se houver (ex. WTSH2YR em PBCD_I) * (1 / K); caso contrário WTMEC2YR * (1 / K).
      - Se em 1999-2004 o componente também tiver peso WTS* próprio distinto de WTSAU*, marca conflito.
    """
    k_cycles = len(present_cycles)
    if k_cycles == 0:
        return pd.Series(np.nan, index=merged.index), {}, "NO_CYCLES"

    both_99_02 = ("1999-2000" in present_cycles) and ("2001-2002" in present_cycles)
    pooled = pd.Series(0.0, index=merged.index, dtype="float64")
    spec_by_cycle: dict[str, str] = {}
    conflict_flags: list[str] = []

    for cyc in present_cycles:
        m = merged["cycle"] == cyc
        own_wts = [w for w in own_wt_by_cycle.get(cyc, []) if not w.startswith("WTSAU")]

        if cyc in {"1999-2000", "2001-2002", "2003-2004"}:
            if own_wts:
                conflict_flags.append(f"{cyc}:AUX_half_x_{own_wts[0]}")
            if cyc in {"1999-2000", "2001-2002"} and both_99_02 and merged.loc[m, "WTSAU4YR"].notna().any():
                col_used = "WTSAU4YR"
                factor = 2.0 / k_cycles
            else:
                col_used = "WTSAU2YR" if merged.loc[m, "WTSAU2YR"].notna().any() else "WTMEC2YR"
                factor = 1.0 / k_cycles
            w_vals = pd.to_numeric(merged.loc[m, col_used], errors="coerce").fillna(0.0) * factor
            pooled.loc[m] = w_vals
            spec_by_cycle[cyc] = f"{col_used} * ({int(round(factor * k_cycles))}/{k_cycles})"
        else:
            if own_wts and own_wts[0] in merged.columns:
                col_used = own_wts[0]
            else:
                col_used = "WTMEC2YR"
            factor = 1.0 / k_cycles
            w_vals = pd.to_numeric(merged.loc[m, col_used], errors="coerce").fillna(0.0) * factor
            pooled.loc[m] = w_vals
            spec_by_cycle[cyc] = f"{col_used} * (1/{k_cycles})"

    status = "WARNING_DUAL_SUBSAMPLE: " + ", ".join(conflict_flags) if conflict_flags else "OK_NCHS_COMPLIANT"
    return pooled, spec_by_cycle, status


# ── Auditoria Completa de Um Componente ──────────────────────────────
def audit_one_component(
    comp: ComponentSpec,
    base: pd.DataFrame,
    cycle_raw_tables: dict[str, dict],
    log: logging.Logger,
) -> dict:
    log.info(f"Auditando componente: {comp.key} ...")
    per_cycle_frames: list[pd.DataFrame] = []
    per_cycle_details: dict[str, dict] = {}
    own_wt_by_cycle: dict[str, list[str]] = {}
    present_cycles: list[str] = []

    for c_spec in ADULT_20_69_CYCLES:
        cyc = c_spec["cycle"]
        yr = c_spec["year"]

        if comp.source_kind == "aux_embedded":
            dl_info = {"status": "embedded_in_aux"}
            df_c = cycle_raw_tables[cyc]["aux"]
        elif comp.source_kind == "auq_embedded":
            dl_info = {"status": "embedded_in_auq"}
            df_c = cycle_raw_tables[cyc]["auq"]
        elif comp.source_kind == "mortality_dat":
            dl_info = resolve_or_download_mortality(c_spec["y1"], c_spec["y2"], log)
            df_c = parse_nchs_mortality_dat(Path(dl_info["path"])) if dl_info.get("path") else pd.DataFrame()
        else:
            cands = comp.candidates_by_cycle.get(cyc, [])
            if not cands:
                per_cycle_details[cyc] = {"status": "not_configured"}
                continue
            dl_info = resolve_or_download_xpt(yr, cands, log)
            df_c = read_xpt_clean(Path(dl_info["path"])) if dl_info.get("path") else pd.DataFrame()

        if df_c.empty:
            per_cycle_details[cyc] = {"download": dl_info, "n_valid_20_69": 0}
            continue

        person_df, struct_meta = extract_person_validity(comp, cyc, df_c)
        own_wt_by_cycle[cyc] = struct_meta.get("component_subsample_wt_cols", [])

        base_cyc = base[base["cycle"] == cyc].copy()
        joined = base_cyc.merge(person_df, on="SEQN", how="left", validate="one_to_one")
        joined["in_comp_table"] = joined["comp_valid"].notna()
        joined["comp_valid"] = joined["comp_valid"].fillna(False).astype(bool)

        age = pd.to_numeric(joined["RIDAGEYR"], errors="coerce")
        age_elig = (age >= comp.age_min) & (age <= comp.age_max)
        joined["age_eligible"] = age_elig
        joined["valid_and_eligible"] = joined["comp_valid"] & age_elig

        n_elig = int(age_elig.sum())
        n_val = int(joined["valid_and_eligible"].sum())
        if n_val > 0:
            present_cycles.append(cyc)
            per_cycle_frames.append(joined)

        cyc_detail = {
            "download": dl_info,
            "structure": struct_meta,
            "n_aux_20_69": int(len(base_cyc)),
            "n_aux_age_eligible": n_elig,
            "n_seqn_in_component_file": int((joined["in_comp_table"] & age_elig).sum()),
            "n_valid_intersection": n_val,
            "pct_missing_within_age_eligible": round(float((1.0 - n_val / n_elig) * 100.0), 2) if n_elig > 0 else None,
        }
        # Contagens marginais específicas de viabilidade (sem tocar em AUX)
        for extra_col in ["auq_general_valid", "auq_noise_valid", "auq_tinnitus_valid", "any_rx_30d", "loop_diuretic_30d", "aminoglycoside_30d", "platinum_30d", "cancer_yes_marginal", "is_deceased_marginal"]:
            if extra_col in joined.columns:
                cyc_detail[f"n_{extra_col}"] = int((joined[extra_col].fillna(False) & age_elig).sum())

        per_cycle_details[cyc] = cyc_detail

    total_aux_5c = int(len(base))
    if not per_cycle_frames:
        return {
            "component": comp.key,
            "description": comp.description,
            "lockbox_regime": comp.lockbox_regime,
            "age_band": f"{comp.age_min}-{comp.age_max}",
            "cycles_present": [],
            "n_cycles": 0,
            "n_valid_raw_20_69": 0,
            "n_valid_complete10_20_69": 0,
            "n_valid_any25_20_69": 0,
            "pct_missing_in_eligible_cycles_age": 100.0,
            "pct_missing_of_all_5cycles_20_69": 100.0,
            "weight_status": "NO_INTERSECTION",
            "weight_spec_by_cycle": {},
            "n_positive_pooled_weight": 0,
            "design_df": 0,
            "n_discovery_valid": 0,
            "n_replication_valid": 0,
            "per_cycle": per_cycle_details,
            "notes": comp.notes,
        }

    pooled_df = pd.concat(per_cycle_frames, ignore_index=True)
    w_pooled, w_spec, w_status = compute_pooled_weights(pooled_df, present_cycles, own_wt_by_cycle)
    pooled_df["w_pooled"] = w_pooled

    elig_mask = pooled_df["age_eligible"]
    val_mask = pooled_df["valid_and_eligible"]

    n_elig_present = int(elig_mask.sum())
    n_val_raw = int(val_mask.sum())
    n_val_comp10 = int((val_mask & pooled_df["is_complete10"]).sum())
    n_val_any25 = int((val_mask & pooled_df["is_any25"]).sum())
    n_pos_wt = int((val_mask & (pooled_df["w_pooled"] > 0)).sum())

    # Graus de liberdade de desenho (PSU - Estratos) nos ciclos presentes
    design_sub = pooled_df[pooled_df["w_pooled"] > 0]
    n_strata = int(design_sub.groupby(["cycle", "SDMVSTRA"]).ngroups) if "SDMVSTRA" in design_sub.columns else 0
    n_psu = int(design_sub.groupby(["cycle", "SDMVSTRA", "SDMVPSU"]).ngroups) if {"SDMVSTRA", "SDMVPSU"}.issubset(design_sub.columns) else 0
    design_df = max(n_psu - n_strata, 0)

    # Partição Descoberta vs Cofre de Replicação
    part_series = pooled_df["cycle"].map(
        lambda c: comp.partition_override.get(c, pooled_df.loc[pooled_df["cycle"] == c, "partition_default"].iloc[0])
    )
    if comp.lockbox_regime != "UNTOUCHED_LOCKBOX_VALID":
        n_disc = None
        n_repl = None
    else:
        n_disc = int((val_mask & (part_series == "discovery")).sum())
        n_repl = int((val_mask & (part_series == "replication")).sum())

    # Totais marginais extras (câncer, óbitos, classes de medicamentos, itens AUQ)
    marginal_totals: dict[str, int] = {}
    for extra_col in ["auq_general_valid", "auq_noise_valid", "auq_tinnitus_valid", "any_rx_30d", "loop_diuretic_30d", "aminoglycoside_30d", "platinum_30d", "cancer_yes_marginal", "is_deceased_marginal"]:
        if extra_col in pooled_df.columns:
            marginal_totals[f"total_{extra_col}"] = int((pooled_df[extra_col].fillna(False) & elig_mask).sum())

    return {
        "component": comp.key,
        "description": comp.description,
        "lockbox_regime": comp.lockbox_regime,
        "age_band": f"{comp.age_min}-{comp.age_max}",
        "cycles_present": present_cycles,
        "n_cycles": len(present_cycles),
        "n_aux_eligible_in_present_cycles": n_elig_present,
        "n_valid_raw_20_69": n_val_raw,
        "n_valid_complete10_20_69": n_val_comp10,
        "n_valid_any25_20_69": n_val_any25,
        "pct_missing_in_eligible_cycles_age": round(float((1.0 - n_val_raw / n_elig_present) * 100.0), 2) if n_elig_present > 0 else 100.0,
        "pct_missing_of_all_5cycles_20_69": round(float((1.0 - n_val_raw / total_aux_5c) * 100.0), 2) if total_aux_5c > 0 else 100.0,
        "weight_status": w_status,
        "weight_spec_by_cycle": w_spec,
        "n_positive_pooled_weight": n_pos_wt,
        "n_strata": n_strata,
        "n_psu": n_psu,
        "design_df": design_df,
        "n_discovery_valid": n_disc,
        "n_replication_valid": n_repl,
        "marginal_feasibility_counts": marginal_totals,
        "per_cycle": per_cycle_details,
        "notes": comp.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Matriz empírica de viabilidade NHANES por SEQN (20-69 anos).")
    parser.add_argument("--force", action="store_true", help="Reexecutar mesmo se outputs/json/37_viability_matrix.json existir.")
    parser.add_argument("--workers", type=int, default=3, help="Workers paralelos p/ componentes (default=3 p/ evitar rate-limit CDC).")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    if OUTPUT_JSON.exists() and not args.force:
        log.info(f"Output já existe em {OUTPUT_JSON}. Use --force para regerar. Pulando.")
        raise SystemExit(0)

    log.info("Construindo base canônica AUX + DEMO + AUQ (5 ciclos com adultos 20-69)...")
    base, cycle_raw_tables, base_audit = build_base_reference(log)
    log.info(
        f"Base 20-69 verificada: raw={base_audit['n_raw_20_69_5cycles']}, "
        f"complete10={base_audit['n_complete10_20_69_5cycles']}, "
        f"any25={base_audit['n_any25_20_69_5cycles']}"
    )

    n_workers = max(1, int(args.workers))
    # Separar componentes embutidos (já em memória) dos externos (rede/disco)
    embedded_specs = [c for c in COMPONENT_CATALOG if c.source_kind in {"aux_embedded", "auq_embedded"}]
    external_specs = [c for c in COMPONENT_CATALOG if c.source_kind not in {"aux_embedded", "auq_embedded"}]

    results_map: dict[str, dict] = {}
    for comp in embedded_specs:
        results_map[comp.key] = audit_one_component(comp, base, cycle_raw_tables, log)

    ext_results = Parallel(n_jobs=n_workers, prefer="threads", verbose=1)(
        delayed(audit_one_component)(comp, base, cycle_raw_tables, log)
        for comp in external_specs
    )
    for res in ext_results:
        results_map[res["component"]] = res

    ordered_results = [results_map[c.key] for c in COMPONENT_CATALOG]

    # Gerar CSV plano resumido para inspeção rápida
    summary_rows = []
    for r in ordered_results:
        summary_rows.append({
            "component": r["component"],
            "lockbox_regime": r["lockbox_regime"],
            "age_band": r["age_band"],
            "n_cycles": r["n_cycles"],
            "cycles_present": "|".join(r["cycles_present"]),
            "n_aux_eligible_in_cycles": r.get("n_aux_eligible_in_present_cycles", 0),
            "n_valid_raw_20_69": r["n_valid_raw_20_69"],
            "n_valid_complete10_20_69": r["n_valid_complete10_20_69"],
            "n_valid_any25_20_69": r["n_valid_any25_20_69"],
            "pct_missing_in_eligible": r["pct_missing_in_eligible_cycles_age"],
            "pct_missing_of_all_5cycles": r["pct_missing_of_all_5cycles_20_69"],
            "n_positive_pooled_weight": r["n_positive_pooled_weight"],
            "design_df": r["design_df"],
            "weight_status": r["weight_status"],
            "n_discovery": r["n_discovery_valid"],
            "n_replication": r["n_replication_valid"],
        })
    summary_df = pd.DataFrame(summary_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    payload = {
        "script": "37_viability_matrix.py",
        "random_state": RANDOM_STATE,
        "lockbox_guarantee": (
            "Nenhuma associação entre limiares audiométricos e variáveis externas foi calculada "
            "neste script; apenas contagens de cobertura por SEQN, ausência e pesos amostrais."
        ),
        "base_cohort_audit": base_audit,
        "components": ordered_results,
        "summary_csv": str(OUTPUT_CSV),
        "lib_versions": lib_versions(),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== RESUMO DA MATRIZ DE VIABILIDADE (20-69 ANOS) ===")
    print(summary_df.to_string(index=False))
    log.info(f"Concluído. JSON={OUTPUT_JSON} | CSV={OUTPUT_CSV}")


if __name__ == "__main__":
    main()
