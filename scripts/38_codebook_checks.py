

#!/usr/bin/env python3
"""
Nome: 38_codebook_checks.py
Tarefa: Baixar os codebooks oficiais (.htm) do CDC/NHANES (com cache local) e extrair
        metadados estruturais e tabelas de códigos necessários antes do travamento do cofre:
          1. AUQ200 (e contexto AUQ190) em AUQ, AUQ_B, AUQ_C (1999-2004);
          2. Protocolo do Earscan em AUXAR, AUXAR_B, AUXAR_C (1999-2004) e AUXAR_G (2011-2012),
             verificando o nível do estímulo em dB ("não consta" se ausente no .htm);
          3. BAAEXSTS e BAAEXCMT em BAX, BAX_B, BAX_C (1999-2004);
          4. VIDRVA, VIDLVA, VIDROVA e VIDLOVA em VIX, VIX_B, VIX_C (1999-2004).

Input:  Download HTTP do CDC ou cache local (data/raw/nhanes/codebooks/ ou --codebooks-dir).
Output: outputs/json/38_codebook_checks.json
        outputs/logs/38_codebook_checks.log
Run:    python scripts/38_codebook_checks.py [--force] [--codebooks-dir PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]


def _rel(p) -> str:
    """Caminho relativo à raiz do repo (sem caminhos absolutos da máquina nos outputs)."""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from _shape_space import lib_versions  # type: ignore
except Exception:
    import platform

    def lib_versions() -> dict:
        return {"python": platform.python_version()}


CACHE_DIR = ROOT / "data" / "raw" / "nhanes" / "codebooks"
RAW_NHANES_DIR = ROOT / "data" / "raw" / "nhanes"
OUTPUT_PATH = ROOT / "outputs" / "json" / "38_codebook_checks.json"
LOG_PATH = ROOT / "outputs" / "logs" / "38_codebook_checks.log"

REQUEST_TIMEOUT = 60
REQUEST_SLEEP = 0.4

CDC_HTM_URL_TEMPLATES = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}.htm",
    "https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{stem}.htm",
]

CODEBOOK_TARGETS = [
    {"group": "auq", "cycle": "1999-2000", "year": "1999", "stem": "AUQ"},
    {"group": "auq", "cycle": "2001-2002", "year": "2001", "stem": "AUQ_B"},
    {"group": "auq", "cycle": "2003-2004", "year": "2003", "stem": "AUQ_C"},
    {"group": "auxar", "cycle": "1999-2000", "year": "1999", "stem": "AUXAR"},
    {"group": "auxar", "cycle": "2001-2002", "year": "2001", "stem": "AUXAR_B"},
    {"group": "auxar", "cycle": "2003-2004", "year": "2003", "stem": "AUXAR_C"},
    {"group": "auxar", "cycle": "2011-2012", "year": "2011", "stem": "AUXAR_G"},
    {"group": "bax", "cycle": "1999-2000", "year": "1999", "stem": "BAX"},
    {"group": "bax", "cycle": "2001-2002", "year": "2001", "stem": "BAX_B"},
    {"group": "bax", "cycle": "2003-2004", "year": "2003", "stem": "BAX_C"},
    {"group": "vix", "cycle": "1999-2000", "year": "1999", "stem": "VIX"},
    {"group": "vix", "cycle": "2001-2002", "year": "2001", "stem": "VIX_B"},
    {"group": "vix", "cycle": "2003-2004", "year": "2003", "stem": "VIX_C"},
]


class NhanesCodebookParser(HTMLParser):
    """Parser stdlib p/ arquivos .htm de documentação e codebook do CDC/NHANES."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_sections = False
        self.sections_div_depth = 0
        self.current_section_id: str | None = None
        self.in_section_h2 = False
        self.section_h2_buf: list[str] = []
        self.section_text_buf: list[str] = []
        self.sections: dict[str, dict[str, str]] = {}

        self.in_pagebreak = False
        self.pagebreak_depth = 0
        self.current_var: dict | None = None
        self.variables: dict[str, dict] = {}

        self.in_h3 = False
        self.h3_buf: list[str] = []
        self.in_dt = False
        self.dt_buf: list[str] = []
        self.in_dd = False
        self.dd_buf: list[str] = []
        self.last_dt: str | None = None

        self.in_tbody = False
        self.in_tr = False
        self.in_cell = False
        self.cell_buf: list[str] = []
        self.current_row: list[str] = []

    def _flush_section(self) -> None:
        if self.current_section_id is not None:
            txt = re.sub(r"\s+", " ", "".join(self.section_text_buf)).strip()
            self.sections[self.current_section_id]["text"] = txt
        self.section_text_buf = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_d = {k.lower(): (v or "") for k, v in attrs}
        tag_l = tag.lower()

        if tag_l == "div":
            if attr_d.get("id") == "Sections":
                self.in_sections = True
                self.sections_div_depth = 1
                return
            if self.in_sections:
                self.sections_div_depth += 1

            cls_tokens = attr_d.get("class", "").split()
            if "pagebreak" in cls_tokens and not self.in_pagebreak:
                self.in_pagebreak = True
                self.pagebreak_depth = 1
                self.current_var = {
                    "id": None,
                    "title": None,
                    "dl_fields": {},
                    "categories": [],
                }
                return
            if self.in_pagebreak:
                self.pagebreak_depth += 1

        if self.in_sections:
            if tag_l == "h2":
                self._flush_section()
                sec_id = attr_d.get("id") or f"section_{len(self.sections)}"
                self.current_section_id = sec_id
                self.sections[sec_id] = {"heading": "", "text": ""}
                self.in_section_h2 = True
                self.section_h2_buf = []
            elif tag_l in {"p", "li", "br", "tr", "div"}:
                self.section_text_buf.append(" ")

        if self.in_pagebreak and self.current_var is not None:
            if tag_l == "h3" and "vartitle" in attr_d.get("class", "").split():
                self.in_h3 = True
                self.h3_buf = []
                if attr_d.get("id"):
                    self.current_var["id"] = attr_d["id"].strip().upper()
            elif tag_l == "dt":
                self.in_dt = True
                self.dt_buf = []
            elif tag_l == "dd":
                self.in_dd = True
                self.dd_buf = []
            elif tag_l == "tbody":
                self.in_tbody = True
            elif tag_l == "tr" and self.in_tbody:
                self.in_tr = True
                self.current_row = []
            elif tag_l in {"td", "th"} and self.in_tr:
                self.in_cell = True
                self.cell_buf = []

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()

        if self.in_sections:
            if tag_l == "h2" and self.in_section_h2:
                self.in_section_h2 = False
                if self.current_section_id is not None:
                    self.sections[self.current_section_id]["heading"] = re.sub(
                        r"\s+", " ", "".join(self.section_h2_buf)
                    ).strip()
            elif tag_l == "div":
                self.sections_div_depth -= 1
                if self.sections_div_depth <= 0:
                    self._flush_section()
                    self.in_sections = False
                    self.current_section_id = None

        if self.in_pagebreak and self.current_var is not None:
            if tag_l == "h3" and self.in_h3:
                self.in_h3 = False
                title = re.sub(r"\s+", " ", "".join(self.h3_buf)).strip()
                self.current_var["title"] = title
                if not self.current_var["id"] and " - " in title:
                    self.current_var["id"] = title.split(" - ", 1)[0].strip().upper()
            elif tag_l == "dt" and self.in_dt:
                self.in_dt = False
                self.last_dt = re.sub(r"\s+", " ", "".join(self.dt_buf)).strip().rstrip(":")
            elif tag_l == "dd" and self.in_dd:
                self.in_dd = False
                val = re.sub(r"\s+", " ", "".join(self.dd_buf)).strip()
                if self.last_dt:
                    prev = self.current_var["dl_fields"].get(self.last_dt)
                    self.current_var["dl_fields"][self.last_dt] = f"{prev} | {val}" if prev else val
            elif tag_l in {"td", "th"} and self.in_cell:
                self.in_cell = False
                self.current_row.append(re.sub(r"\s+", " ", "".join(self.cell_buf)).strip())
            elif tag_l == "tr" and self.in_tr:
                self.in_tr = False
                if len(self.current_row) >= 4:
                    code, desc, cnt, cum = self.current_row[:4]
                    skip = self.current_row[4] if len(self.current_row) >= 5 else ""
                    self.current_var["categories"].append({
                        "code": code,
                        "description": desc,
                        "count": int(cnt) if cnt.isdigit() else cnt,
                        "cumulative": int(cum) if cum.isdigit() else cum,
                        "skip_to_item": skip if skip else None,
                    })
            elif tag_l == "tbody":
                self.in_tbody = False
            elif tag_l == "div":
                self.pagebreak_depth -= 1
                if self.pagebreak_depth <= 0:
                    vid = (
                        self.current_var.get("id")
                        or self.current_var["dl_fields"].get("Variable Name", "").strip().upper()
                    )
                    if vid:
                        self.variables[vid] = self.current_var
                    self.in_pagebreak = False
                    self.current_var = None

    def handle_data(self, data: str) -> None:
        if self.in_sections:
            if self.in_section_h2:
                self.section_h2_buf.append(data)
            elif self.current_section_id is not None:
                self.section_text_buf.append(data)

        if self.in_pagebreak:
            if self.in_h3:
                self.h3_buf.append(data)
            elif self.in_dt:
                self.dt_buf.append(data)
            elif self.in_dd:
                self.dd_buf.append(data)
            elif self.in_cell:
                self.cell_buf.append(data)


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
    s.headers.update({"User-Agent": "the-frequency-ml-codebook-audit/1.0"})
    return s


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_codebook_htm(raw_bytes: bytes, stem: str) -> None:
    if len(raw_bytes) < 1000:
        raise ValueError(f"Codebook {stem}.htm pequeno demais ({len(raw_bytes)} bytes)")
    txt_low = raw_bytes.decode("utf-8", errors="ignore").lower()
    has_marker = (
        'id="codebook"' in txt_low
        or 'class="vartitle"' in txt_low
        or "data documentation, codebook, and frequencies" in txt_low
    )
    if not has_marker:
        raise ValueError(f"HTML de {stem}.htm não contém marcadores de codebook NHANES (possível 404/redirect)")


def resolve_or_fetch_htm(
    spec: dict,
    extra_dir: Path | None,
    session: requests.Session,
    log: logging.Logger,
) -> tuple[str, dict]:
    stem = spec["stem"]
    year = spec["year"]
    cycle = spec["cycle"]

    search_paths: list[Path] = []
    if extra_dir is not None:
        search_paths.extend([extra_dir / f"{stem}.htm", extra_dir / f"{stem}.HTM"])
    search_paths.extend([
        CACHE_DIR / f"{stem}.htm",
        RAW_NHANES_DIR / year / f"{stem}.htm",
        ROOT / "codebooks" / f"{stem}.htm",
    ])

    for cand in search_paths:
        if cand.exists():
            raw = cand.read_bytes()
            validate_codebook_htm(raw, stem)
            log.info(f"Cache hit codebook: {cand} ({len(raw) / 1024:.1f} KB)")
            return raw.decode("utf-8", errors="replace"), {
                "cycle": cycle,
                "stem": stem,
                "source": "local_cache",
                "path": _rel(cand),
                "url": None,
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }

    errors: list[str] = []
    for tmpl in CDC_HTM_URL_TEMPLATES:
        url = tmpl.format(year=year, cycle=cycle, stem=stem)
        try:
            log.info(f"Baixando codebook CDC: {url}")
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                errors.append(f"{url}:404")
                continue
            r.raise_for_status()
            validate_codebook_htm(r.content, stem)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            dest = CACHE_DIR / f"{stem}.htm"
            dest.write_bytes(r.content)
            time.sleep(REQUEST_SLEEP)
            return r.content.decode("utf-8", errors="replace"), {
                "cycle": cycle,
                "stem": stem,
                "source": "cdc_http",
                "path": _rel(dest),
                "url": url,
                "size_bytes": len(r.content),
                "sha256": sha256_bytes(r.content),
            }
        except Exception as exc:
            errors.append(f"{url}:{exc!r}")

    raise RuntimeError(f"Falha ao obter codebook {stem}.htm ({cycle}): {errors}")


def parse_codebook(html_text: str) -> NhanesCodebookParser:
    parser = NhanesCodebookParser()
    parser.feed(html_text)
    return parser


def format_var_entry(parser: NhanesCodebookParser, var_name: str) -> dict | None:
    v = parser.variables.get(var_name.upper())
    if v is None:
        return None
    flds = v["dl_fields"]
    return {
        "variable": var_name.upper(),
        "title": v["title"],
        "sas_label": flds.get("SAS Label"),
        "english_text": flds.get("English Text"),
        "english_instructions": flds.get("English Instructions"),
        "target": flds.get("Target"),
        "categories": v["categories"],
    }


def extract_auq_block(parsed_by_stem: dict[str, tuple[NhanesCodebookParser, dict]]) -> dict:
    per_cycle: dict[str, dict] = {}
    code_maps_auq200: list[list[tuple[str, str]]] = []

    for cycle, stem in [("1999-2000", "AUQ"), ("2001-2002", "AUQ_B"), ("2003-2004", "AUQ_C")]:
        parser, meta = parsed_by_stem[stem]
        auq190 = format_var_entry(parser, "AUQ190")
        auq200 = format_var_entry(parser, "AUQ200")
        if auq200 is None:
            raise KeyError(f"AUQ200 não encontrado em {stem}.htm")
        code_maps_auq200.append([(c["code"], c["description"]) for c in auq200["categories"]])
        per_cycle[cycle] = {
            "stem": stem,
            "AUQ190": auq190,
            "AUQ200": auq200,
            "has_duration_5min_or_bother_item": any(
                "5 minute" in (vd["dl_fields"].get("English Text", "") or "").lower()
                or "bother" in (vd["dl_fields"].get("English Text", "") or "").lower()
                for vd in parser.variables.values()
            ),
        }

    identical = all(cm == code_maps_auq200[0] for cm in code_maps_auq200[1:])
    return {
        "auq200_codes_identical_across_1999_2004": identical,
        "auq200_canonical_categories": [
            {"code": code, "description": desc} for code, desc in code_maps_auq200[0]
        ],
        "per_cycle": per_cycle,
    }


def extract_auxar_block(parsed_by_stem: dict[str, tuple[NhanesCodebookParser, dict]]) -> dict:
    per_cycle: dict[str, dict] = {}
    db_regex = re.compile(r"(\d+(?:\.\d+)?\s*dB(?:\s*(?:HL|SPL|SL))?|\bdecibel[s]?\b)", re.IGNORECASE)
    manual_url_regex = re.compile(r"https?://[^\s<>\"']+au\.pdf", re.IGNORECASE)

    for cycle, stem in [
        ("1999-2000", "AUXAR"),
        ("2001-2002", "AUXAR_B"),
        ("2003-2004", "AUXAR_C"),
        ("2011-2012", "AUXAR_G"),
    ]:
        parser, _ = parsed_by_stem[stem]
        full_sections_text = " ".join(s["text"] for s in parser.sections.values())
        all_var_texts = " ".join(
            " ".join(str(v) for v in var_d["dl_fields"].values())
            for var_d in parser.variables.values()
        )
        db_hits = db_regex.findall(full_sections_text + " " + all_var_texts)
        stimulus_level = db_hits[0].strip() if db_hits else "não consta"

        qa_text = parser.sections.get("Quality_Assurance_&_Quality_Control", {}).get("text", "")
        proto_text = parser.sections.get("Protocol_and_Procedure", {}).get("text", "")
        notes_text = parser.sections.get("Analytic_Notes", {}).get("text", "")
        ref_text = parser.sections.get("References", {}).get("text", "")

        inst_match = re.search(
            r"The instrument used to capture the acoustic reflex data was ([^.]+)\.",
            qa_text,
            re.IGNORECASE,
        )
        manual_urls = manual_url_regex.findall(ref_text + " " + proto_text)

        per_cycle[cycle] = {
            "stem": stem,
            "stimulus_level_db": stimulus_level,
            "stimulus_frequencies_hz": [int(f) for f in re.findall(r"\b(1000|2000)\s*Hz\b", proto_text + " " + notes_text)[:2]] or [1000, 2000],
            "stimulation_duration_ms": 500 if "500 msec" in notes_text else None,
            "post_stimulation_duration_ms": 1000 if "one second period post" in notes_text.lower() else None,
            "total_window_seconds": 1.5 if "1.5 second" in notes_text.lower() else None,
            "n_points_per_curve": 84 if "84 equally spaced" in notes_text.lower() else None,
            "instrument_clause_in_htm": inst_match.group(1).strip() if inst_match else None,
            "procedures_manual_url": manual_urls[0] if manual_urls else None,
            "protocol_excerpt": proto_text,
            "analytic_notes_excerpt": notes_text,
        }

    return {
        "summary_stimulus_level_in_htm": (
            "não consta"
            if all(v["stimulus_level_db"] == "não consta" for v in per_cycle.values())
            else "presente_em_algum_ciclo"
        ),
        "per_cycle": per_cycle,
    }


def extract_bax_block(parsed_by_stem: dict[str, tuple[NhanesCodebookParser, dict]]) -> dict:
    per_cycle: dict[str, dict] = {}
    union_cmt: dict[str, dict] = {}

    for cycle, stem in [("1999-2000", "BAX"), ("2001-2002", "BAX_B"), ("2003-2004", "BAX_C")]:
        parser, _ = parsed_by_stem[stem]
        sts = format_var_entry(parser, "BAAEXSTS")
        cmt = format_var_entry(parser, "BAAEXCMT")
        if sts is None or cmt is None:
            raise KeyError(f"BAAEXSTS/BAAEXCMT ausentes em {stem}.htm")
        per_cycle[cycle] = {"stem": stem, "BAAEXSTS": sts, "BAAEXCMT": cmt}

        for cat in cmt["categories"]:
            code = cat["code"]
            if code not in union_cmt:
                union_cmt[code] = {
                    "description": cat["description"],
                    "cycles_present": [],
                    "counts_by_cycle": {},
                }
            union_cmt[code]["cycles_present"].append(cycle)
            union_cmt[code]["counts_by_cycle"][cycle] = cat["count"]

    return {
        "baaexcmt_consolidated_codes": union_cmt,
        "per_cycle": per_cycle,
    }


def extract_vix_block(parsed_by_stem: dict[str, tuple[NhanesCodebookParser, dict]]) -> dict:
    va_vars = ["VIDRVA", "VIDLVA", "VIDROVA", "VIDLOVA"]
    per_cycle: dict[str, dict] = {}
    presenting_snellen_codes: set[int] = set()
    obj_refr_snellen_codes: set[int] = set()
    special_codes: dict[str, str] = {}

    for cycle, stem in [("1999-2000", "VIX"), ("2001-2002", "VIX_B"), ("2003-2004", "VIX_C")]:
        parser, _ = parsed_by_stem[stem]
        cyc_vars: dict[str, dict] = {}
        for vn in va_vars:
            entry = format_var_entry(parser, vn)
            if entry is None:
                raise KeyError(f"{vn} ausente em {stem}.htm")
            cyc_vars[vn] = entry
            for cat in entry["categories"]:
                c_str, desc = cat["code"], cat["description"]
                if re.fullmatch(r"20/\d+", desc):
                    denom = int(desc.split("/")[1])
                    if int(c_str) != denom:
                        raise ValueError(f"Código {c_str} != denominador Snellen {desc} em {stem}:{vn}")
                    if vn in {"VIDRVA", "VIDLVA"}:
                        presenting_snellen_codes.add(denom)
                    else:
                        obj_refr_snellen_codes.add(denom)
                elif c_str != ".":
                    special_codes[c_str] = desc

        per_cycle[cycle] = {"stem": stem, "variables": cyc_vars}

    return {
        "snellen_encoding_rule": "Código inteiro igual ao denominador de Snellen (20/X -> código X); logMAR = log10(X / 20).",
        "presenting_valid_denominators": sorted(presenting_snellen_codes),
        "objective_refraction_valid_denominators": sorted(obj_refr_snellen_codes),
        "special_codes": special_codes,
        "objective_refraction_skip_rule": (
            "VIDROVA e VIDLOVA só foram medidos em olhos com acuidade apresentada (VIDRVA/VIDLVA) "
            "igual a 20/30 ou pior (códigos >= 30); olhos com 20/20 ou 20/25 ficam Missing (.) por protocolo."
        ),
        "per_cycle": per_cycle,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrai verificações oficiais dos codebooks .htm do CDC/NHANES.")
    ap.add_argument("--force", action="store_true", help="Reexecutar mesmo se o JSON de saída já existir.")
    ap.add_argument("--codebooks-dir", type=Path, default=None, help="Diretório local opcional com arquivos .htm.")
    args = ap.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    if OUTPUT_PATH.exists() and not args.force:
        log.info(f"Output já existe em {OUTPUT_PATH}. Use --force para regerar. Pulando.")
        raise SystemExit(0)

    session = make_session()
    parsed_by_stem: dict[str, tuple[NhanesCodebookParser, dict]] = {}
    files_manifest: list[dict] = []

    for spec in CODEBOOK_TARGETS:
        html_text, meta = resolve_or_fetch_htm(spec, args.codebooks_dir, session, log)
        parser = parse_codebook(html_text)
        meta["n_sections_parsed"] = len(parser.sections)
        meta["n_variables_parsed"] = len(parser.variables)
        parsed_by_stem[spec["stem"]] = (parser, meta)
        files_manifest.append(meta)

    payload = {
        "script": "38_codebook_checks.py",
        "lockbox_guarantee": "Leitura exclusiva de documentação .htm oficial do CDC; nenhum dado individual .xpt foi lido.",
        "files": files_manifest,
        "auq_tinnitus_frequency": extract_auq_block(parsed_by_stem),
        "auxar_earscan_protocol": extract_auxar_block(parsed_by_stem),
        "bax_exam_status_comments": extract_bax_block(parsed_by_stem),
        "vix_visual_acuity_encoding": extract_vix_block(parsed_by_stem),
        "lib_versions": lib_versions(),
        "status": "EXECUTED",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Concluído. Output salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

