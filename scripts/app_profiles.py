"""app_profiles.py — proveniência dos perfis audíveis da ferramenta de empatia.

Os três perfis de exemplo do tool (app/logic.js, app/profiles.json) NÃO são números
escolhidos à mão: são as MEDIANAS por frequência de cada grupo geométrico já commitado
em outputs/json/20_outlier_analysis.json (campo threshold_medians_db), derivadas da
coorte ANALISADA (N=7.695, age 20–69, ≥10/14 limiares, ANY25). Este script é uma
transformação determinística e SEM reclustering: lê o artefato commitado e emite
app/profiles.json no formato que o tool consome, com carimbo de proveniência.

Mapeamento de rótulo -> slot do tool:
  cluster_0  -> "0" "Perda gradual nos agudos"     (forma mais comum, n grande)
  cluster_1  -> "1" "Perda severa em uma orelha"   (cauda rara, n=12 — exemplar, NÃO fenótipo)
  outliers   -> "2" "Queda acentuada nos agudos"   (perda de agudos acentuada)

Saída: app/profiles.json   (rodar: .venv/Scripts/python.exe scripts/app_profiles.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _shape_space import lib_versions  # noqa: E402

SRC = ROOT / "outputs" / "json" / "20_outlier_analysis.json"
OUT = ROOT / "app" / "profiles.json"

FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]

# rótulo no artefato -> (slot do tool, nome exibido, nota de proveniência por perfil)
MAPPING = {
    "cluster_0": ("0", "Perda gradual nos agudos",
                  "Grupo mais comum (n grande): perda leve/gradual, bilateral e quase simétrica."),
    "cluster_1": ("1", "Perda severa em uma orelha",
                  "Cauda rara (n=12): EXEMPLAR de assimetria extrema, NÃO um fenótipo validado "
                  "(rebaixado a cauda de continuum — ver dossiê §2 e audit_06)."),
    "outliers":  ("2", "Queda acentuada nos agudos",
                  "Perda acentuada nos agudos, aproximadamente simétrica entre as orelhas."),
}


def curve(medians: dict, ear: str) -> list[float]:
    """Extrai a curva de 7 bandas (na ordem de FREQS) para a orelha R/L."""
    return [float(medians[f"thr_{ear}_{f}"]) for f in FREQS]


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    by_label = {g["label"]: g for g in data["geometric_profiles"]}

    profiles: dict[str, dict] = {}
    for label, (slot, name, note) in MAPPING.items():
        g = by_label[label]
        med = g["threshold_medians_db"]
        profiles[slot] = {
            "name": name,
            "source_cluster": label,
            "n": g["n"],
            "R": curve(med, "R"),
            "L": curve(med, "L"),
            "note": note,
        }

    out = {
        "_comment": "GERADO por scripts/app_profiles.py — não editar à mão. "
                    "Fonte de verdade dos perfis audíveis; app/logic.js espelha estes "
                    "valores e tests/app_logic.test.js trava a paridade.",
        "generated_by": "scripts/app_profiles.py",
        "source": "outputs/json/20_outlier_analysis.json",
        "statistic": "threshold_medians_db (mediana por frequência de cada grupo)",
        "freqs_hz": FREQS,
        "provenance": {
            "cohort": "NHANES analisada: idade 20–69, >=10/14 limiares, ANY25 (>25 dB em >=1 banda)",
            "n_analyzed": 7695,
            "caveat": "cluster_1 (n=12) é exemplar de cauda, não fenótipo clínico validado.",
        },
        "profiles": profiles,
        "lib_versions": lib_versions(),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK -> {OUT.relative_to(ROOT)}")
    for slot, p in profiles.items():
        print(f"  [{slot}] {p['name']:<28} R={p['R']}  L={p['L']}")


if __name__ == "__main__":
    main()
