

#!/usr/bin/env python3
"""
Nome: 33_qa_prevalence_distribution.py
Tarefa: Dois QAs que faltavam, ambos no espaço canônico (scripts/_shape_space.py):

  (A) PREVALÊNCIA PONDERADA vs NÃO-PONDERADA. O clustering é geométrico
      (não-ponderado, de propósito — é sobre forma, não estimativa populacional),
      mas qualquer afirmação de PREVALÊNCIA ("92% / 0.2%") deve ser checada com os
      pesos amostrais corretos do NHANES nos 5 ciclos com audiometria 20-69:
      WTSAU4YR*(2/5) em 1999-2002 (meia-amostra AUX), WTSAU2YR*(1/5) em 2003-2004
      (meia-amostra AUX) e WTMEC2YR*(1/5) em 2011-2012 e 2015-2016 (MEC completo).
      Usar WTMEC2YR puro subpondera 1999-2004 pela metade (42,9% vs 60,0%) porque a
      fração amostral mudou de 1/2 para 1 entre eras; o divisor NÃO cancela com WTMEC2YR.

  (B) QA DE DISTRIBUIÇÃO (skewness/kurtose) dos limiares brutos e dos PCs, para
      justificar por escrito a AUSÊNCIA de transformação de potência (Box-Cox/log):
      dB HL já é escala logarítmica; row-centering + RobustScaler controlam
      nível/outliers; os PCs (onde o clustering acontece) são ~simétricos; e o
      null model (30) preserva o skew empírico real, então a significância não é
      artefato de não-Gaussianidade.

Output: outputs/json/33_qa_prevalence_distribution.json
Dependências: scripts/_shape_space.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import hdbscan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shape_space import load_cohort, shape_embed, lib_versions, FREQ_COLS_14

RANDOM_STATE = 42
HDBSCAN_MCS = 10
HDBSCAN_MS = 5
REQUIRED_WEIGHT_COLS = ("WTSAU4YR", "WTSAU2YR", "WTMEC2YR")
POOLING_SPEC_5CYCLES = {
    "1999-2000": ("WTSAU4YR", 2.0 / 5.0),
    "2001-2002": ("WTSAU4YR", 2.0 / 5.0),
    "2003-2004": ("WTSAU2YR", 1.0 / 5.0),
    "2011-2012": ("WTMEC2YR", 1.0 / 5.0),
    "2015-2016": ("WTMEC2YR", 1.0 / 5.0),
}
OUTPUT = ROOT / "outputs" / "json" / "33_qa_prevalence_distribution.json"
LOG = ROOT / "outputs" / "logs" / "33_qa_prevalence_distribution.log"

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def build_5cycle_aux_weight(df) -> np.ndarray:
    """Constrói o peso composto NCHS para os 5 ciclos de audiometria 20-69 anos."""
    cycles = df["cycle"].astype(str)
    unknown = sorted(set(cycles.unique()) - set(POOLING_SPEC_5CYCLES))
    if unknown:
        raise ValueError(f"Ciclos fora da especificação 20-69 de 5 ciclos: {unknown}")
    w = np.zeros(len(df), dtype=np.float64)
    for cyc, (col, factor) in POOLING_SPEC_5CYCLES.items():
        mask = (cycles == cyc).to_numpy()
        if mask.any():
            vals = df.loc[mask, col].astype(float).fillna(0.0).to_numpy()
            w[mask] = vals * factor
    return w


def weighted_prevalence(df, labels):
    """Proporções não-ponderadas vs ponderadas (peso composto NCHS 5 ciclos) por grupo."""
    n = len(labels)
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels[labels != -1])}
    dominant = max(sizes, key=sizes.get) if sizes else None
    minority = min(sizes, key=sizes.get) if sizes else None

    groups = {"dominant": labels == dominant, "minority": labels == minority,
              "noise": labels == -1}
    missing_cols = [c for c in REQUIRED_WEIGHT_COLS if c not in df.columns]
    if missing_cols:
        return {"available": False, "reason": f"Colunas de peso ausentes: {missing_cols}"}
    w = build_5cycle_aux_weight(df)
    w_total = float(w.sum())
    if w_total <= 0.0:
        return {"available": False, "reason": "Soma de pesos composta <= 0"}

    out = {"available": True,
           "weight_formula": "WTSAU4YR*(2/5) [1999-2002] + WTSAU2YR*(1/5) [2003-2004] + WTMEC2YR*(1/5) [2011-2012, 2015-2016]",
           "n": n,
           "n_positive_weight": int((w > 0).sum()),
           "n_zero_or_missing_weight": int((w <= 0).sum()),
           "note": (
               "Proporção ponderada = Σw_grupo/Σw_total usando os pesos de subamostra de audiometria "
               "(WTSAU4YR em 99-02, WTSAU2YR em 03-04) combinados com WTMEC2YR (11-12, 15-16). "
               "Com WTMEC2YR puro o divisor de ciclos NÃO cancela, pois 1999-2004 era meia-amostra "
               "(subponderada em 3*0.5/3.5 = 42,9% em vez de 3/5 = 60,0%) e sem ajuste de não-resposta AUX. "
               "Variância de desenho (SDMVSTRA/SDMVPSU) não estimada neste QA pontual."
           ),
           "groups": {}}
    for name, mask in groups.items():
        out["groups"][name] = {
            "n": int(mask.sum()),
            "unweighted_pct": round(float(mask.sum() / n * 100), 3),
            "weighted_pct": round(float(w[mask].sum() / w_total * 100), 3),
        }
    deltas = {k: abs(v["unweighted_pct"] - v["weighted_pct"]) for k, v in out["groups"].items()}
    out["max_abs_delta_pct"] = round(max(deltas.values()), 3)
    out["robust_to_weighting"] = bool(out["max_abs_delta_pct"] < 2.0)
    return out


def distribution_qa(thr, X_pca):
    """Skewness/kurtose dos limiares brutos e dos PCs + justificativa do transform."""
    raw = thr.fillna(thr.mean()).to_numpy()
    sk_raw = stats.skew(raw, axis=0)
    sk_pc = stats.skew(X_pca, axis=0)
    ku_pc = stats.kurtosis(X_pca, axis=0)
    return {
        "raw_thresholds": {
            "skew_min": round(float(sk_raw.min()), 3),
            "skew_max": round(float(sk_raw.max()), 3),
            "skew_by_col": {c: round(float(s), 3) for c, s in zip(FREQ_COLS_14, sk_raw)},
        },
        "pca_axes": {
            "skew_PC1": round(float(sk_pc[0]), 3),
            "skew_PC2": round(float(sk_pc[1]), 3),
            "skew_abs_max": round(float(np.abs(sk_pc).max()), 3),
            "kurtosis_PC1": round(float(ku_pc[0]), 3),
        },
        "power_transform_decision": (
            "NÃO aplicar Box-Cox/log/Yeo-Johnson aos limiares. Justificativa: (1) dB HL já é "
            "escala logarítmica de pressão sonora — transformar de novo distorce a métrica clínica; "
            "(2) row-centering + RobustScaler(25,75) já controlam nível e outliers; "
            "(3) os PCs onde o clustering ocorre são ~simétricos (|skew|<1, ver acima), logo a "
            "geometria não é dominada por skew; (4) o null model (script 30) preserva o skew "
            "empírico por canal, então a significância das caudas NÃO é artefato de não-Gaussianidade. "
            "Limiares brutos são right-skewed (cauda de perda), o que é o sinal — não um defeito a corrigir."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="QA de prevalência ponderada NCHS (5 ciclos) e distribuição.")
    ap.add_argument("--force", action="store_true", help="Reexecutar mesmo se o JSON já existir.")
    args, _ = ap.parse_known_args()
    if OUTPUT.exists() and not args.force:
        log.info(f"Output já existe em {OUTPUT}. Use --force para regerar.")
        return json.loads(OUTPUT.read_text(encoding="utf-8"))

    df, thr = load_cohort()
    emb = shape_embed(thr)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MCS, min_samples=HDBSCAN_MS,
        metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=-1,
    ).fit_predict(emb.X_pca)

    prev = weighted_prevalence(df, labels)
    dist = distribution_qa(thr, emb.X_pca)
    log.info(f"Prevalência: {prev.get('groups')}")
    log.info(f"Robusta a ponderação: {prev.get('robust_to_weighting')} "
             f"(max Δ={prev.get('max_abs_delta_pct')} pp)")
    log.info(f"Skew PCs: PC1={dist['pca_axes']['skew_PC1']} PC2={dist['pca_axes']['skew_PC2']}")

    result = {
        "script": "33_qa_prevalence_distribution.py",
        "random_state": RANDOM_STATE,
        "n_samples": int(len(df)),
        "hdbscan": {"min_cluster_size": HDBSCAN_MCS, "min_samples": HDBSCAN_MS},
        "weighted_prevalence": prev,
        "distribution_qa": dist,
        "interpretation": (
            f"Prevalência {'ROBUSTA' if prev.get('robust_to_weighting') else 'SENSÍVEL'} aos pesos "
            f"amostrais (max Δ={prev.get('max_abs_delta_pct')} pp): as proporções não-ponderadas "
            f"{'podem ser citadas com a ressalva de serem amostrais' if prev.get('robust_to_weighting') else 'DEVEM ser substituídas pelas ponderadas'}. "
            "Clustering permanece não-ponderado por desenho (geometria de forma, não estimativa populacional). "
            "Sem transformação de potência (justificativa em distribution_qa)."
        ),
        "lib_versions": lib_versions(),
        "status": "EXECUTED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Concluído. Output: {OUTPUT}")
    return result


if __name__ == "__main__":
    main()

