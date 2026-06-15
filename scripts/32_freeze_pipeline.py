#!/usr/bin/env python3
"""
Nome: 32_freeze_pipeline.py
Tarefa: Congelar o pipeline de forma canônico (RobustScaler + PCA) num artefato
        puro-numpy servível e reprodutível, usando o spinoff skfreeze (frente3).
        "Pluga" o skfreeze no pipeline principal: o transform canônico vira um
        artefato versionado que projeta novos audiogramas SEM sklearn em runtime
        (útil para o produto / serving / validação externa).

Desenho:
  - Espaço de forma via scripts/_shape_space.py (fonte única).
  - Reaproveita o scaler e o PCA JÁ ajustados (identidade garantida com o
    pipeline principal — não re-ajusta nada).
  - Particiona com a config HDBSCAN principal (mcs=10, ms=5) para gravar
    centróides + distribuições de distância por cluster.
  - PARITY-CHECK: FrozenScorer.transform(raw) deve bater com emb.X_pca
    (np.allclose) — falha dura se divergir.

Output: outputs/json/frozen_shape_pipeline.json  (artefato skfreeze + meta)
Dependências: scripts/_shape_space.py; spinoffs/frente3-pipeline-freeze/src/skfreeze
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
import hdbscan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "spinoffs" / "frente3-pipeline-freeze" / "src"))

from _shape_space import load_cohort, shape_embed, lib_versions, FREQ_COLS_14
from skfreeze.freeze import freeze_pipeline
from skfreeze.score import FrozenScorer

RANDOM_STATE = 42
HDBSCAN_MCS = 10
HDBSCAN_MS = 5
OUTPUT = ROOT / "outputs" / "json" / "frozen_shape_pipeline.json"
LOG = ROOT / "outputs" / "logs" / "32_freeze_pipeline.log"

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def main():
    df, thr = load_cohort()
    emb = shape_embed(thr)
    raw = thr.to_numpy(np.float64)  # 14D bruto (NaN preservado p/ row-centering no scorer)
    log.info(f"Coorte N={raw.shape[0]} × {raw.shape[1]} limiares; PCA {emb.n_components} comp.")

    # Partição principal (idêntica a 26/27) para centróides + distâncias.
    labels = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MCS, min_samples=HDBSCAN_MS,
        metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=-1,
    ).fit_predict(emb.X_pca)
    n_clusters = int(len(set(labels) - {-1}))
    log.info(f"HDBSCAN: {n_clusters} clusters, ruído={(labels==-1).sum()}")

    # Reaproveita scaler + PCA já ajustados (identidade com o pipeline principal).
    pipe = Pipeline([("scaler", emb.scaler), ("pca", emb.pca)])
    artifact = freeze_pipeline(
        pipe,
        feature_cols=FREQ_COLS_14,
        reference_X=raw,
        cluster_labels=labels,
        row_centering=True,
    )

    # ── PARITY-CHECK (falha dura) ─────────────────────────────────────
    scorer = FrozenScorer(artifact)
    proj = scorer.transform(raw)
    max_abs = float(np.max(np.abs(proj - emb.X_pca)))
    assert np.allclose(proj, emb.X_pca, rtol=1e-6, atol=1e-6), (
        f"PARITY FALHOU: FrozenScorer divergiu do pipeline (max|Δ|={max_abs:.2e})")
    log.info(f"PARITY OK: FrozenScorer == pipeline (max|Δ|={max_abs:.2e})")

    # Sanity 1D (um registro projeta igual à sua linha em 2D)
    one = scorer.transform(raw[0])
    assert np.allclose(one, emb.X_pca[0], atol=1e-6)

    result = {
        "script": "32_freeze_pipeline.py",
        "purpose": ("artefato puro-numpy do transform de forma canônico (RobustScaler+PCA), "
                    "servível sem sklearn; centróides/distâncias da partição HDBSCAN principal"),
        "random_state": RANDOM_STATE,
        "n_samples": int(raw.shape[0]),
        "hdbscan": {"min_cluster_size": HDBSCAN_MCS, "min_samples": HDBSCAN_MS,
                    "n_clusters": n_clusters},
        "parity_max_abs_diff": max_abs,
        "artifact": artifact,
        "lib_versions": lib_versions(),
        "status": "EXECUTED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Concluído. Output: {OUTPUT}")
    return result


if __name__ == "__main__":
    main()
