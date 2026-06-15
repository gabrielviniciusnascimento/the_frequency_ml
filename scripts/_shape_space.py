#!/usr/bin/env python3
"""
Nome: _shape_space.py
Tarefa: Fonte ÚNICA de verdade do "espaço de forma" do NHANES usado por todo o
        pipeline de clustering / sensibilidade. Substitui as cópias inline de
        load/embed que viviam em 26_method_comparison, 27_binaural_pooling_ablation
        e audit_06_cluster_stability (risco de divergência silenciosa — ver
        docs/PIPELINE_METHODS_AUDIT.md, itens 1-3).

API:
  load_cohort(...)         -> (df, thr14)  coorte canônica (idade 20-69,
                              completude >=10/14, ANY25), índices resetados e
                              ALINHADOS entre df e thr14.
  shape_embed(thr_df, ...) -> Embed        row-center -> RobustScaler(25,75) ->
                              PCA(var, svd_solver="full"). Aceita 14D (orelhas
                              separadas) OU 7D (média binaural).
  lib_versions()           -> dict         carimbo de versões p/ proveniência
                              (deve ir em todo JSON de saída).

Garantias:
  - Determinístico: PCA com svd_solver="full" (independe de random_state).
  - SEM efeitos colaterais: não escreve disco, não configura logging.
  - Os filtros e o transform são byte-idênticos aos que estavam embutidos nos
    três scripts acima (verificado por re-execução + git diff).
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# ── Caminhos / colunas canônicas ─────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
FEATURE = ROOT / "data" / "processed" / "frequencia_feature_matrix_v1.csv"

FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R_COLS = [f"thr_R_{f}" for f in FREQS]
L_COLS = [f"thr_L_{f}" for f in FREQS]
FREQ_COLS_14 = R_COLS + L_COLS  # ordem R(7) depois L(7) — igual a 26/27/audit_06

# ── Parâmetros canônicos da coorte (pipeline principal — 18/26/27/audit_06) ──
AGE_MIN, AGE_MAX = 20, 69
MIN_COMPLETENESS = 10        # >= 10 de 14 limiares presentes
ANY25_THRESHOLD_DB = 25.0    # >25 dB em ao menos uma banda
PCA_VAR = 0.95               # variância retida
RANDOM_STATE = 42


def load_cohort(
    feature: Path = FEATURE,
    age_min: int = AGE_MIN,
    age_max: int = AGE_MAX,
    min_completeness: int = MIN_COMPLETENESS,
    any25_db: float = ANY25_THRESHOLD_DB,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coorte canônica. Retorna (df, thr14) com índices resetados e alinhados.

    Filtros aplicados na MESMA ordem usada pelos scripts originais:
      1. idade in [age_min, age_max]
      2. completude: >= min_completeness limiares não-nulos (de 14)
      3. ANY25: ao menos uma banda > any25_db
    """
    df = pd.read_csv(feature, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= age_min) & (age <= age_max)].copy()

    thr = df[FREQ_COLS_14].apply(pd.to_numeric, errors="coerce")
    keep = thr.notna().sum(axis=1) >= min_completeness
    df, thr = df[keep], thr[keep]

    any25 = (thr > any25_db).any(axis=1)
    df, thr = df[any25], thr[any25]

    return df.reset_index(drop=True), thr.reset_index(drop=True)


@dataclass
class Embed:
    """Resultado do embedding de forma."""
    X_scaled: np.ndarray   # row-centered + RobustScaler (pré-PCA)
    X_pca: np.ndarray      # espaço PCA (float64)
    scaler: RobustScaler
    pca: PCA

    @property
    def n_components(self) -> int:
        return int(self.X_pca.shape[1])

    @property
    def explained_variance(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


def shape_embed(
    thr_df: pd.DataFrame,
    pca_var: float = PCA_VAR,
    random_state: int = RANDOM_STATE,
) -> Embed:
    """row-center -> RobustScaler(25,75) -> PCA(pca_var, svd_solver="full").

    Determinístico. Aceita qualquer matriz tipo-limiar: 14D (orelhas separadas)
    ou 7D (média binaural). O row-centering isola a FORMA (remove o nível geral).
    """
    X = (
        thr_df.sub(thr_df.mean(axis=1, skipna=True), axis=0)
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    scaler = RobustScaler(quantile_range=(25, 75))
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=pca_var, svd_solver="full", random_state=random_state)
    X_pca = pca.fit_transform(X_scaled).astype(np.float64)
    return Embed(X_scaled=X_scaled, X_pca=X_pca, scaler=scaler, pca=pca)


def lib_versions() -> dict:
    """Versões das libs cientificas — proveniência para reprodutibilidade.
    Inclua o retorno em todo JSON de saída (campo 'lib_versions')."""
    import scipy

    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import hdbscan
        out["hdbscan"] = hdbscan.__version__
    except Exception:
        pass
    try:
        import diptest
        out["diptest"] = getattr(diptest, "__version__", "?")
    except Exception:
        pass
    return out


if __name__ == "__main__":
    # Sanity manual: confirma a coorte canônica e o nº de componentes.
    df_, thr_ = load_cohort()
    emb_ = shape_embed(thr_)
    print(f"coorte N={len(thr_)}  (esperado 7695)")
    print(f"PCA componentes={emb_.n_components}  var={emb_.explained_variance:.4f}")
    print(f"checksum X_pca.sum()={emb_.X_pca.sum():.6f}")
    print(f"lib_versions={lib_versions()}")
