#!/usr/bin/env python3
"""
Contract / drift guard for the canonical shape-space pipeline.

Plain-assert (no pytest dependency, like spinoffs/.../test_roundtrip.py); run with:
    python tests/test_pipeline_contract.py

Guards the invariants that a refactor or a dependency bump must NOT silently change
(this is exactly the failure we hit: a committed JSON drifted from its script). If any
golden value moves, this fails loudly and the change must be justified + re-baselined.
"""
import json
import sys
from pathlib import Path

import numpy as np
import hdbscan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "spinoffs" / "frente3-pipeline-freeze" / "src"))

from _shape_space import load_cohort, shape_embed, FREQ_COLS_14
from sklearn.pipeline import Pipeline
from skfreeze.freeze import freeze_pipeline
from skfreeze.score import FrozenScorer

# ── Golden values (config canônica mcs=10, ms=5) ─────────────────────
GOLDEN = {
    "n_samples": 7695,
    "n_components": 10,
    "explained_variance": 0.9561,
    "hdbscan_n_clusters": 2,
    "hdbscan_noise": 585,
    "minority_size": 13,
    "dominant_size": 7097,
    # sentinelas de artefato (script 26) — pegam drift script<->json
    "s26_best_silhouette": 0.2819,
    "s26_hdbscan_noise_fraction": 0.076,
}


def test_cohort_and_embedding():
    df, thr = load_cohort()
    assert len(thr) == GOLDEN["n_samples"], f"N={len(thr)} != {GOLDEN['n_samples']}"
    assert list(thr.columns) == FREQ_COLS_14, "colunas da coorte divergem do canônico"
    emb = shape_embed(thr)
    assert emb.n_components == GOLDEN["n_components"], f"n_comp={emb.n_components}"
    assert abs(emb.explained_variance - GOLDEN["explained_variance"]) < 5e-4, \
        f"var={emb.explained_variance:.4f}"
    assert np.isfinite(emb.scaler.center_).all() and len(emb.scaler.center_) == 14
    print(f"OK coorte+embedding: N={len(thr)}, {emb.n_components} PCs, var={emb.explained_variance:.4f}")
    return df, thr, emb


def test_partition(emb):
    labels = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                             cluster_selection_method="eom", core_dist_n_jobs=-1).fit_predict(emb.X_pca)
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels[labels != -1])}
    n_noise = int((labels == -1).sum())
    assert len(sizes) == GOLDEN["hdbscan_n_clusters"], f"n_clusters={len(sizes)}"
    assert n_noise == GOLDEN["hdbscan_noise"], f"noise={n_noise}"
    assert min(sizes.values()) == GOLDEN["minority_size"], f"minority={min(sizes.values())}"
    assert max(sizes.values()) == GOLDEN["dominant_size"], f"dominant={max(sizes.values())}"
    print(f"OK partição: {len(sizes)} clusters, ruído={n_noise}, sizes={sorted(sizes.values())}")
    return labels


def test_skfreeze_parity(thr, emb, labels):
    raw = thr.to_numpy(np.float64)
    pipe = Pipeline([("scaler", emb.scaler), ("pca", emb.pca)])
    artifact = freeze_pipeline(pipe, feature_cols=FREQ_COLS_14, reference_X=raw,
                               cluster_labels=labels, row_centering=True)
    proj = FrozenScorer(artifact).transform(raw)
    max_abs = float(np.max(np.abs(proj - emb.X_pca)))
    assert max_abs < 1e-6, f"parity skfreeze falhou: max|d|={max_abs:.2e}"
    assert artifact.get("scaler_type") == "RobustScaler"
    print(f"OK parity skfreeze: max|d|={max_abs:.2e}")


def test_artifact_sentinels():
    """Pega drift entre 26_method_comparison.py e seu JSON commitado."""
    p = ROOT / "outputs" / "json" / "26_method_comparison.json"
    if not p.exists():
        print("SKIP sentinelas do 26 (json ausente — rode o script 26)")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["kmeans"]["best_silhouette_value"] == GOLDEN["s26_best_silhouette"], \
        f"silhouette drift: {d['kmeans']['best_silhouette_value']}"
    assert d["hdbscan"]["noise_fraction"] == GOLDEN["s26_hdbscan_noise_fraction"], \
        f"noise drift: {d['hdbscan']['noise_fraction']}"
    assert d["n_samples"] == GOLDEN["n_samples"]
    print(f"OK sentinelas do 26: silhouette={GOLDEN['s26_best_silhouette']}, noise={GOLDEN['s26_hdbscan_noise_fraction']}")


def main():
    df, thr, emb = test_cohort_and_embedding()
    labels = test_partition(emb)
    test_skfreeze_parity(thr, emb, labels)
    test_artifact_sentinels()
    print("\nTODOS OS CONTRATOS OK — pipeline canônico estável, sem drift.")


if __name__ == "__main__":
    main()
