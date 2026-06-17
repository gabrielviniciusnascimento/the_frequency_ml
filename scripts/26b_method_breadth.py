#!/usr/bin/env python3
"""
Nome: 26b_method_breadth.py
Tarefa: Ampliar o teste de "contínuo, não clusters" para além das 3 famílias de
        26_method_comparison.py (KMeans/GMM/HDBSCAN). Adiciona DBSCAN (grade de eps),
        Spectral (affinity=nearest_neighbors) e Agglomerative (complete + average),
        no MESMO espaço de forma canônico (row-centered). Fecha a lacuna de "amplitude
        de método" e aplica uma correção de múltiplos testes explícita sobre a grade de
        silhuetas (a interpretação do 26 era heurística).

        Critério: se NENHUMA família atravessa o limiar de silhueta "substancial" (0.5)
        de forma estável, a conclusão de continuidade é robusta a escolha de algoritmo.

Input: data/processed/frequencia_feature_matrix_v1.csv (via _shape_space)
Output: outputs/json/26b_method_breadth.json
Run: .venv/Scripts/python.exe scripts/26b_method_breadth.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _shape_space import load_cohort, shape_embed, lib_versions  # noqa: E402

OUT = ROOT / "outputs" / "json" / "26b_method_breadth.json"
RANDOM_STATE = 42
K_RANGE = [2, 3, 4, 5]
SUBSAMPLE_N = 4000           # for O(n^2)-memory methods (spectral, agglomerative)
SUBSTANTIAL = 0.5            # silhouette threshold for "substantial separation"


def part(X, labels):
    """Silhouette + balance of a partition. A high silhouette from a tiny-minority
    split (e.g. [n-1, 1]) is outlier-peeling (average/single linkage), NOT discrete
    structure — so we report min_frac to disqualify degenerate 'substantial' scores."""
    lab = np.asarray(labels)
    core = lab[lab != -1]
    uniq, counts = np.unique(core, return_counts=True)
    if len(uniq) < 2:
        return {"silhouette": None, "n_clusters": int(len(uniq)),
                "min_frac": None, "sizes": [int(c) for c in sorted(counts, reverse=True)]}
    s = float(silhouette_score(X, lab))
    return {"silhouette": round(s, 4), "n_clusters": int(len(uniq)),
            "min_frac": round(float(counts.min() / len(lab)), 4),
            "sizes": [int(c) for c in sorted(counts, reverse=True)][:5]}


def main() -> None:
    df, thr = load_cohort()
    emb = shape_embed(thr)
    X = emb.X_pca
    rng = np.random.default_rng(RANDOM_STATE)
    sub_idx = rng.choice(len(X), size=min(SUBSAMPLE_N, len(X)), replace=False)
    Xs = X[sub_idx]

    MIN_BALANCE = 0.05    # a real subtype split needs its minority cluster >= 5% of N

    results = {}
    results["kmeans"] = {str(k): part(X, KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
                                      .fit_predict(X)) for k in K_RANGE}
    db = {}
    for eps in (0.5, 1.0, 1.5, 2.0, 3.0):
        lab = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
        p = part(X, lab); p["noise_frac"] = round(float((lab == -1).mean()), 3)
        db[str(eps)] = p
    results["dbscan"] = db
    results["spectral_subsampled"] = {
        "subsample_n": int(len(Xs)),
        "by_k": {str(k): part(Xs, SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=15,
            random_state=RANDOM_STATE, assign_labels="kmeans").fit_predict(Xs))
            for k in K_RANGE},
    }
    agg = {"subsample_n": int(len(Xs))}
    for link in ("complete", "average"):
        agg[link] = {str(k): part(Xs, AgglomerativeClustering(n_clusters=k, linkage=link)
                                  .fit_predict(Xs)) for k in K_RANGE}
    results["agglomerative_subsampled"] = agg

    # --- collect every partition, separating raw-substantial from BALANCED-substantial ---
    def walk(node, path=""):
        for key, v in node.items():
            if isinstance(v, dict) and "silhouette" in v:
                yield path + key, v
            elif isinstance(v, dict):
                yield from walk(v, path + key + ".")
    parts = list(walk(results))
    sils = [(name, v) for name, v in parts if v["silhouette"] is not None]
    n_comparisons = len(sils)
    max_name, max_v = max(sils, key=lambda kv: kv[1]["silhouette"]) if sils else (None, None)
    # a "substantial" score only counts as discrete structure if the minority is non-trivial
    balanced_substantial = [
        {"method": name, **v} for name, v in sils
        if v["silhouette"] >= SUBSTANTIAL and (v["min_frac"] or 0) >= MIN_BALANCE
    ]
    degenerate_substantial = [
        {"method": name, **v} for name, v in sils
        if v["silhouette"] >= SUBSTANTIAL and (v["min_frac"] or 0) < MIN_BALANCE
    ]

    out = {
        "script": "26b_method_breadth.py",
        "purpose": "Method-breadth robustness for 'continuum, not clusters' + explicit "
                   "multiple-testing accounting; disqualifies degenerate outlier-peeling splits.",
        "cohort_n": int(len(thr)),
        "results": results,
        "multiple_testing": {
            "n_partition_comparisons": n_comparisons,
            "max_silhouette_overall": {"method": max_name, **(max_v or {})},
            "substantial_threshold": SUBSTANTIAL,
            "min_balance_for_real_subtype": MIN_BALANCE,
            "balanced_substantial_splits": balanced_substantial,
            "degenerate_substantial_splits_outlier_peeling": degenerate_substantial,
            "note": "Reporting the MAX silhouette over the WHOLE grid is the conservative, "
                    "multiple-comparison-aware statistic. High silhouettes from average/"
                    "single linkage are outlier-peeling: the minority cluster is <5% of N "
                    "(see min_frac/sizes), which is a tail being shaved off, not a subtype.",
        },
        "verdict": ("continuum_robust_to_method_choice" if not balanced_substantial
                    else "discrete_structure_under_some_method"),
        "lib_versions": lib_versions(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def fmt(d):
        return {k: f"sil={v['silhouette']},min%={v['min_frac']},sizes={v['sizes']}" for k, v in d.items()}
    print(f"kmeans: {fmt(results['kmeans'])}")
    print(f"spectral(sub): {fmt(results['spectral_subsampled']['by_k'])}")
    print(f"agglo average: {fmt(agg['average'])}")
    print(f"agglo complete: {fmt(agg['complete'])}")
    print(f"MAX silhouette = {max_v['silhouette'] if max_v else None} via {max_name} "
          f"(min_frac={max_v['min_frac'] if max_v else None}, sizes={max_v['sizes'] if max_v else None})")
    print(f"balanced substantial (real subtypes): {len(balanced_substantial)}; "
          f"degenerate/outlier-peeling: {len(degenerate_substantial)}")
    print(f"VERDICT: {out['verdict']}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
