# Tool — pickle-free freeze of an unsupervised sklearn pipeline

> SKELETON ONLY. Tree + public API + JSON schema + signatures. Scorer NOT implemented.

## Package tree
```
skfreeze/
├── pyproject.toml
├── README.md
├── src/skfreeze/
│   ├── __init__.py        # public exports
│   ├── freeze.py          # sklearn pipeline -> artifact dict (fit-side, needs sklearn)
│   ├── score.py           # FrozenScorer — pure numpy, NO sklearn at serve time
│   ├── schema.py          # artifact schema + version + validation
│   └── io.py              # dump/load JSON
└── tests/
    └── test_roundtrip.py  # freeze -> dump -> load -> score == sklearn (parity)
```

## Public API (exported from __init__)
- `freeze_pipeline(pipeline, *, feature_cols, reference_X=None, cluster_labels=None, row_centering=False) -> dict`
- `dump(artifact: dict, path) -> None`
- `load(path) -> dict`
- `FrozenScorer(artifact: dict)` with methods `.transform(X)`, `.nearest(X)`, `.percentile(X)`, `.score(X)`

## JSON artifact contents (schema.py — names only)
- `schema_version`, `created`, `trained_n`, `sklearn_version`
- `feature_cols` (ordered), `row_centering` (bool)
- `scaler`: `{center[], scale[]}`
- `pca`: `{mean[], components[][], explained_variance_ratio[]}`
- `centroids`: `{cluster_id: vec[]}`
- `distance_distributions`: `{cluster_id: {mean, std}}`   # for distance→percentile via CDF
- `metadata`: free dict

## Function signatures (stubs — see score.py / freeze.py)
- freeze side: extract scaler/pca params + compute centroids & per-cluster distance dists from reference_X.
- serve side: center → scale → pca → nearest centroid → percentile = Φ(z), z=(d−μ)/σ.
- transfer: `.transform`/`.score` accept a NEW cohort (fit-on-reference, transform-on-new).
