# Tests

Two independent suites guard the two assets of this project.

## 1. App logic (JavaScript) — self-contained, no data needed

Pure logic of the empathy tool (`app/logic.js`), isolated from the DOM and Web Audio.

```sh
node --test tests/app_logic.test.js
```

Requires Node ≥ 22 (uses the built-in test runner). No external packages, no data
files — runs on any fresh clone.

## 2. Pipeline contract (Python) — reproducible env + one data file

`test_pipeline_contract.py` is a drift guard: it recomputes the canonical
shape-space pipeline from the raw cohort and asserts the frozen golden values
(N=7695, 10 PCs, var=0.9561, HDBSCAN 2 clusters sizes [13, 7097], skfreeze
parity max|d| < 1e-6, artifact sentinels). If any value moves, it fails loudly
and the change must be justified and re-baselined.

### Reproducible environment

The golden values were produced with the **pinned** versions in
[`../requirements-lock.txt`](../requirements-lock.txt) (numpy 2.2.6,
scikit-learn 1.7.2, hdbscan 0.8.44, scipy 1.15.3). This matters: the test
asserts **exact** integer cluster sizes and PCA variance, and both HDBSCAN
cluster counts and PCA variance can shift across library versions. The floors in
`requirements.txt` are *not* sufficient — install the lock file:

```sh
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt   # Windows
# .venv/bin/python  -m pip install -r requirements-lock.txt        # Unix
```

`requirements-lock.txt` is the exact scientific-core env that produces the
goldens; `requirements.txt` is the broader human-facing superset (it also lists
optional documentation tooling — plotly, nltk, PyMuPDF — not needed for this
test).

### Run

```sh
.venv/Scripts/python.exe tests/test_pipeline_contract.py   # Windows
# .venv/bin/python        tests/test_pipeline_contract.py   # Unix
```

> Note: the base `py` / system Python launcher will fail with
> `ModuleNotFoundError` unless the lock file is installed into it — use the venv.

### Data dependency (not in the repo)

The test reads `data/processed/frequencia_feature_matrix_v1.csv` (~14 MB),
which is **gitignored on purpose**: it is *derived* data, and committing a
derived artifact would reintroduce exactly the script↔artifact drift this test
exists to catch. It is regenerable from public NHANES audiometry via the ingest
chain `scripts/00_download_nhanes.py` → `01_ingest_aux.py` → `02_merge_context.py`
→ `03_features_v1.py` → `06_model_ready.py`. (The raw NHANES `.xpt` files under
`data/raw/` are likewise not committed; `00_download_nhanes.py` fetches them from
the CDC.)

Because of this, the Python contract test is currently a **local gate**: it runs
for anyone who has regenerated (or retained) the feature matrix, but cannot run
on a bare CI checkout without first materializing the data. Wiring it into CI is
a separate decision (download + regenerate NHANES in the job, vs. keeping it a
documented local gate alongside CI-run JS tests).
