#!/usr/bin/env python3
"""
Nome: 07_pca_umap.py
Tarefa: Rodar RobustScaler + PCA 95% variância + UMAP 2D exploratório para duas políticas H11.
Input: data/processed/frequencia_model_ready_v1.parquet; data/processed/frequencia_model_ready_v1_666cap125.parquet; outputs/json/model_ready_feature_columns_v1.json.
Output: outputs/dashboards/pca_umap_exploratório.html; outputs/json/07_pca_umap.json; embeddings CSV.
Dependências: 06_model_ready.py.
"""

import logging
import json
from pathlib import Path

# Núcleo científico — sempre presente
import numpy as np
import pandas as pd
from scipy import stats, spatial, linalg
from scipy.spatial.distance import jensenshannon

# ML — scikit-learn para tudo modelável
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
try:
    import shap
except ImportError:
    shap = None

# Clustering
from sklearn.cluster import KMeans, DBSCAN
try:
    import hdbscan
except ImportError:
    hdbscan = None

# UMAP
try:
    import umap
except ImportError:
    umap = None

# Paralelismo
from joblib import Parallel, delayed
import multiprocessing as mp
N_JOBS = max(mp.cpu_count() - 1, 1)

# ── Constantes ───────────────────────────────────────────────────────
RANDOM_STATE = 42
PCA_VARIANCE_TARGET = 0.95
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10
UMAP_METRIC = "euclidean"
POLICIES = {
    "nan": Path("data/processed/frequencia_model_ready_v1.parquet"),
    "cap125": Path("data/processed/frequencia_model_ready_v1_666cap125.parquet"),
}
FEATURE_COLUMNS_JSON = Path("outputs/json/model_ready_feature_columns_v1.json")
OUTPUT_PATH = Path("outputs/json/07_pca_umap.json")
LOG_PATH = Path("outputs/logs/07_pca_umap.log")
DASHBOARD_HTML = Path("outputs/dashboards/pca_umap_exploratório.html")
EMBEDDING_DIR = Path("outputs/json")

# ── Logging padronizado ──────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Checkpointing ────────────────────────────────────────────────────
if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def qa_matrix(X: np.ndarray, nome: str) -> None:
    log.info(f"QA matrix {nome}: shape={X.shape}, dtype={X.dtype}")
    assert not np.isnan(X).any(), f"ERRO: NaN em {nome}"
    assert np.isfinite(X).all(), f"ERRO: inf em {nome}"


def load_feature_meta() -> dict:
    if not FEATURE_COLUMNS_JSON.exists():
        raise FileNotFoundError(FEATURE_COLUMNS_JSON)
    return json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))


def process_policy(policy: str, path: Path, shape_features: list[str], cycle_map: dict[str, int]) -> dict:
    if umap is None:
        raise ImportError("umap-learn não disponível")
    log.info(f"PCA+UMAP política={policy}; input={path}")
    df = pd.read_parquet(path)
    log.info(f"{policy}: model_ready shape={df.shape}")
    missing_features = [c for c in shape_features if c not in df.columns]
    if missing_features:
        raise ValueError(f"Features shape ausentes em {policy}: {missing_features[:10]}")

    X_df = df[shape_features].astype("float32")
    X = X_df.to_numpy(dtype=np.float32, copy=True)
    qa_matrix(X, f"X_shape_{policy}_pre_scaler")

    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    X_scaled = scaler.fit_transform(X).astype("float32")
    qa_matrix(X_scaled, f"X_shape_{policy}_scaled")

    pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver="full", random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled).astype("float32")
    n_components = int(pca.n_components_)
    explained = pca.explained_variance_ratio_.astype(float).tolist()
    log.info(f"{policy}: PCA n_components={n_components}; explained_sum={float(np.sum(pca.explained_variance_ratio_)):.6f}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        n_jobs=1,
        low_memory=True,
        verbose=False,
    )
    embedding = reducer.fit_transform(X_pca).astype("float32")
    qa_matrix(embedding, f"umap_embedding_{policy}")

    inverse_cycle = {v: k for k, v in cycle_map.items()}
    emb = pd.DataFrame({
        "SEQN": df["SEQN"].astype("int64"),
        "cycle_code": df["cycle_code"].astype("int16"),
        "cycle": df["cycle_code"].map(inverse_cycle).astype("string"),
        "RIDAGEYR": df["RIDAGEYR"].astype("float32") if "RIDAGEYR" in df.columns else np.nan,
        "RIAGENDR": df["RIAGENDR"].astype("float32") if "RIAGENDR" in df.columns else np.nan,
        "umap_x": embedding[:, 0],
        "umap_y": embedding[:, 1],
    })
    emb_path = EMBEDDING_DIR / f"umap_embeddings_{policy}.csv"
    emb.to_csv(emb_path, index=False)
    log.info(f"{policy}: embeddings salvos em {emb_path}; shape={emb.shape}")

    return {
        "policy": policy,
        "input": str(path),
        "n_samples": int(df.shape[0]),
        "n_shape_features": int(len(shape_features)),
        "pca_n_components_95pct": n_components,
        "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "pca_explained_variance_ratio": explained,
        "umap_params": {"n_neighbors": UMAP_N_NEIGHBORS, "min_dist": UMAP_MIN_DIST, "metric": UMAP_METRIC, "random_state": RANDOM_STATE},
        "embedding_csv": str(emb_path),
    }


def build_dashboard(results: list[dict]) -> None:
    data_payload = {}
    for r in results:
        emb = pd.read_csv(r["embedding_csv"])
        # Arredondar para HTML mais leve; dados continuam precisos em CSV.
        emb["umap_x"] = emb["umap_x"].round(5)
        emb["umap_y"] = emb["umap_y"].round(5)
        emb["RIDAGEYR"] = emb["RIDAGEYR"].round(2)
        data_payload[r["policy"]] = emb.to_dict(orient="records")

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<title>The Frequency ML — PCA/UMAP exploratório</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; background: #111; color: #eee; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 18px; }}
.card {{ background: #1b1b1b; border: 1px solid #333; border-radius: 12px; padding: 14px; }}
canvas {{ width: 100%; height: 360px; background: #050505; border-radius: 8px; }}
.small {{ color: #aaa; font-size: 13px; }}
code {{ color: #9ad; }}
</style>
</head>
<body>
<h1>The Frequency ML — PCA + UMAP exploratório</h1>
<p class="small">Visualização apenas. Não é prova de cluster nem evidência clínica. UMAP foi calculado após RobustScaler e PCA com 95% de variância explicada.</p>
<div id="meta" class="small"></div>
<div class="grid" id="grid"></div>
<script>
const DATA = {json.dumps(data_payload, ensure_ascii=False)};
const CYCLE_COLORS = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc949','#af7aa1','#ff9da7','#9c755f','#bab0ab'];
function sexColor(v) {{ if (Number(v)===1) return '#4e79a7'; if (Number(v)===2) return '#e15759'; return '#999'; }}
function ageColor(age) {{
  const a = Math.max(0, Math.min(90, Number(age)||0));
  const t = a/90;
  const r = Math.round(255*t), b = Math.round(255*(1-t)), g = Math.round(120*(1-Math.abs(t-0.5)*2));
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function cycleColor(code) {{ return CYCLE_COLORS[Math.abs(Number(code)||0) % CYCLE_COLORS.length]; }}
function draw(canvas, pts, mode) {{
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth * devicePixelRatio;
  const h = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.fillStyle = '#050505'; ctx.fillRect(0,0,w,h);
  const xs = pts.map(p=>p.umap_x), ys = pts.map(p=>p.umap_y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = 24 * devicePixelRatio;
  for (const p of pts) {{
    const x = pad + (p.umap_x-xmin)/(xmax-xmin || 1)*(w-2*pad);
    const y = h - pad - (p.umap_y-ymin)/(ymax-ymin || 1)*(h-2*pad);
    ctx.fillStyle = mode==='cycle' ? cycleColor(p.cycle_code) : (mode==='age' ? ageColor(p.RIDAGEYR) : sexColor(p.RIAGENDR));
    ctx.globalAlpha = 0.58;
    ctx.beginPath(); ctx.arc(x,y,1.35*devicePixelRatio,0,Math.PI*2); ctx.fill();
  }}
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#ddd'; ctx.font = `${{12*devicePixelRatio}}px system-ui`; ctx.fillText(`${{pts.length}} pontos`, 10*devicePixelRatio, 18*devicePixelRatio);
}}
const grid = document.getElementById('grid');
for (const policy of Object.keys(DATA)) {{
  for (const mode of ['cycle','age','sex']) {{
    const card = document.createElement('div'); card.className='card';
    card.innerHTML = `<h3>Política ${{policy}} — cor por ${{mode}}</h3><canvas></canvas><p class="small">Sem rótulo clínico. Coordenadas: UMAP sobre PCA 95%.</p>`;
    grid.appendChild(card);
    draw(card.querySelector('canvas'), DATA[policy], mode);
  }}
}}
document.getElementById('meta').textContent = `Políticas: ${{Object.keys(DATA).join(', ')}}. Gerado offline, sem dependências externas.`;
</script>
</body></html>"""
    DASHBOARD_HTML.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_HTML.write_text(html, encoding="utf-8")
    log.info(f"Dashboard salvo: {DASHBOARD_HTML}")


def main():
    log.info("Iniciando PCA+UMAP exploratório para duas políticas H11...")
    meta = load_feature_meta()
    shape_features = meta["shape_only_intersection"]
    if not shape_features:
        raise RuntimeError("Nenhuma feature shape-only disponível")
    # Ciclo map vem do 06_model_ready; ambas políticas devem ser idênticas.
    model_ready_json = json.loads(Path("outputs/json/06_model_ready.json").read_text(encoding="utf-8"))
    cycle_map = model_ready_json["policies"][0]["cycle_map"]

    results = []
    for policy, path in POLICIES.items():
        results.append(process_policy(policy, path, shape_features, cycle_map))
    build_dashboard(results)

    output = {
        "script": "07_pca_umap.py",
        "random_state": RANDOM_STATE,
        "feature_set": "shape_only_intersection",
        "shape_feature_count": len(shape_features),
        "dashboard_html": str(DASHBOARD_HTML),
        "policies": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
