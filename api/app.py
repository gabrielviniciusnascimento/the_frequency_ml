#!/usr/bin/env python3
"""
The Frequency ML — Web API v1.1.0
Endpoint: POST /api/project
Receives 14 pure-tone thresholds, returns population position.
Validations: range check, NaN check, type enforcement.
"""

import json
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
API_DIR = Path(__file__).parent
ARTIFACT_PATH = API_DIR / "artifacts.json"
INDEX_PATH = API_DIR / "index.html"

# ── Constants ────────────────────────────────────────────────────────
THRESHOLD_MIN = -10.0
THRESHOLD_MAX = 130.0
MIN_VALID_FREQS = 4  # mínimo de frequências válidas para projetar

# ── Load artifacts once at startup ───────────────────────────────────
ARTIFACTS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Artifacts not found: {ARTIFACT_PATH}")
    ARTIFACTS.update(json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")))
    log.info(f"Artifacts loaded: {len(ARTIFACTS['freq_cols'])} freq cols, "
             f"{ARTIFACTS['pca_n_components']} PCA components, "
             f"{len(ARTIFACTS['centroids_pca'])} clusters")
    yield

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="The Frequency ML API",
    description="Project an audiogram into the NHANES-trained hearing loss space.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Request logging middleware ────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    log.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.3f}s)")
    return response

# ── Models with validation ───────────────────────────────────────────
class Audiogram(BaseModel):
    """14 pure-tone thresholds in dB HL. Valid range: -10 to 130."""

    thr_R_500: float = Field(..., description="Right ear 500 Hz (dB HL)", ge=-10, le=130)
    thr_R_1000: float = Field(..., description="Right ear 1000 Hz (dB HL)", ge=-10, le=130)
    thr_R_2000: float = Field(..., description="Right ear 2000 Hz (dB HL)", ge=-10, le=130)
    thr_R_3000: float = Field(..., description="Right ear 3000 Hz (dB HL)", ge=-10, le=130)
    thr_R_4000: float = Field(..., description="Right ear 4000 Hz (dB HL)", ge=-10, le=130)
    thr_R_6000: float = Field(..., description="Right ear 6000 Hz (dB HL)", ge=-10, le=130)
    thr_R_8000: float = Field(..., description="Right ear 8000 Hz (dB HL)", ge=-10, le=130)
    thr_L_500: float = Field(..., description="Left ear 500 Hz (dB HL)", ge=-10, le=130)
    thr_L_1000: float = Field(..., description="Left ear 1000 Hz (dB HL)", ge=-10, le=130)
    thr_L_2000: float = Field(..., description="Left ear 2000 Hz (dB HL)", ge=-10, le=130)
    thr_L_3000: float = Field(..., description="Left ear 3000 Hz (dB HL)", ge=-10, le=130)
    thr_L_4000: float = Field(..., description="Left ear 4000 Hz (dB HL)", ge=-10, le=130)
    thr_L_6000: float = Field(..., description="Left ear 6000 Hz (dB HL)", ge=-10, le=130)
    thr_L_8000: float = Field(..., description="Left ear 8000 Hz (dB HL)", ge=-10, le=130)

    class Config:
        json_schema_extra = {
            "example": {
                "thr_R_500": 15, "thr_R_1000": 15, "thr_R_2000": 20,
                "thr_R_3000": 40, "thr_R_4000": 60, "thr_R_6000": 75, "thr_R_8000": 80,
                "thr_L_500": 15, "thr_L_1000": 15, "thr_L_2000": 20,
                "thr_L_3000": 40, "thr_L_4000": 60, "thr_L_6000": 75, "thr_L_8000": 80,
            }
        }

# ── Helpers ──────────────────────────────────────────────────────────
FREQ_COLS = [
    "thr_R_500", "thr_R_1000", "thr_R_2000", "thr_R_3000",
    "thr_R_4000", "thr_R_6000", "thr_R_8000",
    "thr_L_500", "thr_L_1000", "thr_L_2000", "thr_L_3000",
    "thr_L_4000", "thr_L_6000", "thr_L_8000",
]

def project_audiogram(audiogram: Audiogram) -> dict:
    """Project an audiogram into the NHANES-trained PCA space."""
    vals = np.array([getattr(audiogram, col) for col in FREQ_COLS], dtype=np.float64)

    # Validate: at least MIN_VALID_FREQS non-NaN
    valid_count = int(np.sum(~np.isnan(vals)))
    if valid_count < MIN_VALID_FREQS:
        raise ValueError(f"Only {valid_count} valid frequencies (minimum {MIN_VALID_FREQS} required)")

    # Row-centering
    row_mean = float(np.nanmean(vals))
    centered = vals - row_mean
    centered = np.where(np.isnan(centered), 0.0, centered)

    # Scale
    center = np.array(ARTIFACTS["scaler_center"])
    scale = np.array(ARTIFACTS["scaler_scale"])
    scaled = (centered - center) / scale

    # PCA
    components = np.array(ARTIFACTS["pca_components"])
    pca_mean = np.array(ARTIFACTS["pca_mean"])
    pca_coords = (scaled - pca_mean) @ components.T

    # Distances
    centroids_pca = ARTIFACTS["centroids_pca"]
    distances = {}
    for cid_str, centroid in centroids_pca.items():
        cid = int(cid_str)
        c_arr = np.array(centroid)
        dist = float(np.sqrt(np.sum((pca_coords - c_arr) ** 2)))
        distances[cid] = round(dist, 4)

    nearest_cid = min(distances, key=distances.get)
    nearest_dist = distances[nearest_cid]

    # Percentile
    dist_dists = ARTIFACTS["distance_distributions"]
    percentile = None
    if str(nearest_cid) in dist_dists:
        dist_info = dist_dists[str(nearest_cid)]
        mean_d = dist_info["mean"]
        std_d = dist_info["std"]
        if std_d > 0:
            from scipy.stats import norm
            z_score = (nearest_dist - mean_d) / std_d
            percentile = round(float(norm.cdf(z_score) * 100), 1)

    # PTA
    r_vals = vals[:7]
    l_vals = vals[7:]
    pta_high_r = float(np.nanmean([r_vals[3], r_vals[4], r_vals[5], r_vals[6]]))
    pta_high_l = float(np.nanmean([l_vals[3], l_vals[4], l_vals[5], l_vals[6]]))
    pta_low_r = float(np.nanmean([r_vals[0], r_vals[1], r_vals[2]]))
    pta_low_l = float(np.nanmean([l_vals[0], l_vals[1], l_vals[2]]))
    asym = float(np.nanmean(np.abs(r_vals - l_vals)))

    cluster_descriptions = {
        0: {
            "name": "Mild-to-moderate sloping loss",
            "prevalence": f"{ARTIFACTS['cluster_sizes'].get('0', 0)}/{ARTIFACTS['total_size']} ({ARTIFACTS['cluster_sizes'].get('0', 0)/ARTIFACTS['total_size']*100:.1f}%)",
            "description": "The main body of hearing loss. Mild, age-associated, relatively symmetric.",
        },
        1: {
            "name": "Severe unilateral right-ear asymmetry",
            "prevalence": f"{ARTIFACTS['cluster_sizes'].get('1', 0)}/{ARTIFACTS['total_size']} ({ARTIFACTS['cluster_sizes'].get('1', 0)/ARTIFACTS['total_size']*100:.1f}%)",
            "description": "Severe loss in the right ear with near-normal left ear. Found across 4 NHANES cycles.",
        },
    }

    return {
        "nearest_cluster": nearest_cid,
        "nearest_distance": nearest_dist,
        "percentile_within_cluster": percentile,
        "all_distances": distances,
        "cluster_info": cluster_descriptions.get(nearest_cid, {}),
        "audiometric_summary": {
            "pta_high_R": round(pta_high_r, 1),
            "pta_high_L": round(pta_high_l, 1),
            "pta_low_R": round(pta_low_r, 1),
            "pta_low_L": round(pta_low_l, 1),
            "hf_lf_contrast_R": round(pta_high_r - pta_low_r, 1),
            "hf_lf_contrast_L": round(pta_high_l - pta_low_l, 1),
            "asymmetry_mean": round(asym, 1),
        },
        "pca_coordinates": [round(float(v), 6) for v in pca_coords],
        "row_centering_mean": round(row_mean, 2),
        "input_validation": {
            "valid_frequencies": valid_count,
            "threshold_range": f"{THRESHOLD_MIN} to {THRESHOLD_MAX} dB HL",
        },
        "context": {
            "n_training": ARTIFACTS["total_size"],
            "n_clusters": len(centroids_pca),
            "n_noise": ARTIFACTS["noise_size"],
            "noise_fraction": round(ARTIFACTS["noise_size"] / ARTIFACTS["total_size"], 4),
            "disclaimer": "This projection places your audiogram in a population-trained space. It is not a diagnosis.",
        },
    }

# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/")
def index():
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH, media_type="text/html")
    return {"name": "The Frequency ML API", "version": "1.1.0", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "artifacts_loaded": bool(ARTIFACTS),
        "n_clusters": len(ARTIFACTS.get("centroids_pca", {})),
        "n_training": ARTIFACTS.get("total_size", 0),
        "threshold_range": f"{THRESHOLD_MIN} to {THRESHOLD_MAX} dB HL",
        "min_valid_frequencies": MIN_VALID_FREQS,
    }

@app.get("/api/clusters")
def clusters():
    return {
        "clusters": ARTIFACTS.get("centroids_db", {}),
        "sizes": ARTIFACTS.get("cluster_sizes", {}),
        "noise_size": ARTIFACTS.get("noise_size", 0),
        "total_size": ARTIFACTS.get("total_size", 0),
    }

@app.post("/api/project")
def project(audiogram: Audiogram):
    """Project an audiogram into the NHANES-trained hearing loss space."""
    try:
        result = project_audiogram(audiogram)
        log.info(f"Projection: cluster={result['nearest_cluster']}, "
                 f"dist={result['nearest_distance']:.4f}, "
                 f"percentile={result['percentile_within_cluster']}")
        return result
    except ValueError as e:
        log.warning(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Projection failed")
        raise HTTPException(status_code=500, detail=str(e))

# ── Run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
