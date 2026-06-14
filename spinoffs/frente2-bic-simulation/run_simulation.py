#!/usr/bin/env python3
"""
run_simulation.py
Simulates GMM BIC curves under different DGPs to calibrate decision rules for interior minima.
"""
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture

# Setup logging
LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "frente2_simulation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
DIMS = 14
N_SAMPLES = 2000
K_RANGE = list(range(1, 9))  # K = 1..8
REPLICATES = 15             # Replicates per DGP configuration
OUTPUT_PATH = Path("outputs/json/bic_simulation_results.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def generate_dgp_g0_flat(n_samples, dims, seed):
    """G0: Single anisotropic Gaussian (continuous, no clusters)."""
    rng = np.random.RandomState(seed)
    # Generate anisotropic covariance
    variances = rng.exponential(scale=5.0, size=dims) + 0.1
    # Create correlation structure
    rot = rng.normal(size=(dims, dims))
    q, _ = np.linalg.qr(rot)
    cov = q @ np.diag(variances) @ q.T
    mean = rng.normal(scale=10.0, size=dims)
    return rng.multivariate_normal(mean, cov, size=n_samples)


def generate_dgp_g0_curved(n_samples, dims, seed):
    """G0-Curved: Curved 1D manifold in dims-dimensional space + noise."""
    rng = np.random.RandomState(seed)
    t = rng.uniform(-2.0, 2.0, size=n_samples)
    X = np.zeros((n_samples, dims))
    for j in range(dims):
        if j % 2 == 0:
            X[:, j] = t ** 2  # Quadratic curve
        else:
            X[:, j] = t        # Linear manifold
    # Add isotropic noise
    noise = rng.normal(scale=0.8, size=(n_samples, dims))
    # Rotate manifold to distribute variance across all dimensions
    rot = rng.normal(size=(dims, dims))
    q, _ = np.linalg.qr(rot)
    return (X + noise) @ q.T


def generate_dgp_g1_separated(n_samples, dims, separation, seed):
    """G1: Mixture of 3 Gaussians with specified separation."""
    rng = np.random.RandomState(seed)
    n_per_cluster = n_samples // 3
    
    # Centers
    centers = [
        np.zeros(dims),
        rng.normal(scale=separation, size=dims),
        rng.normal(scale=separation, size=dims)
    ]
    
    # Generate isotropic clusters for simplicity but rotated
    X_parts = []
    for i, center in enumerate(centers):
        size = n_per_cluster if i < 2 else n_samples - 2 * n_per_cluster
        variances = rng.uniform(0.5, 1.5, size=dims)
        cov = np.diag(variances)
        X_parts.append(rng.multivariate_normal(center, cov, size=size))
        
    X = np.vstack(X_parts)
    # Shuffle
    shuffle_idx = rng.permutation(len(X))
    return X[shuffle_idx]


def run_sweep():
    log.info("Starting GMM BIC Simulation Sweep...")
    results = []
    
    # Config definition
    dgps = {
        "g0_flat": lambda s: generate_dgp_g0_flat(N_SAMPLES, DIMS, s),
        "g0_curved": lambda s: generate_dgp_g0_curved(N_SAMPLES, DIMS, s),
        "g1_overlap": lambda s: generate_dgp_g1_separated(N_SAMPLES, DIMS, separation=1.5, seed=s),
        "g1_separated": lambda s: generate_dgp_g1_separated(N_SAMPLES, DIMS, separation=4.0, seed=s)
    }
    
    total_runs = len(dgps) * REPLICATES * 2  # 2 covariance types (full, diag)
    run_idx = 0
    
    for dgp_name, dgp_fn in dgps.items():
        is_true_cluster = 1 if dgp_name.startswith("g1") else 0
        
        for rep in range(REPLICATES):
            rep_seed = RANDOM_STATE + rep
            X = dgp_fn(rep_seed)
            
            for cov_type in ["full", "diag"]:
                run_idx += 1
                if run_idx % 10 == 0 or run_idx == total_runs:
                    log.info(f"Progress: {run_idx}/{total_runs} runs completed...")
                
                # We want to measure the BIC curves for different n_init
                # To compare stability, we run for n_init = 1, 5, 20
                for n_init in [1, 5, 20]:
                    bic_curve = []
                    
                    for k in K_RANGE:
                        gmm = GaussianMixture(
                            n_components=k,
                            covariance_type=cov_type,
                            n_init=n_init,
                            random_state=rep_seed
                        )
                        gmm.fit(X)
                        bic_curve.append(float(gmm.bic(X)))
                        
                    # Calculate metrics on the BIC curve
                    bic_curve = np.array(bic_curve)
                    argmin_k = int(K_RANGE[np.argmin(bic_curve)])
                    
                    # Interior minimum is defined as minimum not at K=1 or K=max_k
                    has_interior_min = int(argmin_k > K_RANGE[0] and argmin_k < K_RANGE[-1])
                    
                    # Depth calculation: (BIC[K=1] - min(BIC)) / abs(BIC[K=1]) * 100
                    k1_val = bic_curve[0]
                    min_val = bic_curve[np.argmin(bic_curve)]
                    depth_k1 = float((k1_val - min_val) / abs(k1_val) * 100) if k1_val > min_val else 0.0
                    
                    # Alternative depth: (max(BIC) - min(BIC)) / abs(mean(BIC)) * 100
                    depth_range = float((np.max(bic_curve) - np.min(bic_curve)) / abs(np.mean(bic_curve)) * 100)
                    
                    results.append({
                        "dgp": dgp_name,
                        "is_true_cluster": is_true_cluster,
                        "replicate": rep,
                        "covariance_type": cov_type,
                        "n_init": n_init,
                        "argmin_k": argmin_k,
                        "has_interior_min": has_interior_min,
                        "depth_k1": depth_k1,
                        "depth_range": depth_range,
                        "bic_curve": bic_curve.tolist()
                    })
                    
    # Save output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    log.info(f"Simulation sweep finished. Saved raw results to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_sweep()
