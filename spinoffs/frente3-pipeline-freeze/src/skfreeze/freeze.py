"""Fit-side: sklearn pipeline -> artifact dict."""
from __future__ import annotations
import datetime
import numpy as np
from sklearn.pipeline import Pipeline


def freeze_pipeline(
    pipeline: Pipeline,
    *,
    feature_cols: list[str],
    reference_X: np.ndarray | None = None,
    cluster_labels: np.ndarray | None = None,
    row_centering: bool = False
) -> dict:
    """Extract scaler and PCA parameters from a fitted sklearn Pipeline.
    
    If reference_X and cluster_labels are provided, calculates the PCA coordinates,
    centroids for each cluster, cluster sizes, and distance distributions (mean and std)
    to allow distance-to-percentile lookup at inference time.
    """
    import sklearn
    
    artifact = {
        "schema_version": "1.0.0",
        "created": datetime.datetime.utcnow().isoformat() + "Z",
        "sklearn_version": sklearn.__version__,
        "feature_cols": list(feature_cols),
        "row_centering": row_centering,
        "scaler_center": None,
        "scaler_scale": None,
        "pca_mean": None,
        "pca_components": None,
        "pca_explained_variance_ratio": None,
        "centroids_pca": {},
        "distance_distributions": {},
        "cluster_sizes": {},
        "noise_size": 0,
        "total_size": 0
    }
    
    # Extract steps from Pipeline or sequential list of estimators
    estimators = []
    if isinstance(pipeline, Pipeline):
        estimators = [step for _, step in pipeline.steps]
    elif isinstance(pipeline, (list, tuple)):
        estimators = list(pipeline)
    else:
        estimators = [pipeline]
        
    # Find Scaler and PCA
    scaler = None
    pca = None
    for est in estimators:
        if hasattr(est, "mean_") and hasattr(est, "scale_"):
            scaler = est
        elif hasattr(est, "components_"):
            pca = est
            
    if scaler is not None:
        artifact["scaler_center"] = scaler.mean_.tolist()
        artifact["scaler_scale"] = scaler.scale_.tolist()
        
    if pca is not None:
        artifact["pca_components"] = pca.components_.tolist()
        if hasattr(pca, "mean_") and pca.mean_ is not None:
            artifact["pca_mean"] = pca.mean_.tolist()
        else:
            artifact["pca_mean"] = [0.0] * pca.components_.shape[1]
        if hasattr(pca, "explained_variance_ratio_") and pca.explained_variance_ratio_ is not None:
            artifact["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
            
    if reference_X is not None:
        # Number of samples
        artifact["total_size"] = int(reference_X.shape[0])
        
        # Apply row centering if requested
        X_proc = reference_X.copy()
        if row_centering:
            row_means = np.nanmean(X_proc, axis=1, keepdims=True)
            X_proc = X_proc - row_means
            X_proc = np.where(np.isnan(X_proc), 0.0, X_proc)
            
        # Transform through scaler and pca manually to verify math and compute coordinates
        if scaler is not None:
            mean = np.array(artifact["scaler_center"])
            scale = np.array(artifact["scaler_scale"])
            X_proc = (X_proc - mean) / scale
            
        if pca is not None:
            p_mean = np.array(artifact["pca_mean"])
            p_comp = np.array(artifact["pca_components"])
            X_pca = (X_proc - p_mean) @ p_comp.T
        else:
            X_pca = X_proc
            
        if cluster_labels is not None:
            unique_labels = np.unique(cluster_labels)
            
            # Identify noise
            noise_mask = (cluster_labels == -1)
            artifact["noise_size"] = int(np.sum(noise_mask))
            
            for label in unique_labels:
                if label == -1:
                    continue
                mask = (cluster_labels == label)
                c_size = int(np.sum(mask))
                label_str = str(label)
                
                artifact["cluster_sizes"][label_str] = c_size
                
                # Compute centroid
                coords_subset = X_pca[mask]
                centroid = np.mean(coords_subset, axis=0)
                artifact["centroids_pca"][label_str] = centroid.tolist()
                
                # Compute distances to centroid
                dists = np.sqrt(np.sum((coords_subset - centroid) ** 2, axis=1))
                mean_d = float(np.mean(dists))
                std_d = float(np.std(dists)) if len(dists) > 1 else 0.0
                
                # Store distance distribution parameters
                dist_info = {
                    "mean": round(mean_d, 6),
                    "std": round(std_d, 6)
                }
                
                # Add percentiles for reference
                for p in [10, 25, 50, 75, 90, 95, 99]:
                    dist_info[f"p{p}"] = round(float(np.percentile(dists, p)), 6) if len(dists) > 0 else 0.0
                    
                artifact["distance_distributions"][label_str] = dist_info
                
    return artifact

