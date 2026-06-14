"""FrozenScorer — pure-numpy serving. NO sklearn import here."""
from __future__ import annotations
import numpy as np


class FrozenScorer:
    def __init__(self, artifact: dict) -> None:
        self.artifact = artifact
        self.row_centering = artifact.get("row_centering", False)
        
        self.scaler_center = (
            np.array(artifact["scaler_center"]) 
            if artifact["scaler_center"] is not None 
            else None
        )
        self.scaler_scale = (
            np.array(artifact["scaler_scale"]) 
            if artifact["scaler_scale"] is not None 
            else None
        )
        self.pca_mean = (
            np.array(artifact["pca_mean"]) 
            if artifact["pca_mean"] is not None 
            else None
        )
        self.pca_components = (
            np.array(artifact["pca_components"]) 
            if artifact["pca_components"] is not None 
            else None
        )
        
        self.centroids_pca = {
            int(k): np.array(v) 
            for k, v in artifact.get("centroids_pca", {}).items()
        }
        self.distance_distributions = artifact.get("distance_distributions", {})

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Raw vectors -> PCA coords (center -> scale -> pca). New cohort allowed."""
        is_1d = X.ndim == 1
        X_arr = X[None, :].copy() if is_1d else X.copy()
        
        # Row-centering
        if self.row_centering:
            row_means = np.nanmean(X_arr, axis=1, keepdims=True)
            X_arr = X_arr - row_means
            X_arr = np.where(np.isnan(X_arr), 0.0, X_arr)
        else:
            X_arr = np.nan_to_num(X_arr, nan=0.0)
            
        # Scale
        if self.scaler_center is not None and self.scaler_scale is not None:
            X_arr = (X_arr - self.scaler_center) / self.scaler_scale
            
        # PCA
        if self.pca_mean is not None and self.pca_components is not None:
            X_arr = (X_arr - self.pca_mean) @ self.pca_components.T
            
        if is_1d:
            return X_arr[0]
        return X_arr

    def nearest(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[int, float]:
        """Return (nearest_cluster_id, distance). Handles both 1D and 2D arrays."""
        X_pca = self.transform(X)
        is_1d = X.ndim == 1
        X_pca_2d = X_pca[None, :] if is_1d else X_pca
        
        n_samples = X_pca_2d.shape[0]
        nearest_cids = np.zeros(n_samples, dtype=int)
        nearest_dists = np.zeros(n_samples, dtype=float)
        
        if not self.centroids_pca:
            if is_1d:
                return -1, 0.0
            return nearest_cids, nearest_dists
            
        for i in range(n_samples):
            pt = X_pca_2d[i]
            dists = {}
            for cid, centroid in self.centroids_pca.items():
                dist = np.sqrt(np.sum((pt - centroid) ** 2))
                dists[cid] = dist
            best_cid = min(dists, key=dists.get)
            nearest_cids[i] = best_cid
            nearest_dists[i] = dists[best_cid]
            
        if is_1d:
            return int(nearest_cids[0]), float(nearest_dists[0])
        return nearest_cids, nearest_dists

    def percentile(self, X: np.ndarray) -> np.ndarray | float:
        """Distance -> within-cluster percentile via Normal CDF on stored (mean,std)."""
        nearest_cids, nearest_dists = self.nearest(X)
        is_1d = X.ndim == 1
        
        cids = [nearest_cids] if is_1d else nearest_cids
        dists = [nearest_dists] if is_1d else nearest_dists
        
        percentiles = []
        import math
        for cid, dist in zip(cids, dists):
            cid_str = str(cid)
            if cid_str in self.distance_distributions:
                dist_info = self.distance_distributions[cid_str]
                mean_d = dist_info["mean"]
                std_d = dist_info["std"]
                if std_d > 0:
                    z_score = (dist - mean_d) / std_d
                    # Use standard normal CDF (via math.erf) to calculate percentile
                    cdf_val = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
                    pct = float(cdf_val * 100)
                    percentiles.append(round(pct, 1))
                else:
                    percentiles.append(0.0)
            else:
                percentiles.append(0.0)
                
        if is_1d:
            return float(percentiles[0])
        return np.array(percentiles, dtype=float)

    def score(self, X: np.ndarray) -> dict:
        """Full record: coords, nearest, distance, percentile."""
        X_pca = self.transform(X)
        is_1d = X.ndim == 1
        
        if is_1d:
            nearest_cid, nearest_dist = self.nearest(X)
            pct = self.percentile(X)
            all_dists = {}
            for cid, centroid in self.centroids_pca.items():
                all_dists[str(cid)] = float(np.sqrt(np.sum((X_pca - centroid) ** 2)))
            return {
                "pca_coordinates": X_pca.tolist(),
                "nearest_cluster": int(nearest_cid),
                "distance": float(nearest_dist),
                "percentile": float(pct),
                "all_distances": all_dists
            }
        else:
            nearest_cids, nearest_dists = self.nearest(X)
            percentiles = self.percentile(X)
            
            all_dists_list = []
            for pt in X_pca:
                all_dists = {}
                for cid, centroid in self.centroids_pca.items():
                    all_dists[str(cid)] = float(np.sqrt(np.sum((pt - centroid) ** 2)))
                all_dists_list.append(all_dists)
                
            return {
                "pca_coordinates": X_pca.tolist(),
                "nearest_cluster": nearest_cids.tolist(),
                "distance": nearest_dists.tolist(),
                "percentile": percentiles.tolist(),
                "all_distances": all_dists_list
            }

