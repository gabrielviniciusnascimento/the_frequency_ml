"""Tests for skfreeze: verify parity between sklearn and FrozenScorer."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import our package modules (relative to package root or via sys.path insertion)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from skfreeze.freeze import freeze_pipeline
from skfreeze.score import FrozenScorer


def test_skfreeze_roundtrip():
    # 1. Generate dummy data representing 100 individuals and 14 audiometric thresholds
    np.random.seed(42)
    n_samples = 120
    n_features = 14
    X = np.random.normal(loc=15.0, scale=10.0, size=(n_samples, n_features))
    
    # Introduce some missing values (NaNs) to verify robust row-centering
    X[0, 2] = np.nan
    X[5, 8] = np.nan
    
    feature_cols = [f"freq_{i}" for i in range(n_features)]
    
    # 2. Row centering preprocessing
    row_means = np.nanmean(X, axis=1, keepdims=True)
    X_centered = X - row_means
    X_centered = np.where(np.isnan(X_centered), 0.0, X_centered)
    
    # 3. Fit sklearn pipeline on row-centered reference data
    scaler = StandardScaler()
    pca = PCA(n_components=3, random_state=42)
    pipeline = Pipeline([
        ("scaler", scaler),
        ("pca", pca)
    ])
    
    X_pca = pipeline.fit_transform(X_centered)
    
    # Fit some clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    
    # 4. Freeze the pipeline
    artifact = freeze_pipeline(
        pipeline,
        feature_cols=feature_cols,
        reference_X=X,
        cluster_labels=labels,
        row_centering=True
    )
    
    # 5. Build FrozenScorer
    scorer = FrozenScorer(artifact)
    
    # 6. Test transform parity
    X_pca_frozen = scorer.transform(X)
    
    np.testing.assert_allclose(X_pca_frozen, X_pca, rtol=1e-5, atol=1e-5)
    print("OK Parity test passed: transform PCA coordinates match scikit-learn exactly!")
    
    # 7. Test nearest cluster parity
    nearest_cids, nearest_dists = scorer.nearest(X)
    
    # Verify centroids match
    for cid_str, centroid in artifact["centroids_pca"].items():
        cid = int(cid_str)
        expected_centroid = np.mean(X_pca[labels == cid], axis=0)
        np.testing.assert_allclose(np.array(centroid), expected_centroid, rtol=1e-5, atol=1e-5)
        
    print("OK Parity test passed: calculated centroids match actual cluster means!")
    
    # 8. Test 1D array handling
    sample_1d = X[12]
    score_1d = scorer.score(sample_1d)
    score_2d = scorer.score(X)
    
    # Verify 1D output matches row 12 of 2D output
    np.testing.assert_allclose(score_1d["pca_coordinates"], score_2d["pca_coordinates"][12], rtol=1e-5, atol=1e-5)
    assert score_1d["nearest_cluster"] == score_2d["nearest_cluster"][12]
    np.testing.assert_allclose(score_1d["distance"], score_2d["distance"][12], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(score_1d["percentile"], score_2d["percentile"][12], rtol=1e-5, atol=1e-5)
    
    print("OK Parity test passed: 1D array projection and scoring yields correct scalar equivalents!")
    
    # 9. Print sample scoring output
    print("\nSample 1D scoring result:")
    for k, v in score_1d.items():
        if isinstance(v, dict):
            print(f"  {k}: { {sub_k: round(sub_v, 4) for sub_k, sub_v in v.items()} }")
        elif isinstance(v, list):
            print(f"  {k}: {[round(x, 4) for x in v]}")
        else:
            print(f"  {k}: {v}")
            
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_skfreeze_roundtrip()
