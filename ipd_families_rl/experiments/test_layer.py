import json
import numpy as np

from vectorize import load_fingerprints, build_matrix
from clustering import ClusterConfig, hierarchical_cluster_labels


def main():
    records = load_fingerprints("fingerprints_100.json")
    assert len(records) > 0, "No fingerprints loaded."

    vr = build_matrix(records, key_mode="intersection")
    assert vr.X.shape[0] == len(vr.names)
    assert vr.X.shape[1] > 0, "No common features in intersection."

    cfg = ClusterConfig(method="average", metric="cosine", scaler="robust")
    labels = hierarchical_cluster_labels(vr.X, K=6, cfg=cfg)
    assert labels.shape[0] == vr.X.shape[0]
    assert len(set(labels.tolist())) <= 6 and len(set(labels.tolist())) >= 2

    # No NaNs
    assert np.isfinite(vr.X).all(), "Feature matrix contains NaNs or inf."

    print("Layer 3 sanity tests passed.")


if __name__ == "__main__":
    main()
