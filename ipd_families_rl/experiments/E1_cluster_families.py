import json

from vectorize import load_fingerprints, build_matrix
from clustering import ClusterConfig, hierarchical_cluster_labels, export_family_labels
from stability import StabilityConfig, bootstrap_stability_curve
from family_stats import compute_family_stats, save_family_stats


def choose_K_by_max_stability(curve: dict) -> int:
    # Simple heuristic: max mean ARI (you can replace with plateau detection)
    bestK = None
    best = -1.0
    for K, v in curve.items():
        m = v["mean_ari"]
        if m == m and m > best:  # check not nan
            best = m
            bestK = int(K)
    return int(bestK) if bestK is not None else 6


def main():
    fp_path = "/home/nazar/PycharmProjects/AxelrodTournament/ipd_families_rl/experiments/fingerprints_100.json"
    out_labels = "labels.json"
    out_stats = "family_stats.json"
    out_curve = "stability_curve.json"

    records = load_fingerprints(fp_path)
    vr = build_matrix(records, key_mode="intersection")

    cluster_cfg = ClusterConfig(method="average", metric="cosine", scaler="robust")
    stab_cfg = StabilityConfig(B=50, seed=0)
    K_values = list(range(3, 13))

    curve = bootstrap_stability_curve(vr.X, K_values, cluster_cfg, stab_cfg)
    with open(out_curve, "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2, ensure_ascii=False)

    K = choose_K_by_max_stability(curve)
    labels = hierarchical_cluster_labels(vr.X, K=K, cfg=cluster_cfg)

    export_family_labels(vr.names, labels, out_labels)

    stats = compute_family_stats(vr.names, labels, vr.X, vr.keys, top_features=10)
    stats["chosen_K"] = K
    stats["stability_curve"] = curve
    save_family_stats(stats, out_stats)

    print("Done.")
    print("K =", K)
    print("labels ->", out_labels)
    print("stats  ->", out_stats)
    print("curve  ->", out_curve)


if __name__ == "__main__":
    main()
