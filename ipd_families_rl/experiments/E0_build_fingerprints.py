import json
import axelrod as axl
from tqdm import tqdm
from catalog import load_catalog
from base import resolve_strategy  # if you keep qualname resolver; or store factories directly
from fingreprints import FingerprintConfig, fingerprint_strategy, save_fingerprints


def main():
    # Load your frozen catalog created in Layer 1
    catalog = load_catalog("axelrod_catalog_100.json")

    fp_cfg = FingerprintConfig(
        p_end=0.1,
        max_rounds=50,
        tremble_eps=0.0,
        seeds_per_pair=2,
        base_seed=0,
    )

    records = []
    with tqdm(total=len(catalog)) as pbar:
        for item in catalog:
            # If you only store qualname, resolve to a class via axl strategies.
            # (Note: qualname-only resolution can be ambiguous; acceptable for now.)
            strategy_factory = resolve_strategy(item["qualname"])
            rec = fingerprint_strategy(strategy_factory, fp_cfg)
            records.append(rec)
            pbar.update()

    save_fingerprints(records, "fingerprints_100.json")
    print("saved:", len(records))


if __name__ == "__main__":
    main()
