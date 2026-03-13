# experiments/build_strategy_catalog.py
from __future__ import annotations

import argparse
from strategies.catalog import CatalogConfig, build_catalog, save_catalog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="families/axelrod_catalog_100.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stochastic", action="store_true")
    ap.add_argument("--no-deterministic", action="store_true")
    ap.add_argument("--max-memory-depth", type=int, default=None)
    args = ap.parse_args()

    cfg = CatalogConfig(
        target_size=args.n,
        include_stochastic=not args.no_stochastic,
        include_deterministic=not args.no_deterministic,
        max_memory_depth=args.max_memory_depth,
        seed=args.seed,
    )
    specs = build_catalog(cfg)
    print("Built catalog size:", len(specs))
    save_catalog(specs, args.out)
    print(f"Saved {len(specs)} strategies to {args.out}")


if __name__ == "__main__":
    main()
