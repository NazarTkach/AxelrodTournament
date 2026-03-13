# ipd_families_rl/experiments/convert_display_lists_to_qualname.py
"""
Convert "display-name based" train/test lists into "true qualname" lists using your frozen catalog.

Why this exists:
- Your split JSONs currently use human/display names (e.g. "First by Nydegger", "$\\phi$").
- Your resolver wants __qualname__ (e.g. "TitForTat") or at least something unambiguous.
- Your frozen catalog (families/axelrod_catalog_*.json) contains BOTH:
    - spec["name"]  : display name (player.name)
    - spec["qualname"]: cls.__qualname__ (no module path)
  (based on your earlier StrategySpec build.)

This script:
1) loads the frozen catalog
2) builds a mapping: display_name -> qualname
3) rewrites train/test lists replacing qualname with true qualname
4) writes new files: *_resolved.json
5) reports missing / ambiguous matches

Usage:
  python experiments/convert_display_lists_to_qualname.py \
      --catalog families/axelrod_catalog_100.json \
      --in_train experiments/sets/train_60.json \
      --in_test  experiments/sets/test_20.json
"""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def norm(s: str) -> str:
    # Mild normalization: strip and collapse spaces. Keep case.
    return " ".join(str(s).strip().split())


def build_maps(catalog: List[Dict[str, Any]]):
    """
    Returns:
      name_to_qualnames: normalized display-name -> list of qualnames (to detect ambiguity)
      qualname_set: set of known qualnames
      name_set: set of known names
    """
    name_to_qualnames = defaultdict(list)
    qualname_set = set()
    name_set = set()

    for spec in catalog:
        n = spec.get("name")
        q = spec.get("qualname")
        if not n or not q:
            continue
        nn = norm(n)
        name_to_qualnames[nn].append(q)
        qualname_set.add(q)
        name_set.add(nn)

    return name_to_qualnames, qualname_set, name_set


def resolve_item(item: Dict[str, Any], name_to_qualnames, qualname_set) -> Tuple[Dict[str, Any], str]:
    """
    item can have:
      - item["qualname"] : currently display name OR already a qualname
      - item["name"]     : display name
    We try, in order:
      A) if item["qualname"] is already in qualname_set => accept
      B) map using item["name"]
      C) map using item["qualname"] as display name
    Returns (new_item, status) where status in {"ok","missing","ambiguous"}.
    """
    it = dict(item)

    q_or_name = norm(it.get("qualname", ""))
    display = norm(it.get("name", ""))

    # A) already a true qualname
    if q_or_name in qualname_set:
        it["qualname"] = q_or_name
        if not it.get("name"):
            it["name"] = q_or_name
        return it, "ok"

    # B) resolve by item["name"]
    if display and display in name_to_qualnames:
        qs = name_to_qualnames[display]
        if len(qs) == 1:
            it["qualname"] = qs[0]
            it["name"] = it.get("name", display)
            return it, "ok"
        else:
            it["_candidates"] = qs
            return it, "ambiguous"

    # C) resolve by item["qualname"] treated as display name
    if q_or_name and q_or_name in name_to_qualnames:
        qs = name_to_qualnames[q_or_name]
        if len(qs) == 1:
            it["qualname"] = qs[0]
            # keep original "name" if present; else use the display name
            it["name"] = it.get("name", q_or_name)
            return it, "ok"
        else:
            it["_candidates"] = qs
            return it, "ambiguous"

    return it, "missing"


def resolve_list(items: List[Dict[str, Any]], name_to_qualnames, qualname_set):
    ok, missing, ambiguous = [], [], []
    for it in items:
        new_it, status = resolve_item(it, name_to_qualnames, qualname_set)
        if status == "ok":
            ok.append(new_it)
        elif status == "missing":
            missing.append(new_it)
        else:
            ambiguous.append(new_it)
    return ok, missing, ambiguous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="axelrod_catalog_100.json")
    ap.add_argument("--in_train", default="train.json")
    ap.add_argument("--in_test", default="test.json")
    ap.add_argument("--out_train", default=None)
    ap.add_argument("--out_test", default=None)
    ap.add_argument("--report", default="resolve_report.json")
    args = ap.parse_args()

    catalog = load_json(args.catalog)
    train = load_json(args.in_train)
    test = load_json(args.in_test)

    name_to_qualnames, qualname_set, _ = build_maps(catalog)

    train_ok, train_missing, train_amb = resolve_list(train, name_to_qualnames, qualname_set)
    test_ok, test_missing, test_amb = resolve_list(test, name_to_qualnames, qualname_set)

    out_train = args.out_train or args.in_train.replace(".json", "_resolved.json")
    out_test = args.out_test or args.in_test.replace(".json", "_resolved.json")

    save_json(train_ok, out_train)
    save_json(test_ok, out_test)

    report = {
        "catalog_path": args.catalog,
        "in_train": args.in_train,
        "in_test": args.in_test,
        "out_train": out_train,
        "out_test": out_test,
        "counts": {
            "train_in": len(train),
            "train_ok": len(train_ok),
            "train_missing": len(train_missing),
            "train_ambiguous": len(train_amb),
            "test_in": len(test),
            "test_ok": len(test_ok),
            "test_missing": len(test_missing),
            "test_ambiguous": len(test_amb),
        },
        "train_missing": train_missing,
        "train_ambiguous": train_amb,
        "test_missing": test_missing,
        "test_ambiguous": test_amb,
    }
    save_json(report, args.report)

    print("Wrote:")
    print(" ", out_train, f"(ok={len(train_ok)} missing={len(train_missing)} amb={len(train_amb)})")
    print(" ", out_test, f"(ok={len(test_ok)} missing={len(test_missing)} amb={len(test_amb)})")
    print("Report:")
    print(" ", args.report)

    if train_missing or test_missing or train_amb or test_amb:
        print("\nWARNING: some items could not be resolved cleanly.")
        print("Open the report JSON to see missing/ambiguous entries.")


if __name__ == "__main__":
    main()
