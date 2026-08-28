#!/usr/bin/env python3
"""Summarise the AutoGluon negative control across the leave-one-regime-out folds.

The manuscript reported this control from a single fold and had to explain why one fold
was enough. With the full grid it becomes a measured statement, so what is needed is not a
ranking but an honest description of the failure: how often it collapsed, how far below
zero the coefficient of determination sat, and how often it did not fit at all.

    python runner/summarise_autogluon.py --in runs/final

Prints a per-fold table and the summary lines to quote in the paper. Computes nothing that
the run did not already record.
"""
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--model", default="autogluon")
    ap.add_argument("--dataset", default="ebw")
    a = ap.parse_args()

    units, skipped = [], []
    for f in sorted(Path(a.indir).glob(f"{a.dataset}__*__{a.model}__*.json")):
        try:
            u = json.loads(f.read_text())
        except Exception:
            continue
        if u.get("skipped"):
            skipped.append((u.get("fold"), u.get("seed"), str(u.get("skipped"))[:70]))
        else:
            units.append(u)

    if not units and not skipped:
        print(f"no {a.model} units found in {a.indir}")
        return 1

    targets = sorted({t for u in units for t in (u.get("metrics", {}).get("per_target") or {})})
    print(f"{'fold':>5} {'seed':>5} " + " ".join(f"{('R2 ' + t):>12}" for t in targets))
    print("-" * (12 + 13 * len(targets)))

    by_t = defaultdict(list)
    for u in sorted(units, key=lambda x: (x.get("fold", 0), x.get("seed", 0))):
        pt = u.get("metrics", {}).get("per_target") or {}
        row = f"{u.get('fold', '-'):>5} {u.get('seed', '-'):>5} "
        for t in targets:
            r2 = (pt.get(t) or {}).get("r2")
            row += f"{r2:>12.3f} " if isinstance(r2, (int, float)) else f"{'-':>12} "
            if isinstance(r2, (int, float)):
                by_t[t].append(r2)
        print(row)

    for fold, seed, why in skipped:
        print(f"{fold:>5} {seed:>5} {'FAILED TO FIT: ' + why:>26}")

    print()
    print("=" * 62)
    print(f"units that fitted      : {len(units)}")
    print(f"units that failed      : {len(skipped)}")
    n_folds = len({u.get('fold') for u in units} | {f for f, _, _ in skipped})
    print(f"distinct folds covered : {n_folds}")
    print()
    for t in targets:
        v = by_t[t]
        if not v:
            continue
        neg = sum(1 for x in v if x < 0)
        print(f"  {t}: R2 median {st.median(v):+.3f}  "
              f"min {min(v):+.3f}  max {max(v):+.3f}  "
              f"negative on {neg}/{len(v)} units")
    print()
    print("For the manuscript, quote the median and the range, not a single fold, and state")
    print("the number of units that failed to fit. A negative control is only persuasive if")
    print("its failure is measured and its own failures are disclosed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
