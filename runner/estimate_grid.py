#!/usr/bin/env python3
"""
Extrapolate the full-grid wall-clock from the micro slice.

Reads runs/micro/manifest.jsonl (one fold, one seed, regimes full+augment) and
prices the whole grid from MEASURED per-unit cost. Nothing here is a prior.

PRICING RULES
  full                measured directly
  fewshot50/25        priced at the `full` rate. They are strictly cheaper (smaller
                      context, smaller tuning inner loop), so this is an upper bound
                      and the estimate is deliberately conservative.
  augment             measured directly. Classical-only: a TFM unit in this regime
                      is a skip and costs milliseconds, which the slice also
                      measures, so no special case is needed.
  folds               from each dataset's declared cv mode, read from the registry.

Anything the slice did not measure is reported as MISSING rather than guessed —
a silent default is how a 14 h estimate becomes a 58 h run.

    python3 runner/estimate_grid.py --micro runs/micro   # after the micro slice
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

SEEDS = 10
REGIMES = ["full", "fewshot50", "fewshot25", "augment"]
PRICED_AS = {"full": "full", "fewshot50": "full", "fewshot25": "full", "augment": "augment"}


def n_folds_of(dataset: str, data_dir: Path) -> int:
    import yaml
    import pandas as pd
    spec = yaml.safe_load((data_dir / "datasets.yaml").read_text())[dataset]
    if spec.get("cv", "holdout") != "logo":
        return 1
    df = pd.read_csv(data_dir / spec["path"], sep=spec.get("sep", ","))
    return int(df.groupby(spec["group_by"], sort=False).ngroup().nunique())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--micro", default="runs/micro")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()

    mf = Path(a.micro) / "manifest.jsonl"
    if not mf.exists():
        sys.exit(f"[estimate] {mf} not found — run the micro slice first (RUN_ORDER section 4-6)")

    cost = defaultdict(list)          # (dataset, model, regime) -> [wall_s]
    for line in mf.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "ok":
            cost[(r["dataset"], r["model"], r["regime"])].append(r["wall_s"])
    if not cost:
        sys.exit("[estimate] no ok units in the slice")

    datasets = sorted({k[0] for k in cost})
    models = sorted({k[1] for k in cost})
    folds = {d: n_folds_of(d, Path(a.data_dir)) for d in datasets}

    print("=" * 78)
    print(f"GRID ESTIMATE from {mf}   (folds: {folds}, seeds: {a.seeds})")
    print("=" * 78)

    def price(d, m, folds_d):
        """Hours for one (dataset, model) across all four regimes, from measured rates."""
        rates = {}
        for src in ("full", "augment"):
            v = cost.get((d, m, src))
            if v:
                rates[src] = sum(v) / len(v)
        if "full" not in rates:
            return None, rates
        h = 0.0
        for reg in REGIMES:
            src = PRICED_AS[reg]
            if src not in rates:
                missing.append(f"{d}/{m}/{reg} (needs {src})"); continue
            h += rates[src] * folds_d * a.seeds
        return h, rates

    total, missing = 0.0, []
    for d in datasets:
        sub = 0.0
        print(f"\n{d}  ({folds[d]} folds)")
        print(f"  {'model':12s} {'full s':>8s} {'augment s':>10s} {'units':>7s} {'hours':>8s}")
        for m in models:
            h, rates = price(d, m, folds[d])
            if h is None:
                missing.append(f"{d}/{m}/full"); continue
            sub += h
            n_units = folds[d] * a.seeds * len(REGIMES)
            print(f"  {m:12s} {rates['full']:8.1f} "
                  f"{rates.get('augment', float('nan')):10.1f} {n_units:7d} {h/3600:8.2f}")
        total += sub
        print(f"  {'':12s} {'':8s} {'':10s} {'subtotal':>7s} {sub/3600:8.2f} h")

    # gmaw_e2 is the same campaign, same shape, same fold count as gmaw_e1: price it
    # with the SAME per-regime rules rather than a shortcut. (An earlier version summed
    # the measured records once each, which priced 4 regimes as 2 and halved it.)
    if "gmaw_e2" not in datasets and "gmaw_e1" in datasets:
        g2 = sum(h for h in (price("gmaw_e1", m, folds["gmaw_e1"])[0] for m in models)
                 if h is not None)
        print(f"\n  gmaw_e2 not in the slice; same campaign and shape as gmaw_e1 "
              f"-> add {g2/3600:.1f} h")
        total += g2

    print("\n" + "=" * 78)
    print(f"ESTIMATED FULL GRID: {total/3600:.1f} h  (~{total/3600/24:.1f} days on one box)")
    print("Conservative: fewshot priced at the full rate. Sequential, single box.")
    if missing:
        print(f"\nMISSING measurements — NOT guessed, extend the slice:")
        for x in sorted(set(missing)):
            print(f"   {x}")
    print("=" * 78)


if __name__ == "__main__":
    main()
