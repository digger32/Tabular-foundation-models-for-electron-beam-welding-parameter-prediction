#!/usr/bin/env python3
"""Package the per-unit predictions for release.

The Data availability statement had to be narrowed because these were not published.
This turns the run directory into a single archive a reviewer can use to re-derive every
interval in the paper: for each unit, the held-out truth, the point prediction and the
full quantile grid.

    python runner/export_predictions.py --in runs/final --out release/predictions

Writes one CSV per unit plus a manifest, then a .tar.gz. No metrics are recomputed here:
whatever the run recorded is what is released.
"""
import argparse, csv, json, tarfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--archive", action="store_true", help="also write a .tar.gz")
    a = ap.parse_args()

    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest, n_q, skipped = [], 0, 0
    for f in sorted(indir.glob("*.json")):
        try:
            u = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(u, dict) or "y_test" not in u:
            continue
        y, pm, pq = u.get("y_test"), u.get("pred_mean"), u.get("pred_q")
        if y is None:
            skipped += 1
            continue
        targets = u.get("targets") or [f"t{i}" for i in range(len(y[0]))]
        qlevels = u.get("q_levels") or u.get("quantiles")

        header = ["row"]
        header += [f"y_{t}" for t in targets]
        if pm is not None:
            header += [f"pred_{t}" for t in targets]
        if pq is not None:
            n_q += 1
            if qlevels:
                header += [f"q{q}_{t}" for t in targets for q in qlevels]
            else:
                header += [f"q{j}_{t}" for t in targets for j in range(len(pq[0][0]))]

        rows = []
        for i in range(len(y)):
            r = [i] + list(y[i])
            if pm is not None:
                r += list(pm[i])
            if pq is not None:
                for j in range(len(targets)):
                    r += list(pq[i][j]) if isinstance(pq[i][0], (list, tuple)) else [pq[i][j]]
            rows.append(r)

        name = f.stem + ".csv"
        with (outdir / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

        manifest.append({"file": name, "dataset": u.get("dataset"), "regime": u.get("regime"),
                         "model": u.get("model"), "seed": u.get("seed"), "fold": u.get("fold"),
                         "n_rows": len(rows), "targets": targets,
                         "has_quantiles": pq is not None, "q_levels": qlevels})

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (outdir / "README.txt").write_text(
        "Per-unit predictions for the electron-beam welding acceptance study.\n\n"
        "One CSV per evaluation unit, named dataset__regime__model__seed__fold.\n"
        "Columns: the held-out measurement (y_*), the point prediction (pred_*) and,\n"
        "where the model is distributional, the full predictive quantile grid (q*_*).\n"
        "manifest.json lists every unit with its provenance.\n\n"
        "These are the raw outputs the reported intervals and decisions were derived\n"
        "from; nothing here is recomputed or rounded.\n")

    print(f"[export] {len(manifest)} units written to {outdir} "
          f"({n_q} carry a quantile grid, {skipped} had no y_test)")

    if a.archive:
        tgz = outdir.with_suffix(".tar.gz")
        with tarfile.open(tgz, "w:gz") as t:
            t.add(outdir, arcname=outdir.name)
        print(f"[export] archive -> {tgz} ({tgz.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
