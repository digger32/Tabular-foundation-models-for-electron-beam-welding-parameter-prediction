#!/usr/bin/env python3
"""
A (EBW) statistics — reads the runner's per-unit JSONs and produces the analysis
the experiment plan promises. Writes the artifacts the review gate's E1 asserts.

Per unit JSON carries: dataset, regime, model, seed, targets, and
metrics{ per_target{<target>{rmse,mae,r2}}, coverage_80pi, ece, optimism_gap }.

Outputs (into <out>/stats/):
  omnibus.json      Friedman across models on macro-RMSE (regime=full), per dataset
  posthoc.json      Nemenyi + mean ranks + critical difference
  pairwise.json     Wilcoxon-Holm: each TFM vs the best classical control (paired)
  summary.json      per-model per-target RMSE/MAE/R2 with bootstrap 95% CI
  calibration.json  coverage_80pi / interval-ECE per distributional model
  fewshot.json      macro-RMSE vs regime (full/fewshot50/fewshot25) per model

Usage:
    python stats.py --in runs/final --out runs/final
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TFM = {"tabpfn_v2", "tabpfn_v25", "tabpfn_v3", "mitra"}
CLASSICAL = {"catboost", "xgb", "ngb", "mlp"}


def load_units(indir: Path):
    out = []
    for p in indir.glob("*__*__*__seed*.json"):
        try:
            u = json.loads(p.read_text())
            if not u.get("skipped"):
                out.append(u)
        except Exception:
            pass
    return out


def macro_rmse(u):
    pt = (u.get("metrics") or {}).get("per_target") or {}
    vals = [v["rmse"] for v in pt.values() if v.get("rmse") is not None]
    return float(np.mean(vals)) if vals else None


def conformal_cqr(y, q_lo, q_hi, seed=0, alpha=0.20):
    """Split conformalized quantile regression (Romano et al. 2019) evaluated within a
    unit: the test points are split into a calibration half and an evaluation half; the
    conformity offset is fit on calibration and applied on evaluation, giving a coverage
    that is valid by construction. Returns (raw_cov_eval, conformal_cov_eval, n_eval)."""
    import numpy as np
    y=np.ravel(np.asarray(y)); q_lo=np.ravel(np.asarray(q_lo)); q_hi=np.ravel(np.asarray(q_hi))
    n=len(y)
    if n<8: return None
    rng=np.random.default_rng(seed); idx=rng.permutation(n); c=idx[:n//2]; e=idx[n//2:]
    s=np.maximum(q_lo[c]-y[c], y[c]-q_hi[c])                       # conformity scores
    k=int(np.ceil((len(c)+1)*(1-alpha)))/len(c); k=min(max(k,0.0),1.0)
    Q=float(np.quantile(s, k, method="higher"))
    raw =float(np.mean((y[e]>=q_lo[e]) & (y[e]<=q_hi[e])))
    conf=float(np.mean((y[e]>=q_lo[e]-Q) & (y[e]<=q_hi[e]+Q)))
    return raw, conf, len(e)

def bootstrap_ci(vals, n=2000, seed=0):
    vals = np.asarray([v for v in vals if v is not None], float)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def wilcoxon_holm(pairs):
    """pairs: dict name -> (a_vals, b_vals) paired. Returns Wilcoxon greater-than
    (a worse? here we test a<b on RMSE => a better) with Holm correction."""
    from scipy.stats import wilcoxon
    res = {}
    raw = {}
    for name, (a, b) in pairs.items():
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 6:
            res[name] = {"n": int(m.sum()), "p_value": None, "note": "n<6"}
            continue
        try:
            stat, p = wilcoxon(a[m], b[m])  # two-sided on paired RMSE
            raw[name] = p
            res[name] = {"n": int(m.sum()), "stat": float(stat), "p_raw": float(p),
                         "median_delta": float(np.median(a[m] - b[m]))}
        except Exception as e:
            res[name] = {"error": str(e)}
    # Holm across the tested pairs
    tested = sorted(raw.items(), key=lambda kv: kv[1])
    k = len(tested)
    for rank, (name, p) in enumerate(tested):
        res[name]["p_holm"] = float(min(1.0, p * (k - rank)))
    return res


def main():
    from scipy.stats import friedmanchisquare
    try:
        import scikit_posthocs as sp
    except ImportError:
        sp = None

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    a = ap.parse_args()
    indir, outdir = Path(a.indir), Path(a.outdir)
    (outdir / "stats").mkdir(parents=True, exist_ok=True)

    units = load_units(indir)
    models = sorted({u["model"] for u in units})
    datasets = sorted({u["dataset"] for u in units})
    targets = sorted({t for u in units for t in u.get("targets", [])})

    # ---- summary: per-model per-target RMSE/MAE/R2 + CI (regime=full) ----------
    summary = {}
    by_model_target = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for u in units:
        if u.get("regime") != "full":
            continue
        for t, v in ((u.get("metrics") or {}).get("per_target") or {}).items():
            for k in ("rmse", "mae", "r2", "crps"):
                if v.get(k) is not None:
                    by_model_target[u["model"]][t][k].append(v[k])
    for m in models:
        summary[m] = {}
        for t in targets:
            summary[m][t] = {}
            for k in ("rmse", "mae", "r2", "crps"):
                vals = by_model_target[m][t][k]
                summary[m][t][k] = {"mean": float(np.mean(vals)) if vals else None,
                                    "ci95": bootstrap_ci(vals)}
    (outdir / "stats" / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- macro-RMSE table keyed by (dataset, seed, model), regime=full ---------
    cell = defaultdict(dict)   # (dataset, seed) -> {model: macro_rmse}
    for u in units:
        if u.get("regime") != "full":
            continue
        r = macro_rmse(u)
        if r is not None:
            cell[(u["dataset"], u["seed"])][u["model"]] = r

    # ---- Friedman + Nemenyi across models over (dataset,seed) blocks -----------
    blocks = [c for c in cell.values() if all(m in c for m in models)]
    omnibus = {"test": "Friedman across models on macro-RMSE (regime=full)",
               "n_blocks": len(blocks), "models": models}
    posthoc = {"test": "Nemenyi post-hoc + mean ranks", "models": models}
    if len(blocks) >= 6 and len(models) >= 3:
        mat = np.array([[b[m] for m in models] for b in blocks])   # (blocks, models)
        fr, p = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
        omnibus["friedman_stat"] = float(fr); omnibus["p_value"] = float(p)
        ranks = np.argsort(np.argsort(mat, axis=1), axis=1) + 1     # 1=best (lowest RMSE)
        posthoc["mean_rank"] = {m: float(ranks[:, j].mean()) for j, m in enumerate(models)}
        k, N = len(models), len(blocks)
        q_alpha = 3.314  # Nemenyi q for alpha=0.05, k up to ~10 (interp.); refine per k
        posthoc["critical_difference"] = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * N)))
        if sp is not None:
            posthoc["nemenyi"] = sp.posthoc_nemenyi_friedman(mat).values.tolist()
    (outdir / "stats" / "omnibus.json").write_text(json.dumps(omnibus, indent=2))
    (outdir / "stats" / "posthoc.json").write_text(json.dumps(posthoc, indent=2))

    # ---- pairwise Wilcoxon-Holm: each TFM vs the best classical control --------
    means = {m: np.mean([b[m] for b in blocks]) for m in models} if blocks else {}
    best_classical = min((m for m in models if m in CLASSICAL),
                         key=lambda m: means.get(m, np.inf), default=None)
    pairwise = {"reference_best_classical": best_classical}
    if best_classical and blocks:
        ref = np.array([b[best_classical] for b in blocks])
        pairs = {}
        for m in models:
            if m in TFM:
                pairs[f"{m}_vs_{best_classical}"] = (np.array([b[m] for b in blocks]), ref)
        pairwise["tests"] = wilcoxon_holm(pairs)
        pairwise["delta_ci95_tfm_minus_classical"] = {
            m: bootstrap_ci(list(np.array([b[m] for b in blocks]) - ref))
            for m in models if m in TFM}
    (outdir / "stats" / "pairwise.json").write_text(json.dumps(pairwise, indent=2))

    # ---- calibration: coverage_80pi / ECE per distributional model ------------
    calib = defaultdict(lambda: {"coverage": [], "ece": [], "crps": [], "rel": defaultdict(list), "conf": [], "raw_eval": []})
    for u in units:
        if u.get("regime") != "full":
            continue
        mm = u.get("metrics") or {}
        if mm.get("coverage_80pi") is not None:
            calib[u["model"]]["coverage"].append(mm["coverage_80pi"])
            if mm.get("ece") is not None:
                calib[u["model"]]["ece"].append(mm["ece"])
            cr = [v.get("crps") for v in (mm.get("per_target") or {}).values() if v.get("crps") is not None]
            if cr:
                calib[u["model"]]["crps"].append(float(np.mean(cr)))
            for lv, c in (mm.get("coverage_by_level") or {}).items():
                calib[u["model"]]["rel"][lv].append(c)
            if mm.get("coverage_80pi_conformal") is not None:
                calib[u["model"]]["conf"].append(mm["coverage_80pi_conformal"])
                calib[u["model"]]["raw_eval"].append(mm["coverage_80pi_raw_eval"])
    calibration = {m: {"coverage_80pi_mean": float(np.mean(v["coverage"])),
                       "coverage_ci95": bootstrap_ci(v["coverage"]),
                       "interval_ece_mean": float(np.mean(v["ece"])) if v["ece"] else None,
                       "crps_mean": float(np.mean(v["crps"])) if v["crps"] else None,
                       "reliability": {lv: float(np.mean(cs)) for lv, cs in v["rel"].items()},
                       "coverage_conformal_mean": float(np.mean(v["conf"])) if v["conf"] else None,
                       "coverage_raw_eval_mean": float(np.mean(v["raw_eval"])) if v["raw_eval"] else None,
                       "nominal": 0.80}
                   for m, v in calib.items()}
    (outdir / "stats" / "calibration.json").write_text(json.dumps(calibration, indent=2))

    # ---- few-shot curve: macro-RMSE vs regime per model -----------------------
    fs = defaultdict(lambda: defaultdict(list))
    for u in units:
        if u.get("regime") in ("full", "fewshot50", "fewshot25"):
            r = macro_rmse(u)
            if r is not None:
                fs[u["model"]][u["regime"]].append(r)
    fewshot = {m: {reg: {"mean": float(np.mean(v)), "ci95": bootstrap_ci(v)}
                   for reg, v in regs.items()}
               for m, regs in fs.items()}
    (outdir / "stats" / "fewshot.json").write_text(json.dumps(fewshot, indent=2))

    print(f"[stats] {len(units)} units | {len(models)} models | {len(datasets)} datasets "
          f"| blocks={len(blocks)} | best_classical={best_classical}")
    print(f"[stats] wrote omnibus, posthoc, pairwise, summary, calibration, fewshot to {outdir/'stats'}")


if __name__ == "__main__":
    main()
