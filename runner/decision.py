#!/usr/bin/env python3
"""
Acceptance-decision layer — turns a predictive interval into a shop-floor decision.

This is a POST-HOC pass over a finished run. It reads the persisted quantile grid
(`pred_q`, payload v2) and never refits anything, so the tolerance band, the risk
level and the criterion can all be changed without touching the GPU.

WHAT IT DOES
Given a coupon, a model's predictive interval at risk alpha, and an engineering
acceptance band [lo, hi] (either side may be None = one-sided):

    ACCEPT   interval entirely inside the band   -> ship without sectioning it
    REJECT   interval entirely outside the band  -> scrap, no need to section it
    ABSTAIN  interval straddles a limit          -> cut the cross-section

and scores the decisions against the measured geometry:

    decision rate     fraction ruled on = destructive tests avoided
    false accept      accepted but actually out of band  -> the SAFETY error
    false reject      rejected but actually in band      -> the COST error

The asymmetry matters and is not cosmetic. On this weldment the depth band is
one-sided (1.0 +0.5/-0): under-penetration is lack of fusion, over-penetration up
to +0.5 is tolerated. A false accept below the lower limit is a structural defect;
a false reject is a scrapped part. They are not interchangeable and are reported
separately, never pooled into an "accuracy".

WHY THIS IS NOT JUST COVERAGE
A model with badly narrow intervals looks DECISIVE here — it accepts more, because
a narrow interval fits inside the band more often — while its accepts are unfounded.
Interval coverage alone cannot express that; false-accept rate at a fixed decision
rate can. That is the whole point of the layer.

CONFORMAL CONSTRUCTION (--conformal)
Two constructions, both post-hoc over the same persisted grid, so reporting both
costs nothing beyond a second read of the run:

  cvplus (default)  Conformalise the CQR score (Romano et al. 2019) across the
                    leave-one-group-out folds in the CV+ style (Barber et al. 2021).
                    For fold g the offset comes from the scores of every OTHER fold,
                    so no coupon calibrates its own interval, and NO settings are
                    spent on a calibration split — decisive on 15 regimes. The
                    guarantee is 1-2*alpha rather than 1-alpha: conservative in
                    theory, and in practice CV+ coverage sits close to nominal.

  split             Textbook split-conformal: hold out whole GROUPS as a calibration
                    set, calibrate once, evaluate on the rest. Gives the clean
                    1-alpha guarantee, but spends settings that the campaign cannot
                    spare (--cal-frac 0.2 of 15 regimes = 3 settings withdrawn from
                    training). Reported as the comparison arm: the gap between the
                    two IS a finding — it prices, in decision rate, what the textbook
                    guarantee costs on a 15-setting campaign.

THE H/B RATIO
The process recommendation is a RATIO criterion (H/B >= 1). A ratio interval does
not follow from the marginal quantiles of H and of B: dividing interval endpoints
assumes a dependence that was never estimated. Instead the ratio is treated as its
own predictand — point estimate mean(H)/mean(B), interval by conformalising the
ratio residual directly on the out-of-fold scores. This is exact under the same
exchangeability assumption as the other two, and it needs no joint quantiles.

JOINT ACCEPTANCE
A coupon is shippable only if EVERY criterion holds. Guaranteeing joint risk alpha
across k criteria needs a union bound: each criterion is built at alpha/k
(Bonferroni). `--joint-correction none` reports the naive marginal version for
comparison; the paper reports both, because the gap is worth a sentence.

USAGE
    python runner/decision.py --in runs/final --out runs/final \
        --dataset ebw --alpha 0.05 --tier production
    python runner/decision.py --in runs/final --out runs/final --dataset ebw --sweep

Writes stats/decision_<tier>.json and, with --sweep, stats/decision_sweep.json:
the whole decision-rate / false-accept curve against a scaling of the band, whose
headline number is the TIGHTEST band each model can certify at the given risk.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_units(indir: Path, dataset: str, regime: str = "full"):
    """Every non-skipped unit of one dataset/regime that carries a quantile grid,
    reduced to ONE record per (model, fold) by collapsing the seed axis.

    Why collapse. The seed axis was created to give the tables per-seed dispersion. It
    is NOT a source of independent test coupons: under a LOGO split the split is
    deterministic, so the ten seeds of a TFM are bit-identical (measured sd = 0.000000),
    and even for the stochastic controls the ten predictions are of the SAME held-out
    coupons. Pooling all ten into the decision layer would report n = 720 where there
    are 72 distinct coupons, shrinking every confidence interval by ~sqrt(10) on point
    estimates that are largely copies — a false precision exactly analogous to the
    input-leakage the whole repositioning was about, now in the denominator.

    So: average the per-instance mean and quantile grid across seeds within each
    (model, fold). For a deterministic TFM this changes nothing (the seeds are equal);
    for a stochastic control it reports the seed-averaged predictor, which is the right
    object to make a decision with. n then equals the number of held-out coupons.
    """
    by_key = defaultdict(list)
    for p in sorted(indir.glob(f"{dataset}__{regime}__*__seed*__fold*.json")):
        u = json.loads(p.read_text())
        if u.get("skipped") or u.get("pred_q") is None:
            continue
        if u.get("payload_version", 1) < 2:
            sys.exit(f"[decision] {p.name} is payload v1 (no pred_q). "
                     f"This run predates the quantile-persistence fix; re-run required.")
        by_key[(u["model"], u["fold"])].append(u)
    if not by_key:
        sys.exit(f"[decision] no usable units for {dataset}/{regime} in {indir}")

    units = []
    for (model, fold), group in by_key.items():
        base = dict(group[0])
        base["pred_mean"] = np.mean([np.asarray(g["pred_mean"]) for g in group], axis=0).tolist()
        base["pred_q"] = np.mean([np.asarray(g["pred_q"]) for g in group], axis=0).tolist()
        base["n_seeds_collapsed"] = len(group)
        units.append(base)
    return units


def band_of(spec, tier, target):
    """(lo, hi) for one target in one acceptance tier; None = unbounded that side."""
    b = spec["acceptance"][tier].get(target) or {}
    return b.get("lo"), b.get("hi")


# --------------------------------------------------------------------------- #
# Conformal (CQR score, CV+ aggregation)                                       #
# --------------------------------------------------------------------------- #
def cqr_scores(y, lo, hi):
    """Romano et al. (2019) conformity score: how far outside its own interval the
    truth fell. Negative when the interval already contains y."""
    return np.maximum(lo - y, y - hi)


def _base_quantile_idx(levels, alpha):
    """Indices of the base quantiles for a (1-alpha) interval, or raise.

    The old code silently fell back to the extreme grid levels when alpha/2 was not
    present, which is how the production tier came back as a flat 0% with no warning:
    a 5% joint risk over 2 criteria needs the 1.25% and 98.75% quantiles, the grid
    stops at 5% and 95%, so a 90% base was quietly inflated by a scalar conformal
    offset and reported as if it were a 97.5% interval. A scalar offset shifts an
    interval, it does not re-shape it into a tighter tail — the result is valid but
    needlessly wide, and nothing said so.

    Refuse instead. The grid is QLEVELS from the run; representable joint risks under
    Bonferroni over k criteria are alpha such that alpha/(2k) is IN the grid. For
    QLEVELS=[.05,.1,.25,.5,.75,.9,.95] and k=2 that means alpha in {0.2, 0.4}; for
    k=1, alpha in {0.1, 0.2}. Anything tighter needs a denser grid and a re-run.
    """
    lo, hi = alpha / 2, 1 - alpha / 2
    if lo not in levels or hi not in levels:
        raise ValueError(
            f"alpha={alpha:g} needs base quantiles {lo:g}/{hi:g}, and the run's grid is "
            f"{levels}. Refusing to substitute the extremes and call the result a "
            f"{100*(1-alpha):g}% interval. Either pick a representable alpha "
            f"(--alpha such that alpha/(2*n_criteria) is in the grid), or re-run with "
            f"a denser QLEVELS.")
    return levels.index(lo), levels.index(hi)


def cvplus_offset(scores_other, alpha):
    """Offset from the folds that are NOT being evaluated. The ceil((n+1)(1-alpha))/n
    quantile is the finite-sample-valid choice; clipped to [0, 1] for tiny n."""
    n = len(scores_other)
    if n == 0:
        return 0.0
    k = min(max(np.ceil((n + 1) * (1 - alpha)) / n, 0.0), 1.0)
    return float(np.quantile(scores_other, k, method="higher"))


def conformal_intervals(units, target_idx, alpha, ratio=False, mode="cvplus",
                        cal_frac=0.2, seed=0):
    """Conformalised intervals for one model.

    mode='cvplus'  offset for fold g from the scores of all OTHER folds (no fold
                   calibrates itself, nothing is withdrawn from training).
    mode='split'   whole folds (= whole settings) are withdrawn as a calibration
                   set; one offset, applied to the remaining folds. The withdrawn
                   settings are NOT evaluated, so the returned n is smaller — that
                   shrinkage is exactly the cost this arm exists to measure.

    Returns (y_true, lo, hi, fold) pooled over the evaluated folds and seeds. For
    ratio=True the predictand is Depth/Width and the interval comes from the ratio
    residual, not from dividing the marginal quantiles.
    """
    per_fold = defaultdict(list)
    for u in units:
        q = np.asarray(u["pred_q"])            # (n_te, n_targets, n_levels)
        y = np.asarray(u["y_test"])            # (n_te, n_targets)
        m = np.asarray(u["pred_mean"])
        lv = list(u["q_levels"])
        i_lo, i_hi = _base_quantile_idx(lv, alpha)
        if ratio:
            yy = y[:, 0] / y[:, 1]
            pt = m[:, 0] / m[:, 1]
            lo = hi = pt                       # residual conformal: interval built below
        else:
            yy = y[:, target_idx]
            lo, hi = q[:, target_idx, i_lo], q[:, target_idx, i_hi]
        per_fold[u["fold"]].append((yy, lo, hi))

    packed = {f: tuple(np.concatenate(x) for x in zip(*v)) for f, v in per_fold.items()}
    scores = {f: cqr_scores(*v) for f, v in packed.items()}
    out_y, out_lo, out_hi, out_f = [], [], [], []

    if mode == "split":
        folds = sorted(packed)
        rng = np.random.default_rng(seed)
        k = max(1, int(round(cal_frac * len(folds))))
        cal = set(rng.choice(folds, size=min(k, len(folds) - 1), replace=False).tolist())
        off = cvplus_offset(np.concatenate([scores[f] for f in cal]), alpha)
        for f in folds:
            if f in cal:                       # spent on calibration, not evaluated
                continue
            yy, lo, hi = packed[f]
            out_y.append(yy); out_lo.append(lo - off); out_hi.append(hi + off)
            out_f.append(np.full(len(yy), f))
    else:
        for f, (yy, lo, hi) in packed.items():
            other = np.concatenate([s for g, s in scores.items() if g != f]) \
                    if len(scores) > 1 else scores[f]
            off = cvplus_offset(other, alpha)
            out_y.append(yy); out_lo.append(lo - off); out_hi.append(hi + off)
            out_f.append(np.full(len(yy), f))

    return (np.concatenate(out_y), np.concatenate(out_lo),
            np.concatenate(out_hi), np.concatenate(out_f))


# --------------------------------------------------------------------------- #
# Decisions                                                                    #
# --------------------------------------------------------------------------- #
def decide(lo, hi, band_lo, band_hi):
    """accept / reject / abstain per coupon, against a possibly one-sided band."""
    bl = -np.inf if band_lo is None else band_lo
    bh = np.inf if band_hi is None else band_hi
    accept = (lo >= bl) & (hi <= bh)
    reject = (hi < bl) | (lo > bh)
    return accept, reject


def score_decisions(y, lo, hi, band_lo, band_hi):
    bl = -np.inf if band_lo is None else band_lo
    bh = np.inf if band_hi is None else band_hi
    truth_in = (y >= bl) & (y <= bh)
    accept, reject = decide(lo, hi, band_lo, band_hi)
    n = len(y)
    return {
        "n": int(n),
        "decision_rate": float(np.mean(accept | reject)),
        "abstain_rate": float(np.mean(~(accept | reject))),
        "accept_rate": float(np.mean(accept)),
        "reject_rate": float(np.mean(reject)),
        # SAFETY error: shipped a coupon whose real geometry is out of band.
        "false_accept_rate": float(np.mean(accept & ~truth_in)),
        "false_accept_of_accepted": float(np.mean(~truth_in[accept])) if accept.any() else None,
        # COST error: scrapped a coupon that was actually fine.
        "false_reject_rate": float(np.mean(reject & truth_in)),
        "false_reject_of_rejected": float(np.mean(truth_in[reject])) if reject.any() else None,
        "prevalence_in_band": float(np.mean(truth_in)),
    }


def evaluate(units_by_model, spec, tier, alpha, joint_correction="bonferroni",
             mode="cvplus", cal_frac=0.2):
    targets = spec["targets"]
    crits = [t for t in list(targets) + ["HB"]
             if any(v is not None for v in band_of(spec, tier, t))]
    a_eff = alpha / len(crits) if (joint_correction == "bonferroni" and crits) else alpha

    out = {"tier": tier, "alpha": alpha, "alpha_per_criterion": a_eff,
           "criteria": crits, "joint_correction": joint_correction,
           "conformal": mode, "cal_frac": (cal_frac if mode == "split" else None),
           "models": {}}
    for model, units in units_by_model.items():
        rec, joint_acc, joint_rej, joint_ok = {}, None, None, None
        for c in crits:
            is_ratio = (c == "HB")
            idx = 0 if is_ratio else list(targets).index(c)
            y, lo, hi = conformal_intervals(units, idx, a_eff, ratio=is_ratio,
                                            mode=mode, cal_frac=cal_frac)[:3]
            bl, bh = band_of(spec, tier, c)
            rec[c] = score_decisions(y, lo, hi, bl, bh)
            acc, rej = decide(lo, hi, bl, bh)
            ok = (y >= (-np.inf if bl is None else bl)) & (y <= (np.inf if bh is None else bh))
            joint_acc = acc if joint_acc is None else (joint_acc & acc)
            joint_rej = rej if joint_rej is None else (joint_rej | rej)
            joint_ok = ok if joint_ok is None else (joint_ok & ok)
        rec["JOINT"] = {
            "n": int(len(joint_acc)),
            "decision_rate": float(np.mean(joint_acc | joint_rej)),
            "accept_rate": float(np.mean(joint_acc)),
            "false_accept_rate": float(np.mean(joint_acc & ~joint_ok)),
            "false_accept_of_accepted": (float(np.mean(~joint_ok[joint_acc]))
                                         if joint_acc.any() else None),
            "prevalence_in_band": float(np.mean(joint_ok)),
        }
        out["models"][model] = rec
    return out


def sweep(units_by_model, spec, tier, alpha, scales, mode="cvplus"):
    """Scale the band half-width about its centre and re-decide. The reportable
    number is the tightest band each model still certifies at risk alpha."""
    targets = list(spec["targets"])
    rows = []
    for s in scales:
        tightened = {"acceptance": {tier: {}}, "targets": targets}
        for t in targets + ["HB"]:
            lo, hi = band_of(spec, tier, t)
            if lo is None and hi is None:
                tightened["acceptance"][tier][t] = {"lo": None, "hi": None}
            elif lo is not None and hi is not None:
                c, h = (lo + hi) / 2, (hi - lo) / 2 * s
                tightened["acceptance"][tier][t] = {"lo": c - h, "hi": c + h}
            else:
                tightened["acceptance"][tier][t] = {"lo": lo, "hi": hi}   # one-sided: unscaled
        r = evaluate(units_by_model, tightened, tier, alpha, mode=mode)
        rows.append({"scale": s, "models": {m: v["JOINT"] for m, v in r["models"].items()}})
    return {"tier": tier, "alpha": alpha, "conformal": mode, "sweep": rows}


def sweep_alpha(units_by_model, spec, tier, alphas, mode="cvplus"):
    """Sweep the RISK, at the real tolerance. The band sweep asks 'how tight a tolerance
    can this model certify'; this asks the dual and more actionable question: 'at the
    tolerance the shop actually uses, what risk must I accept to get a decision at all'.

    The production band is 0.5 mm on depth. At a 5% joint risk the interval a model must
    produce is ~0.7 mm wide — wider than the band, so nothing can be accepted by anyone,
    and the layer correctly returns a flat zero. That is arithmetic, not a property of
    any model, and reporting only that point would hide the whole result.
    """
    rows = []
    for a in alphas:
        try:
            r = evaluate(units_by_model, spec, tier, a, mode=mode)
        except ValueError as e:
            rows.append({"alpha": a, "unrepresentable": str(e)}); continue
        rows.append({"alpha": a, "models": {m: v["JOINT"] for m, v in r["models"].items()}})
    return {"tier": tier, "conformal": mode, "sweep_alpha": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--dataset", default="ebw")
    ap.add_argument("--regime", default="full")
    ap.add_argument("--alpha", type=float, default=0.05, help="risk of a false accept")
    ap.add_argument("--tier", default="production", help="acceptance tier in datasets.yaml")
    ap.add_argument("--joint-correction", default="bonferroni", choices=["bonferroni", "none"])
    ap.add_argument("--conformal", default="cvplus", choices=["cvplus", "split"],
                    help="cvplus: no settings spent on calibration, guarantee 1-2a. "
                         "split: textbook 1-a, but withdraws --cal-frac of the settings.")
    ap.add_argument("--cal-frac", dest="cal_frac", type=float, default=0.2,
                    help="split only: fraction of GROUPS withdrawn for calibration")
    ap.add_argument("--sweep", action="store_true", help="sweep the tolerance band")
    ap.add_argument("--sweep-alpha", dest="sweep_alpha", action="store_true",
                    help="sweep the risk at the real band")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()

    spec = yaml.safe_load((Path(a.data_dir) / "datasets.yaml").read_text())[a.dataset]
    if "acceptance" not in spec:
        sys.exit(f"[decision] dataset '{a.dataset}' declares no acceptance bands")

    units = load_units(Path(a.indir), a.dataset, a.regime)
    by_model = defaultdict(list)
    for u in units:
        by_model[u["model"]].append(u)

    outdir = Path(a.outdir) / "stats"; outdir.mkdir(parents=True, exist_ok=True)
    res = evaluate(by_model, spec, a.tier, a.alpha, a.joint_correction,
                   mode=a.conformal, cal_frac=a.cal_frac)
    tag = a.tier if a.conformal == "cvplus" else f"{a.tier}_split"
    (outdir / f"decision_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[decision] tier={a.tier} alpha={a.alpha} conformal={a.conformal} "
          f"-> stats/decision_{tag}.json")
    for m, r in res["models"].items():
        j = r["JOINT"]
        print(f"   {m:12s} decides {j['decision_rate']:5.1%} of coupons | "
              f"false accepts {j['false_accept_rate']:5.1%}")

    if a.sweep_alpha:
        s = sweep_alpha(by_model, spec, a.tier, [0.4, 0.2], mode=a.conformal)
        (outdir / f"decision_alpha_sweep_{a.tier}.json").write_text(json.dumps(s, indent=2))
        print(f"[decision] alpha sweep -> stats/decision_alpha_sweep_{a.tier}.json")

    if a.sweep:
        s = sweep(by_model, spec, a.tier, a.alpha,
                  [round(x, 2) for x in np.arange(0.1, 1.55, 0.05)], mode=a.conformal)
        (outdir / "decision_sweep.json").write_text(json.dumps(s, indent=2))
        print("[decision] sweep -> stats/decision_sweep.json")


if __name__ == "__main__":
    main()
