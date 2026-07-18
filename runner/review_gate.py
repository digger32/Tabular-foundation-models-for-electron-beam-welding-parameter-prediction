#!/usr/bin/env python3
"""
Review-proofing gate — A (EBW applied TFM benchmark).

Reads run_meta.json + manifest.jsonl + per-unit JSONs and asserts:
  A1  clean final run     resume DISABLED, no unit skipped-by-resume, one run_started
  B1  external validity    the comparative claim is replicated on >=1 public dataset
  C1  calibration present  interval coverage / ECE recorded for the distributional arm
  D1  optimism gap         inner-CV vs held-out gap recorded for the tuned controls
  E1  stats present        omnibus + post-hoc artifacts exist
  F1  no input leakage      no test input vector also occurs in its own training set
  G1  splitting declared    every dataset states group_by (or group_by: null on purpose)

F1 and G1 exist because of a real failure. The EBW set was taken to be 72
independent experiments; it is 18 coupons x 4 metallographic cross-sections over
15 distinct settings, so a random split put bit-identical input vectors in both
train and test and the reported errors sat at the measurement-noise floor
(within-setting sd 0.041 mm) rather than at the scale of the signal (between-setting
sd 0.222 mm). No checklist item caught it; an assert does. F1 is cheap, mechanical
and would have failed that run on unit one.

Exits NON-ZERO on any failure so it blocks freezing:
    python review_gate.py runs/final --config gate_config.yaml && python make_figures.py
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[gate] PyYAML required: pip install pyyaml --break-system-packages"); sys.exit(2)


def load_manifest(o):
    mf = o / "manifest.jsonl"
    return [json.loads(l) for l in mf.read_text().splitlines() if l.strip()] if mf.exists() else []


def load_units(o):
    u = []
    for p in o.glob("*__*__*__seed*.json"):
        try:
            u.append(json.loads(p.read_text()))
        except Exception:
            pass
    return u


def check_A1(o, manifest, cfg):
    mp = o / "run_meta.json"
    if not mp.exists():
        return False, "run_meta.json missing"
    meta = json.loads(mp.read_text())
    if not meta.get("no_resume", False):
        return False, "final pass ran WITHOUT --no-resume"
    if any(r.get("status") == "skip" for r in manifest):
        return False, "manifest shows resume-skipped units in a no-resume pass"
    rs = meta.get("run_started")
    stale = [r["unit"] for r in manifest if r.get("run_started", r.get("started")) != rs]
    if stale:
        return False, f"{len(stale)} unit(s) carry a different run_started: {stale[:3]}..."
    # H1, folded into A1: every unit must have run on the SAME device. CatBoost's GPU
    # and CPU implementations are not numerically identical (different default border
    # handling, different tree construction), so a run stitched across devices is not
    # one experiment. The manifest records the device per unit; this asserts one value.
    devs = {(r.get("device"), r.get("tree_cpu")) for r in manifest}
    if len(devs) > 1:
        return False, f"units span multiple device configurations: {sorted(map(str, devs))}"
    if devs and None in {d for d, _ in devs}:
        return False, "units predate device recording — provenance of the compute is unverifiable"
    d, tc = next(iter(devs)) if devs else (None, None)
    return True, (f"final pass clean: --no-resume, no skips, single run_started, "
                  f"device={d}{' (trees on CPU)' if tc else ''}")


def check_B1(o, units, cfg):
    claims = cfg.get("comparative_claims", [])
    if not claims:
        return False, "no comparative_claims declared"
    present = {u.get("dataset") for u in units if not u.get("skipped")}
    fails = []
    for c in claims:
        needed = set(c.get("independent_datasets", []))
        if not (needed & present):
            fails.append(f"claim '{c.get('id')}' has no run on any of {sorted(needed)}")
    return (False, "; ".join(fails)) if fails else (True, f"{len(claims)} claim(s) replicated on a public dataset")


def check_C1(o, units, cfg):
    have = [u for u in units if (u.get("metrics") or {}).get("coverage_80pi") is not None]
    return (True, f"calibration (coverage/ECE) present in {len(have)} unit(s)") if have \
        else (False, "no unit recorded interval coverage / ECE for the distributional arm")


def check_D1(o, units, cfg):
    tuned = set(cfg.get("tuned_controls", []))
    have = [u for u in units if u.get("model") in tuned
            and (u.get("metrics") or {}).get("optimism_gap") is not None]
    if not tuned:
        return False, "tuned_controls not declared"
    return (True, f"optimism gap recorded for tuned controls ({len(have)} unit(s))") if have \
        else (False, f"no optimism_gap recorded for any of {sorted(tuned)} (HPO honesty)")


def check_E1(o, cfg):
    art = cfg.get("stats_artifacts", ["stats/omnibus.json", "stats/posthoc.json"])
    missing = [a for a in art if not (o / a).exists() and not Path(a).exists()]
    return (False, f"missing stats artifacts: {missing}") if missing else (True, f"stats artifacts present: {art}")


def check_F1(o, units, cfg):
    """No test input vector may also occur in its own training set. run_unit records
    this as `leak_check`; anything above 0 means the model was asked to predict a row
    whose exact inputs it had already been shown, and the run measures replicate
    recall, not prediction."""
    scored = [u for u in units if "leak_check" in u]
    if not scored:
        return False, ("no unit records leak_check — payload v1 run, provenance of the "
                       "split cannot be verified; re-run with the current runner")
    bad = [u for u in scored if u.get("leak_check", 0) > 0]
    if bad:
        ex = ", ".join(sorted({u["dataset"] for u in bad})[:3])
        return False, (f"{len(bad)}/{len(scored)} units leak test inputs into train "
                       f"(datasets: {ex}) — check group_by in datasets.yaml")
    return True, f"0 leaked test inputs across {len(scored)} units"


def check_G1(o, units, cfg):
    """Every dataset in the run must state its grouping in the registry: either a
    group_by list, or `group_by: null` written deliberately with a reason. An ABSENT
    key is the failure mode — silence reads as 'independent rows' and nobody checks."""
    reg_path = Path(cfg.get("datasets_yaml", "data/datasets.yaml"))
    if not reg_path.exists():
        return False, f"registry {reg_path} not found (set datasets_yaml in the gate config)"
    reg = yaml.safe_load(reg_path.read_text())
    used = sorted({u["dataset"] for u in units})
    undeclared = [d for d in used if d in reg and "group_by" not in reg[d]]
    if undeclared:
        return False, (f"datasets with no group_by declaration: {undeclared} — state the "
                       f"grouping explicitly, or `group_by: null` with a reason")
    return True, f"grouping declared for all {len(used)} datasets: {used}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir"); ap.add_argument("--config", default="gate_config.yaml")
    a = ap.parse_args()
    o = Path(a.outdir)
    cp = Path(a.config)
    if not cp.exists():
        cp = o / a.config
    cfg = yaml.safe_load(cp.read_text()) if cp.exists() else {}
    manifest, units = load_manifest(o), load_units(o)

    results = [("A1 clean-final-run", *check_A1(o, manifest, cfg)),
               ("B1 external-validity", *check_B1(o, units, cfg)),
               ("C1 calibration", *check_C1(o, units, cfg)),
               ("D1 optimism-gap", *check_D1(o, units, cfg)),
               ("F1 no-input-leakage", *check_F1(o, units, cfg)),
               ("G1 splitting-declared", *check_G1(o, units, cfg))]
    if cfg.get("require_stats"):
        results.append(("E1 stats", *check_E1(o, cfg)))

    print("=" * 64); print(f"REVIEW-PROOFING GATE | outdir={o}"); print("=" * 64)
    ok = True
    for name, passed, msg in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name:22s} {msg}"); ok = ok and passed
    print("=" * 64)
    if not ok:
        print("GATE FAILED — do not freeze these numbers into figures."); sys.exit(1)
    print("GATE PASSED — numbers are clean to freeze.")


if __name__ == "__main__":
    main()
