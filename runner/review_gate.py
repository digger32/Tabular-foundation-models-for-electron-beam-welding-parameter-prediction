#!/usr/bin/env python3
"""
Review-proofing gate — A (EBW applied TFM benchmark).

Reads run_meta.json + manifest.jsonl + per-unit JSONs and asserts:
  A1  clean final run     resume DISABLED, no unit skipped-by-resume, one run_started
  B1  external validity    the comparative claim is replicated on >=1 public dataset
  C1  calibration present  interval coverage / ECE recorded for the distributional arm
  D1  optimism gap         inner-CV vs held-out gap recorded for the tuned controls
  E1  stats present        omnibus + post-hoc artifacts exist

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
    stale = [r["unit"] for r in manifest if r.get("started") != rs]
    if stale:
        return False, f"{len(stale)} unit(s) carry a different run_started: {stale[:3]}..."
    return True, "final pass clean: --no-resume, no skips, single run_started"


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
               ("D1 optimism-gap", *check_D1(o, units, cfg))]
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
