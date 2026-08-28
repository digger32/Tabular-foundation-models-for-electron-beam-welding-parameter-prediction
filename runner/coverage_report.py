"""Coverage of the nominal 80% interval per conformal mode, on the frozen residuals.
This is the methodological artifact for the RESS version: nexCP holds coverage near
nominal AND carries a bounded gap, split does not. Writes stats/coverage_by_mode.json."""
import sys, json; sys.path.insert(0,'runner')
import numpy as np
from pathlib import Path
from decision import load_units, conformal_intervals, regime_centers

def main():
    alpha=0.20
    units=load_units(Path("runs/final"),"ebw","full")
    by={}
    for u in units: by.setdefault(u["model"],[]).append(u)
    folds=sorted({u["fold"] for u in units})
    centers=regime_centers("ebw","data/datasets.yaml",folds)
    out={"nominal":1-alpha,"target":"Depth","by_model":{},"tfm_mean":{}}
    for m,us in by.items():
        out["by_model"][m]={}
        for mode in ("cvplus","split","nexcp"):
            y,lo,hi,_=conformal_intervals(us,0,alpha,mode=mode,centers=centers)
            out["by_model"][m][mode]=round(float(np.mean((y>=lo)&(y<=hi))),3)
    # Average over every tabular foundation model present, not a hard-coded trio. The
    # table is captioned "averaged over the foundation models", so a member that is
    # excluded because it was added later makes the caption false, and the released
    # per-unit predictions let a reader recompute the mean and find the discrepancy.
    TFM=("tabpfn_v2","tabpfn_v25","tabpfn_v3","tabiclv2","mitra")
    tfm=[m for m in TFM if m in out["by_model"]]
    out["tfm_models"]=tfm
    for mode in ("cvplus","split","nexcp"):
        cs=[out["by_model"][m][mode] for m in tfm]
        out["tfm_mean"][mode]={"coverage":round(float(np.mean(cs)),3),
                               "gap_from_nominal":round(float(abs(np.mean(cs)-(1-alpha))),3)}
    Path("runs/final/stats/coverage_by_mode.json").write_text(json.dumps(out,indent=2))
    print("wrote stats/coverage_by_mode.json")
    print("averaged over:", out["tfm_models"])
    print(json.dumps(out["tfm_mean"],indent=2))

if __name__=="__main__": main()
