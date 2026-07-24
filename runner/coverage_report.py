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
    for mode in ("cvplus","split","nexcp"):
        cs=[out["by_model"][m][mode] for m in ("tabpfn_v2","tabpfn_v25","tabpfn_v3")]
        out["tfm_mean"][mode]={"coverage":round(float(np.mean(cs)),3),
                               "gap_from_nominal":round(float(abs(np.mean(cs)-(1-alpha))),3)}
    Path("runs/final/stats/coverage_by_mode.json").write_text(json.dumps(out,indent=2))
    print("wrote stats/coverage_by_mode.json")
    print(json.dumps(out["tfm_mean"],indent=2))

if __name__=="__main__": main()
