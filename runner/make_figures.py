#!/usr/bin/env python3
"""
A (EBW) figures. Reads the runner's per-unit JSONs + stats/*.json and renders the
paper display items (figure_plan.md) as PDF + PNG.

  Fig 1  C1  per-model macro-RMSE on EBW (regime=full), TFM vs tuned classical,
             with bootstrap CI and Wilcoxon-Holm significance vs the best classical.
  Fig 2  C2  few-shot curve: macro-RMSE vs context fraction (100/50/25%) per model.
  Fig 3  C3  TFM(full) vs best classical(full) vs best classical(augment) on EBW.
  Fig 4  C4  calibration: 80% prediction-interval coverage vs nominal, per model.
  Fig A1     external validity: per-dataset (ebw, gmaw_e1, gmaw_e2) TFM vs classical.

Usage:
    python make_figures.py --in runs/final --out outputs/figures
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TFM = ["tabpfn_v2", "tabpfn_v25", "tabpfn_v3"]
CLASSICAL = ["catboost", "xgb", "ngb", "mlp"]
# colourblind-safe (Wong); TFMs warm, classical cool
COLOR = {"tabpfn_v2": "#E69F00", "tabpfn_v25": "#D55E00", "tabpfn_v3": "#CC79A7",
         "mitra": "#F0E442", "catboost": "#0072B2", "xgb": "#56B4E9",
         "ngb": "#009E73", "mlp": "#999999"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120})


def load_units(indir):
    out = []
    for p in indir.glob("*__*__*__seed*.json"):
        try:
            u = json.loads(p.read_text())
            if not u.get("skipped"):
                out.append(u)
        except Exception:
            pass
    return out


def load_json(indir, name):
    p = indir / "stats" / name
    return json.loads(p.read_text()) if p.exists() else {}


def macro_rmse(u):
    pt = (u.get("metrics") or {}).get("per_target") or {}
    v = [t["rmse"] for t in pt.values() if t.get("rmse") is not None]
    return float(np.mean(v)) if v else None


def ci(vals):
    vals = np.asarray([v for v in vals if v is not None], float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(0)
    b = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(2000)]
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}.pdf/.png")


def models_present(units, dataset=None, regime=None):
    ms = {u["model"] for u in units
          if (dataset is None or u["dataset"] == dataset)
          and (regime is None or u["regime"] == regime)}
    return [m for m in TFM + CLASSICAL if m in ms]


def fig1(units, pairwise, out):
    ds = "ebw"
    ms = models_present(units, ds, "full")
    vals = {m: [macro_rmse(u) for u in units if u["dataset"] == ds and u["regime"] == "full"
                and u["model"] == m] for m in ms}
    means = {m: ci(vals[m]) for m in ms}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(ms))
    ax.bar(x, [means[m][0] for m in ms],
           yerr=[[means[m][0]-means[m][1] for m in ms], [means[m][2]-means[m][0] for m in ms]],
           color=[COLOR.get(m, "#777") for m in ms], capsize=3)
    tests = (pairwise or {}).get("tests", {})
    ref = (pairwise or {}).get("reference_best_classical")
    for i, m in enumerate(ms):
        key = f"{m}_vs_{ref}"
        ph = tests.get(key, {}).get("p_holm")
        if ph is not None and ph < 0.05:
            ax.text(i, means[m][2], "*", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=30, ha="right")
    ax.set_ylabel("macro-RMSE (Depth, Width)")
    ax.set_title(f"EBW: foundation models vs tuned classical  (* Holm p<0.05 vs {ref})")
    save(fig, out, "fig1_ebw_model_rmse")


def fig2(fewshot, out):
    order = {"full": 1.0, "fewshot50": 0.5, "fewshot25": 0.25}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m, regs in (fewshot or {}).items():
        pts = sorted([(order[r], v["mean"]) for r, v in regs.items() if r in order])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=COLOR.get(m, "#777"), label=m)
    ax.set_xlabel("context fraction"); ax.set_ylabel("macro-RMSE")
    ax.set_title("Few-shot: performance as the context shrinks")
    ax.invert_xaxis(); ax.legend(fontsize=8, ncol=2)
    save(fig, out, "fig2_fewshot_curve")


def fig3(units, out):
    ds = "ebw"
    def mean_rmse(model, regime):
        v = [macro_rmse(u) for u in units if u["dataset"] == ds and u["model"] == model
             and u["regime"] == regime]
        v = [x for x in v if x is not None]
        return np.mean(v) if v else np.nan
    tfm_full = {m: mean_rmse(m, "full") for m in models_present(units, ds, "full") if m in TFM}
    best_tfm = min(tfm_full, key=tfm_full.get) if tfm_full else None
    classic = [m for m in CLASSICAL if not np.isnan(mean_rmse(m, "full"))]
    best_cls = min(classic, key=lambda m: mean_rmse(m, "full")) if classic else None
    bars = {}
    if best_tfm:
        bars[f"{best_tfm}\n(full)"] = mean_rmse(best_tfm, "full")
    if best_cls:
        bars[f"{best_cls}\n(full)"] = mean_rmse(best_cls, "full")
        aug = mean_rmse(best_cls, "augment")
        if not np.isnan(aug):
            bars[f"{best_cls}\n(+CTGAN/TVAE)"] = aug
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(range(len(bars)), list(bars.values()),
           color=["#D55E00", "#0072B2", "#56B4E9"][:len(bars)])
    ax.set_xticks(range(len(bars))); ax.set_xticklabels(list(bars.keys()))
    ax.set_ylabel("macro-RMSE"); ax.set_title("In-context TFM vs the augmentation pipeline (EBW)")
    save(fig, out, "fig3_tfm_vs_augmentation")


def fig4(calibration, out):
    cal = calibration or {}
    ms = [m for m in TFM + ["mitra", "ngb"] if m in cal]
    if not ms:
        print("[fig] fig4 skipped (no calibration data)"); return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(ms))
    cov = [cal[m]["coverage_80pi_mean"] for m in ms]
    lo = [cal[m]["coverage_80pi_mean"] - cal[m]["coverage_ci95"][0] for m in ms]
    hi = [cal[m]["coverage_ci95"][1] - cal[m]["coverage_80pi_mean"] for m in ms]
    ax.bar(x, cov, yerr=[lo, hi], color=[COLOR.get(m, "#777") for m in ms], capsize=3)
    ax.axhline(0.80, ls="--", color="k", lw=1, label="nominal 0.80")
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=30, ha="right")
    ax.set_ylabel("80% PI coverage"); ax.set_ylim(0, 1)
    ax.set_title("Calibration of predictive intervals"); ax.legend(fontsize=8)
    save(fig, out, "fig4_calibration")


def figA1(units, out):
    datasets = [d for d in ["ebw", "gmaw_e1", "gmaw_e2"]
                if any(u["dataset"] == d for u in units)]
    def grp_mean(ds, group):
        v = [macro_rmse(u) for u in units if u["dataset"] == ds and u["regime"] == "full"
             and u["model"] in group]
        v = [x for x in v if x is not None]
        return np.mean(v) if v else np.nan
    tfm = [grp_mean(d, TFM) for d in datasets]
    cls = [grp_mean(d, CLASSICAL) for d in datasets]
    x = np.arange(len(datasets)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(x - w/2, tfm, w, label="TFM (mean)", color="#D55E00")
    ax.bar(x + w/2, cls, w, label="classical (mean)", color="#0072B2")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("macro-RMSE"); ax.set_title("External validity across datasets")
    ax.legend(fontsize=8)
    save(fig, out, "figA1_external_validity")




# ---- extended figures (P1/P2): CD, reliability, scatter, residuals, heatmap, time ----
def fig_cd(indir, out):
    p = load_json(indir, "posthoc.json")
    if not p or "mean_rank" not in p: 
        print("[fig] cd skipped"); return
    ranks = p["mean_rank"]; CD = p["critical_difference"]
    items = sorted(ranks.items(), key=lambda kv: kv[1]); vals=[v for _,v in items]
    n=len(items); lo,hi=1,n
    fig,ax=plt.subplots(figsize=(7.0,2.6)); ax.set_xlim(lo-0.2,hi+0.2); ax.set_ylim(0,1); ax.axis("off")
    ax.plot([lo,hi],[0.85,0.85],"k-",lw=1)
    for r in range(lo,hi+1): ax.plot([r,r],[0.85,0.88],"k-"); ax.text(r,0.92,str(r),ha="center",fontsize=9)
    ax.plot([lo,lo+CD],[0.72,0.72],"k-",lw=2); ax.text(lo+CD/2,0.64,f"CD = {CD:.2f}",ha="center",fontsize=8)
    half=(n+1)//2
    for i,(m,v) in enumerate(items):
        y=(0.55-0.45*(i/half)) if i<half else (0.55-0.45*((i-half)/max(1,n-half)))
        side_x = lo-0.15 if i<half else hi+0.15; ha="right" if i<half else "left"
        ax.plot([v,v],[0.85,y],"k-",lw=0.8); ax.plot([v,side_x],[y,y],"k-",lw=0.8)
        ax.text(side_x+(-0.03 if i<half else 0.03),y,m,ha=ha,va="center",fontsize=9)
    yb=0.80
    for i in range(n):
        j=i
        while j+1<n and vals[j+1]-vals[i]<=CD: j+=1
        if j>i: ax.plot([vals[i]-0.03,vals[j]+0.03],[yb,yb],"-",lw=3,color="0.25"); yb-=0.03
    save(fig,out,"fig_cd_diagram")

def fig_reliability(indir, out):
    cal = load_json(indir, "calibration.json")
    rel = {m:v.get("reliability") for m,v in (cal or {}).items() if v.get("reliability")}
    if not rel: print("[fig] reliability skipped"); return
    fig,ax=plt.subplots(figsize=(4.6,4.4))
    ax.plot([0,1],[0,1],"k--",lw=1,label="ideal")
    for m,r in rel.items():
        lv=sorted(float(k) for k in r); ax.plot(lv,[r[str(x)] for x in lv],"-o",ms=3,
                  color=COLOR.get(m,"#777"),label=m)
    ax.set_xlabel("nominal quantile"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability of predicted quantiles"); ax.legend(fontsize=7)
    save(fig,out,"fig_reliability")

def fig_scatter(units, out, ds="ebw"):
    best_tfm, best_cls = "tabpfn_v3", "catboost"
    def collect(model):
        xs,ys=[],[]
        for u in units:
            if u["dataset"]==ds and u["regime"]=="full" and u["model"]==model:
                pm=u.get("pred_mean"); yt=u.get("y_test")
                if pm and yt:
                    import numpy as np
                    xs+= list(np.ravel(yt)); ys+= list(np.ravel(pm))
        return xs,ys
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.6))
    for ax,model in zip(axes,[best_tfm,best_cls]):
        xs,ys=collect(model)
        if xs:
            ax.scatter(xs,ys,s=6,alpha=0.4,color=COLOR.get(model,"#777"))
            lim=[min(xs+ys),max(xs+ys)]; ax.plot(lim,lim,"k--",lw=1)
        ax.set_title(model); ax.set_xlabel("actual"); ax.set_ylabel("predicted")
    fig.suptitle(f"Predicted vs actual on {ds} (full context)")
    save(fig,out,"fig_pred_vs_actual")

def fig_residuals(units, out, ds="ebw"):
    import numpy as np
    fig,ax=plt.subplots(figsize=(5.2,3.4))
    for model in ["tabpfn_v3","catboost","mlp"]:
        res=[]
        for u in units:
            if u["dataset"]==ds and u["regime"]=="full" and u["model"]==model:
                pm=u.get("pred_mean"); yt=u.get("y_test")
                if pm and yt: res+= list((np.ravel(np.array(yt))-np.ravel(np.array(pm))))
        if res: ax.hist(res,bins=30,histtype="step",lw=1.5,color=COLOR.get(model,"#777"),label=model)
    ax.axvline(0,color="k",lw=0.8); ax.set_xlabel("residual (actual - predicted)")
    ax.set_ylabel("count"); ax.set_title(f"Residual distribution on {ds}"); ax.legend(fontsize=8)
    save(fig,out,"fig_residuals")

def fig_corr_heatmap(out, data_csv="data/ebw_real_72.csv"):
    import numpy as np, pandas as pd, os
    if not os.path.exists(data_csv): print("[fig] heatmap skipped"); return
    df=pd.read_csv(data_csv); cols=["IW","IF","VW","FP","Depth","Width"]
    cols=[c for c in cols if c in df.columns]; C=df[cols].corr().to_numpy()
    fig,ax=plt.subplots(figsize=(4.6,4.0)); im=ax.imshow(C,vmin=-1,vmax=1,cmap="coolwarm")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=45,ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)): ax.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,fraction=0.046); ax.set_title("EBW input-target correlation")
    save(fig,out,"fig_corr_heatmap")

def fig_inference_time(indir, out):
    import json,numpy as np
    mf=indir/"manifest.jsonl"
    if not mf.exists(): print("[fig] inference-time skipped"); return
    wall=defaultdict(list)
    for line in mf.read_text().splitlines():
        try:
            r=json.loads(line)
            if r.get("status")=="ok" and r.get("regime")=="full" and r.get("wall_s"):
                wall[r["model"]].append(r["wall_s"])
        except Exception: pass
    if not wall: print("[fig] inference-time skipped"); return
    ms=[m for m in TFM+CLASSICAL if m in wall]; vals=[np.mean(wall[m]) for m in ms]
    fig,ax=plt.subplots(figsize=(6.0,3.4))
    ax.bar(range(len(ms)),vals,color=[COLOR.get(m,"#777") for m in ms])
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms,rotation=30,ha="right")
    ax.set_ylabel("mean wall time per unit (s)")
    ax.set_title("Cost: in-context inference vs tuned-control training")
    save(fig,out,"fig_inference_time")




def fig_conformal(indir, out):
    cal = load_json(indir, "calibration.json")
    rows=[(m,v.get("coverage_raw_eval_mean"),v.get("coverage_conformal_mean"))
          for m,v in (cal or {}).items()
          if v.get("coverage_conformal_mean") is not None]
    if not rows: print("[fig] conformal skipped"); return
    import numpy as np
    ms=[r[0] for r in rows]; raw=[r[1] for r in rows]; con=[r[2] for r in rows]
    x=np.arange(len(ms)); w=0.38
    fig,ax=plt.subplots(figsize=(6.2,3.6))
    ax.bar(x-w/2,raw,w,label="raw TFM interval",color="#c9c9e6")
    ax.bar(x+w/2,con,w,label="conformalized",color="#7b7bc0")
    ax.axhline(0.80,color="k",ls="--",lw=1,label="0.80 nominal")
    ax.set_xticks(x); ax.set_xticklabels(ms,rotation=25,ha="right")
    ax.set_ylabel("80% interval coverage"); ax.set_ylim(0,1)
    ax.set_title("Raw vs split-conformal coverage"); ax.legend(fontsize=7)
    save(fig,out,"fig_conformal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", default="outputs/figures")
    a = ap.parse_args()
    indir, outdir = Path(a.indir), Path(a.outdir)
    units = load_units(indir)
    if not units:
        raise SystemExit(f"no unit JSONs in {indir}")
    fig1(units, load_json(indir, "pairwise.json"), outdir)
    fig2(load_json(indir, "fewshot.json"), outdir)
    fig3(units, outdir)
    fig4(load_json(indir, "calibration.json"), outdir)
    figA1(units, outdir)
    fig_cd(indir, outdir)
    fig_reliability(indir, outdir)
    fig_scatter(units, outdir)
    fig_residuals(units, outdir)
    fig_corr_heatmap(outdir)
    fig_inference_time(indir, outdir)
    fig_conformal(indir, outdir)
    print(f"[figs] done -> {outdir}")


if __name__ == "__main__":
    main()
