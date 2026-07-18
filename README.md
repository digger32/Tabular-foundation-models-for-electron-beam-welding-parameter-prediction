# Tabular foundation models for electron-beam welding parameter prediction

Reproducibility package for the study *In-context tabular foundation models for
calibrated weld-geometry prediction and acceptance decisions in electron-beam
welding* (Kurashkin, Tynchenko, Borodulin, Nelyub, Kalutsky, Connie; under review).

The study benchmarks TabPFN v2 / v2.5 / v3 (in-context, no training) against
nested-CV-tuned CatBoost, XGBoost, NGBoost and an MLP, plus a CTGAN/TVAE
augmentation arm, on a 72-row electron-beam welding (EBW) campaign and two public
gas-metal-arc-welding (GMAW) datasets. Evaluation is leakage-controlled: the EBW
coupons are sectioned four times each, so whole welding schedules are held out
(leave-one-regime-out, 15 folds); the GMAW datasets are split by welding run. On
top of accuracy (per-target RMSE/MAE/R2 with Friedman/Nemenyi and Wilcoxon–Holm
statistics, and an optimism gap for every tuned control), the package audits
interval calibration (coverage, ECE, CRPS, reliability), conformalises the
intervals (cross-validation-plus, with split conformal as a comparison), and
converts them into accept / reject / abstain decisions against the engineering
tolerance. Every prediction persists its full quantile grid, so the decision
layer can be re-run at any admissible risk level without re-running the
benchmark. The final pass sits behind an automated review-proofing gate.

## Layout
- `runner/bench_runner.py` - job-based runner (one unit = dataset × regime ×
  model × seed × fold; resume by skipping existing outputs, per-unit hard
  timeout; foundation models on GPU, tree controls pinned to CPU).
- `runner/stats.py` - aggregation and statistics (`stats/*.json`).
- `runner/decision.py` - conformalisation and the acceptance layer.
- `runner/review_gate.py` + `runner/gate_config.yaml` - the pre-freeze gate.
- `runner/make_figures.py`, `runner/make_fig01_framework.py` - all figures.
- `data/datasets.yaml` - dataset registry; `data/ebw_real_72.csv` (vendored).
- `fetch_data.py` - pulls the external GMAW datasets on a networked machine.
- `RUN_ORDER.md`, `experiments/exp_plan.md` - run order and the experiment plan.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py --check          # place GMAW CSVs per data/datasets.yaml
export EBW_DEVICE=cuda EBW_TREE_CPU=1 # hybrid placement: TFMs on GPU, trees on CPU
python runner/bench_runner.py --datasets ebw,gmaw_e1,gmaw_e2 \
  --protocols full,fewshot50,fewshot25,augment \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0,1,2,3,4,5,6,7,8,9 --outdir runs/final --no-resume
python runner/stats.py       --in runs/final --out runs/final
python runner/review_gate.py runs/final --config runner/gate_config.yaml  # must PASS
python runner/decision.py    --in runs/final --out runs/final --dataset ebw --regime full
python runner/make_figures.py --in runs/final --out outputs/figures
```
TabPFN checkpoints are read from `~/.cache/tabpfn` (override via `TABPFN_*_CKPT`).
For a long run, start it inside `tmux new -s ebw_final` and detach.

## Reproducing the published results
The frozen run behind the manuscript comprises **4,760 units** (EBW: 15
leave-one-regime-out folds × 4 regimes × 7 models × 10 seeds; each GMAW dataset:
4 regimes × 7 models × 10 seed-indexed group splits). Every unit terminated and
was recorded: 4,036 computed, 510 are by-design skips (the augmentation regime is
defined for the classical controls only), and 214 are recorded NGBoost fitting
failures concentrated in the few-shot regimes. The run was executed from scratch
with `--no-resume` under the hybrid device configuration above and passed the
gate before any number was frozen; the commands are exactly those of the
Quickstart. `results/stats/` holds the aggregated statistics of that run,
`results/manifest.jsonl` its unit-level log, and `figures/` the figures as
published.

## Data
- EBW (72 cross-sections, 18 coupons, 15 schedules): included here; the dataset
  first appeared in the authors' earlier benchmark, archived at Zenodo
  (https://doi.org/10.5281/zenodo.21204586).
- GMAW (external validity): Mendeley Data, https://doi.org/10.17632/2nyjpb89bf.1
  (CC BY 4.0).

## Citing
If you use this code or the EBW dataset, please cite the manuscript and this
archive:

> Kurashkin, S.O.; Tynchenko, V.S.; Borodulin, A.S.; Nelyub, V.A.; Kalutsky,
> N.O.; Connie, T. In-context tabular foundation models for calibrated
> weld-geometry prediction and acceptance decisions in electron-beam welding.
> Under review, 2026.

> Kurashkin, S.O.; Tynchenko, V.S.; Borodulin, A.S.; Nelyub, V.A.; Kalutsky,
> N.O.; Connie, T. Tabular Foundation Models for Electron-Beam Welding Parameter
> Prediction: Code, Data and Frozen Results. Zenodo, 2026.
> https://doi.org/10.5281/zenodo.21277220

Repository: https://github.com/digger32/Tabular-foundation-models-for-electron-beam-welding-parameter-prediction
