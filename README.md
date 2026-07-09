# Tabular foundation models for electron-beam welding parameter prediction

Reproducibility package for the study *Out-of-the-Box Without Training: Tabular
Foundation Models for Electron-Beam Welding Parameter Prediction in the
Small-Sample Regime* (target: *AI*, MDPI).

Benchmarks TabPFN v2 / v2.5 / v3 (in-context, no training) against nested-CV-tuned
CatBoost, XGBoost, NGBoost and an MLP, plus a CTGAN/TVAE augmentation arm, on a
72-row electron-beam welding dataset and two public gas-metal-arc-welding datasets
(group-split, leakage-controlled). Reports per-target RMSE/MAE/R2, interval
calibration, a few-shot curve, and Friedman/Nemenyi + Wilcoxon-Holm statistics,
behind a review-proofing gate.

## Layout
- `runner/bench_runner.py` - job-based runner (unit = dataset x regime x model x seed;
  resume, per-unit hard timeout; TFMs on GPU, tree controls configurable).
- `runner/stats.py` - aggregation + statistics.
- `runner/review_gate.py` + `runner/gate_config.yaml` - the pre-freeze gate.
- `runner/make_figures.py` - figures from `stats/*.json`.
- `data/datasets.yaml` - dataset registry; `data/ebw_real_72.csv` (vendored).
- `fetch_data.py` - pull the external GMAW datasets on a networked machine.
- `RUN_ORDER.md`, `BUILD.md`, `CODE_REVIEW.md`, `experiments/exp_plan.md`.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py --check          # place GMAW CSVs per data/datasets.yaml
export EBW_DEVICE=cuda
python runner/bench_runner.py --datasets ebw,gmaw_e1,gmaw_e2 \
  --protocols full,fewshot50,fewshot25,augment \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0,1,2,3,4,5,6,7,8,9 --outdir runs/final --no-resume
python runner/stats.py --in runs/final --out runs/final
python runner/review_gate.py runs/final --config runner/gate_config.yaml
python runner/make_figures.py --in runs/final --out outputs/figures
```
TabPFN checkpoints are read from `~/.cache/tabpfn` (override via `TABPFN_*_CKPT`).

## Data
- EBW (72 rows): included here; archived in Zenodo 10.5281/zenodo.21204586.
- GMAW (external validity): Mendeley Data 10.17632/2nyjpb89bf.1 (CC BY 4.0).

## Citation
See `CITATION` / the manuscript once published.


## Reproducing the published results
The frozen run behind the manuscript (840 units: 3 datasets x 4 regimes x 7 models x 10 seeds)
was produced with a single clean pass:

```bash
export EBW_DEVICE=cuda
tmux new -s ebw_final
python runner/bench_runner.py --datasets ebw,gmaw_e1,gmaw_e2 \
  --protocols full,fewshot50,fewshot25,augment \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0,1,2,3,4,5,6,7,8,9 --outdir runs/final --no-resume
python runner/stats.py       --in runs/final --out runs/final
python runner/review_gate.py runs/final --config runner/gate_config.yaml   # must PASS
python runner/make_figures.py --in runs/final --out outputs/figures
```
`results/stats/` holds the aggregated statistics of that run and `results/manifest.jsonl`
its unit-level log; `figures/` holds the figures as published.

## Citing
If you use this code or the EBW dataset, please cite the manuscript (AI, MDPI) and this archive.
