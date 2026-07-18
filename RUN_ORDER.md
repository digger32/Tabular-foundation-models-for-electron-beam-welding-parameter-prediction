# RUN_ORDER — A (EBW applied TFM bench → Springer JIM)

Inference-only, GPU-bound; runs on the idle GPU while AR-02 holds the CPU.
EBW runs immediately (data vendored); public welding sets are the external-validity leg.


## 0a. READ FIRST — what changed (2026-07)

Two corrections invalidate every EBW number from the AI/MDPI submission:

1. **The EBW split was leaking.** The 72 rows are 18 coupons x 4 metallographic
   cross-sections over 15 distinct settings, not 72 independent experiments. Under
   the old random 80/20 split **14 of 14 test rows, on every seed, had their exact
   input vector in the training set**. The reported errors (Depth RMSE 0.075 mm) sat
   at the within-setting measurement floor (sd 0.041 mm), not at the scale of the
   signal (between-setting sd 0.222 mm). EBW now uses `cv: logo` — leave-one-regime-out,
   15 folds. GMAW was always grouped by run and is unaffected.
2. **The quantile grid was never persisted.** It was computed, consumed inside
   `metrics()` and discarded, so no interval question could be answered without a
   re-run. Payload v2 persists `pred_q`; the acceptance-decision layer and any future
   conformal work are now post-hoc.

New gate items: **F1** (no test input vector occurs in its own training set) and
**G1** (every dataset declares `group_by`, or `group_by: null` on purpose).

The grid is no longer 840 units: EBW alone is 15 folds x 4 regimes x 7 models x
10 seeds = 4200. **Run the micro slice (section 4-6) and read the real per-unit `wall_s`
before committing to the full pass.**

## 0. Environment
```bash
python3 -m venv ~/ebw_tfm && source ~/ebw_tfm/bin/activate
pip install -r requirements.txt --break-system-packages
```

## 1. Device (checkpoints are already baked into the code)
The regressor checkpoints in ~/.cache/tabpfn are wired in bench_runner.py:
  tabpfn_v2  -> tabpfn-v2-regressor.ckpt
  tabpfn_v25 -> tabpfn-v2.5-regressor-v2.5_default.ckpt   (synthetic-only default)
  tabpfn_v3  -> tabpfn-v3-regressor-v3_default.ckpt
Override only if the cache moves: export TABPFN_CACHE_DIR=/new/dir (or per-model
TABPFN_V2_CKPT / TABPFN_V25_CKPT / TABPFN_V3_CKPT).
```bash
export EBW_DEVICE=cuda    # everything that can -> GPU (TFMs + xgb + catboost), since the
                         # CPU is busy with AR-02. NGBoost is CPU-only (n_jobs=1). Force
                         # trees to CPU with EBW_TREE_CPU=1 if ever needed.
```

## 2. Fetch the public welding datasets (external validity)
```bash
python fetch_data.py            # downloads where a direct URL is set, else prints sources
# then edit data/datasets.yaml: confirm path/inputs/targets for each fetched set
python fetch_data.py --check    # confirm >=1 public set is ready
```

## 3. Currency re-check (Stage C — before any final run)
```bash
pip freeze | grep -E "tabpfn|ngboost|xgboost|catboost|sdv|scikit-posthocs" >> requirements.lock
# confirm: TabPFN predict(output_type="quantiles"), model_path kwarg; SDV metadata API.
```

## 4-6. Smoke -> micro -> final

House pattern: create the session FIRST, activate the venv INSIDE it, then run.
Do NOT pass the command to `tmux new -d "..."` — that spawns a fresh non-interactive
shell which sources neither ~/.bashrc nor the venv, so `python` does not exist
there (Ubuntu ships python3 only; `python` comes from the venv). That is a silent
failure: the command dies in milliseconds and tmux closes the session with it.

### Smoke — cheap path check
```bash
cd ~/Documents/AI-EBW
tmux new -s ebwsmoke
source ~/ebw_tfm/bin/activate
python runner/bench_runner.py --datasets ebw --protocols full \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0 --folds 0 --outdir runs/smoke --timeout-s 1800
# Ctrl-b d to detach
```

### Device policy — MEASURED, not assumed

Two micro slices settled it. The families want opposite hardware:

| model | GPU s/unit | CPU s/unit | verdict |
|---|---|---|---|
| tabpfn_v2 | 8.6 | 92.6 | GPU (11x) |
| tabpfn_v25 | 9.3 | 129.0 | GPU (14x) |
| tabpfn_v3 | 13.4 | 300.1 | GPU (22x) |
| catboost | 237.4 | 9.7 | **CPU (24x)** |
| xgb | 38.8 | 1593.5 unpinned / see below | CPU once threads are pinned |
| ngb | 51.1 | 57.4 | CPU either way (no GPU build) |
| mlp | 2.3 | 2.5 | CPU either way |

A tree fit on 64 rows is microseconds of arithmetic; on GPU it is device init, host
copies and per-iteration sync. Unpinned on CPU it is OpenMP thread churn over every
core of the node. Both are overhead, in opposite directions.

**Hybrid is the default:**
```bash
export EBW_DEVICE=cuda EBW_TREE_CPU=1     # TFMs -> GPU, tree controls -> CPU
export EBW_TREE_THREADS=1                 # pin; raise only if a slice says it helps
```
This is ONE configuration: the manifest records `device` and `tree_cpu` per unit and
gate A1 fails a run stitched across configurations. Note CatBoost GPU and CPU are
not numerically identical, so a device change is a numbers change — fine here, since
the leak fix already invalidates the old EBW numbers, but never mix within a run.

### Micro — MANDATORY timing slice
One fold (`--folds 0`), one seed, real data, and BOTH cost regimes. `full` and
`augment` are not comparable: augment fits a CTGAN/TVAE on the training fold and
dominates the grid. Pricing all four regimes at the `full` rate underestimates the
run badly. fewshot50/25 are strictly cheaper than full and are priced at the full
rate as an upper bound.

A timing slice is a MEASUREMENT: always start from an empty runs/micro, never
resume into it — stale units would be skipped while the manifest grew, and the
estimate would be wrong invisibly.

```bash
cd ~/Documents/AI-EBW
rm -rf runs/micro
tmux new -s ebwmicro
source ~/ebw_tfm/bin/activate
python runner/bench_runner.py --datasets ebw --protocols full,augment \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0 --folds 0 --outdir runs/micro --timeout-s 1800
python runner/bench_runner.py --datasets gmaw_e1 --protocols full,augment \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0 --folds 0 --outdir runs/micro --timeout-s 1800
# Ctrl-b d
```
~14 ebw units + ~14 gmaw units.

### Estimate — read the slice, price the grid
```bash
python3 runner/estimate_grid.py --micro runs/micro
```
Prints measured s/unit per (dataset, model) for full and augment, unit counts, and
the extrapolated total. Anything the slice did not measure is printed as MISSING,
never guessed. Read that number BEFORE committing to the final pass.

### Final — the clean run
```bash
cd ~/Documents/AI-EBW
tmux new -s ebwfinal
source ~/ebw_tfm/bin/activate
python runner/bench_runner.py \
  --datasets ebw,gmaw_e1,gmaw_e2 \
  --protocols full,fewshot50,fewshot25,augment \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --outdir runs/final --no-resume --timeout-s 1800
# Ctrl-b d
```
The runner refuses `--no-resume` together with `--folds`: a sliced final pass would
pass gate A1 while missing units.

## 7. Statistics -> decision layer -> gate -> figures

Short and CPU-only; no tmux needed.
```bash
cd ~/Documents/AI-EBW && source ~/ebw_tfm/bin/activate

python runner/stats.py --in runs/final --out runs/final

# Acceptance layer. Both conformal constructions are POST-HOC over the persisted
# quantile grid: split costs no GPU, it is a second read of the same run.
for tier in production design ratio; do
  python runner/decision.py --in runs/final --out runs/final --dataset ebw \
      --alpha 0.05 --tier $tier --conformal cvplus
done
python runner/decision.py --in runs/final --out runs/final --dataset ebw \
    --alpha 0.05 --tier production --conformal split --cal-frac 0.2
python runner/decision.py --in runs/final --out runs/final --dataset ebw \
    --alpha 0.05 --tier production --conformal cvplus --sweep

python runner/review_gate.py runs/final --config runner/gate_config.yaml
python runner/make_figures.py --in runs/final --out outputs/figures
```
The gate must print GATE PASSED: A1 clean run / B1 external validity / C1
calibration / D1 optimism gap / E1 stats / F1 no input leakage / G1 splitting
declared.

### Changing the tolerance or the risk costs no GPU
```bash
python runner/decision.py --in runs/final --out runs/final --dataset ebw \
    --alpha 0.10 --tier production --sweep
```
decision.py only reads `pred_q`. That is the whole point of payload v2.


```bash
python runner/stats.py --in runs/final --out runs/final
python runner/review_gate.py runs/final --config runner/gate_config.yaml
python runner/make_figures.py --in runs/final --out outputs/figures  # Fig 1-4 + Fig A1 (PDF+PNG)
```

Note: HPO-honesty (gate D1) requires the tuned controls' optimism_gap to be recorded.
Wire a nested-CV tuning step for {catboost,xgb,ngb,mlp} that writes optimism_gap into
each unit's metrics before the final pass (see BUILD.md).
