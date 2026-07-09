# RUN_ORDER — A (EBW applied TFM bench → AI MDPI)

Inference-only, GPU-bound; runs on the idle GPU while AR-02 holds the CPU.
EBW runs immediately (data vendored); public welding sets are the external-validity leg.

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

## 4. Smoke (EBW only, 3 seeds, cheap — verifies the whole path)
```bash
tmux new -s ebwsmoke -d "python runner/bench_runner.py --datasets ebw \
  --protocols full --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0,1,2 --outdir runs/smoke --timeout-s 1800 ; echo DONE"
```

## 5. Dev run (single GPU box, no sharding)
```bash
tmux new -s ebwdev -d "python runner/bench_runner.py \
  --datasets ebw,gmaw_e1,gmaw_e2 \
  --protocols full,fewshot50,fewshot25,augment \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0,1,2,3,4 --outdir runs/dev --timeout-s 1800 ; echo DONE"
```

## 6. Final pass (single A100, resume disabled)
```bash
tmux new -s ebwfinal -d "python runner/bench_runner.py \
  --datasets ebw,gmaw_e1,gmaw_e2 \
  --protocols full,fewshot50,fewshot25,augment \
  --models tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --outdir runs/final --no-resume --timeout-s 1800 ; echo DONE"
```

## 7. Statistics -> gate -> figures
```bash
python runner/stats.py --in runs/final --out runs/final
python runner/review_gate.py runs/final --config runner/gate_config.yaml
# the gate must print GATE PASSED; it asserts A1 clean run (no resume, no skipped units
# beyond the by-design ones), B1 external validity, C1 calibration recorded,
# D1 optimism gap present for every tuned control, and E1 statistics outputs.
python runner/make_figures.py --in runs/final --out outputs/figures  # 13 figures (PDF+PNG)
```

