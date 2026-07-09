# Experiment plan — A (EBW applied TFM benchmark)

Applied multi-output regression benchmark. Every claim maps to an experiment and a
display item. Reproducibility legs baked in (the reviewer concern).

## Experiment matrix

| ID | Claim it supports | Figure/Table | Datasets / units | Baselines | Metrics | Stat test | RP items |
|----|-------------------|--------------|------------------|-----------|---------|-----------|----------|
| E1 | C1 TFMs match/beat the best tuned classical on small-n EBW | Fig 1, Table 1 | ebw × full × {tabpfn v2/v25/v3, mitra} vs tuned {catboost,xgb,ngb,mlp} × 10 seeds | tuned classical | RMSE/MAE/R² per output (depth,width) | Friedman+Nemenyi across models; Wilcoxon-Holm pairwise | 1,2,3,6,9,11 |
| E2 | C2 Few-shot curve: TFMs hold as context shrinks | Fig 2 | ebw × {full,fewshot50,fewshot25} × TFMs + best control × 10 | best tuned control | RMSE vs context size | trend + CI | 1,4,5 |
| E3 | C3 TFMs (no augmentation) vs the augmentation pipeline | Fig 3 | ebw × {full, augment} × {TFMs; classical+CTGAN/TVAE} × 10 | classical+augment | RMSE/MAE + calibration | Wilcoxon-Holm | 1,2,6,11 |
| E4 | C4 TFMs are calibrated for manufacturing use | Fig 4 | ebw × full × distributional {tabpfn*, mitra, ngb} × 10 | ngb | 80% PI coverage, interval-ECE, CRPS | coverage vs nominal | 4 |
| E5 | external validity | Fig A1 | {laser_bead, gmaw_bead} × full × TFMs vs tuned classical × 10 | tuned classical | RMSE/MAE/R² | replication | 2 |
| E-final | clean final pass | (all) | full matrix, from scratch, resume DISABLED, one box | — | — | — | 1 |

## Mandatory legs (do not omit)
- **External validity (E5).** The headline "TFM matches/beats tuned classical" and
  "TFM beats augmentation" claims are replicated on 2 public welding datasets. Gate B1.
- **HPO honesty (D1).** Tuned classical controls use a matched/nested protocol; the
  optimism gap (inner-CV minus held-out) is recorded per control. This directly
  addresses the prior spurious-HPO-gain ding. Gate D1.
- **Calibration (E4).** TFMs are distributional; report interval coverage + ECE + CRPS,
  not only point error. Gate C1.
- **Final pass (E-final).** From scratch, resume DISABLED, one pinned env, one box. Gate A1.
- **Determinism note.** TFMs predict by a deterministic forward pass (no training);
  the only stochasticity is permutation-ensembling, fixed by seed — state this as the
  reproducibility strength.
- **Split integrity.** No tuning on test; if EBW rows have run/plate structure, use a
  grouped split (choose at Stage A from the data).

## Claim → experiment → display item map
| Claim | Experiment(s) | Display item | Distinct from IJAMT because |
|-------|---------------|--------------|-----------------------------|
| C1 TFM matches/beats tuned classical out-of-the-box | E1, E5 | Fig 1, Table 1 | object is pretrained in-context models, not a regressor+optimiser sweep |
| C2 few-shot robustness | E2 | Fig 2 | small-n context regime, not HPO |
| C3 TFM vs augmentation | E3 | Fig 3 | augmentation is the control, not the method |
| C4 calibration | E4 | Fig 4 | distributional eval, absent from the classical study |

## Protocol note (distinct from IJAMT)
IJAMT used 3-fold CV to keep a 41x12x4x4 grid tractable and did NOT include any
tabular foundation model (TabPFN is named there only as future work). A puts the foundation models in as the object and uses a **repeated held-out** split
(seed indexes the partition, models paired per seed). EBW is static -> random split;
gmaw_e1/e2 are dynamic time series -> **group split by run** (+ GroupKFold tuning) to
remove temporal-neighbour leakage. This is
both the novelty and a protocol upgrade the small grid can afford.

## Compute budget
Units ≈ datasets(3) × regimes(≤4) × models(8) × seeds(10) ≈ 960, each a cheap
inference (TFM) or a small fit (classical, n small). A few GPU-hours; dev used A100, final on A100. TFMs + xgb/catboost on the A100 GPU; NGBoost (CPU-only) and the nested-CV
tuning run on the A100 box's CPU. Single box, no sharding. Augment regime applies to classical controls only (TFMs skip it).
