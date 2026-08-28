# Tabular foundation models for electron-beam welding parameter prediction

Reproducibility package for *Risk-controlled acceptance decisions from calibrated
predictions under grouped distribution shift, with electron-beam weld qualification as a
case study* (Kurashkin, Tynchenko, Borodulin, Nelyub, Kalutsky, Kukartsev, Connie; under
review at *Reliability Engineering & System Safety*).

The paper turns a calibrated predictive interval into an accept, reject or abstain
qualification decision against an engineering tolerance at a controlled consumer's risk,
so that some coupons can skip destructive metallographic sectioning. Eight predictors are
compared, four tolerance criteria are evaluated, and three conformal constructions are set
side by side; the headline finding is that on small grouped data the choice of conformal
construction, not the choice of predictor, decides whether a decision layer functions at
all.

## What changed since v2.0

v2.0 accompanied an earlier version of this work aimed at a different journal. It is
superseded, and the numbers in it do **not** correspond to the present manuscript.

| | v2.0 | this release |
|---|---|---|
| predictors | 7 | **9** (TabICLv2 and AutoGluon added) |
| omnibus | 30 blocks, chi-square 78.59, no block definition recorded | **35 blocks, chi-square 67.2**, seeds averaged within fold, all three datasets complete |
| conformal constructions | cross-validation-plus, split | **cross-validation-plus, split, non-exchangeable (nexCP)** |
| acceptance-decision results | not included | **11 `decision_*.json` files, three constructions, three tiers, risk and band sweeps** |
| AutoGluon | absent | full grid on all three datasets; the sharpest predictor on the electron-beam data and the worst calibrated |
| figure names | `fig01`--`fig12`, `figB1`, `figB2` | `fig05`--`fig12` (article), `figS1`--`figS5` (supplement) |

## Layout

```
data/        ebw_real_72.csv   the electron-beam campaign, 72 cross-sections
             datasets.yaml     dataset and split declarations
runner/      bench_runner.py   one job per dataset x regime x model x fold x seed
             stats.py          per-target metrics, omnibus, post-hoc, calibration
             decision.py       conformal constructions and the acceptance layer
             coverage_report.py  coverage by construction
             make_figures.py   every data figure in the article and the supplement
             review_gate.py    the pre-freeze gate (gate_config.yaml)
results/stats/                 the frozen statistics behind every table and figure
```

## Reproducing the reported numbers

```bash
pip install -r requirements.txt
python fetch_data.py                       # retrieves the two public GMAW datasets
python runner/bench_runner.py --out runs/final --no-resume
python runner/stats.py      --in runs/final --out runs/final
python runner/decision.py   --in runs/final --out runs/final --dataset ebw \
        --alpha 0.2 --tier production --conformal nexcp --sweep-alpha --data-dir data
python runner/coverage_report.py --in runs/final
python runner/review_gate.py runs/final --config runner/gate_config.yaml
python runner/make_figures.py --in runs/final --out outputs/figures
```

The values in `results/stats/` are the frozen ones the article reports. `omnibus.json`
must show `n_blocks` 35 and a `block_definition` of "fold for cv=logo (seeds averaged
within fold)": blocking on (fold x seed) treats deterministic seed copies as independent
observations and inflates the test.

## Scope of this release

Included: the electron-beam dataset, the split declarations, the full runner, and the
frozen aggregate statistics, which is everything needed to check the reported tables,
figures and decisions.

Not included: the per-unit prediction files (roughly 4,700 JSON records carrying `y_test`,
`pred_mean` and the full quantile grid `pred_q` of every unit). They are large, and the
aggregate statistics above are derived from them. Anyone wishing to re-derive the
intervals from raw predictions can regenerate them with the command sequence above, or
request the archive from the corresponding author.

The two gas-metal-arc-welding datasets are public and are fetched by `fetch_data.py`
rather than redistributed here.

## Citation

Cite the article. If the software itself needs citing, use the Zenodo record for this
release.

## Licence

See `LICENSE`.
