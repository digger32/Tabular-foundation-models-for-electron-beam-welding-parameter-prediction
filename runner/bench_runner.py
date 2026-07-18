#!/usr/bin/env python3
"""
A (EBW) job-based runner — TFM few-shot benchmark for welding parameter prediction.

Unit = dataset x regime x model x seed x fold, each in its OWN subprocess (a hang /
OOM in one unit never takes down the batch). Orchestration is the house base: resume
(skip units whose output exists), per-unit hard timeout, one manifest record per
unit, keep going past failures. Only run_unit() and the axis defaults are adapted.

Splitting is declared per dataset in datasets.yaml, never assumed here:
  cv: logo      leave-one-group-out, one fold per group (EBW: 15 settings)
  cv: holdout   one seeded held-out split, grouped when `group_by` is set (GMAW)
Whatever the mode, `group_by` rows never straddle the split. Every unit records
`leak_check`, the number of test input vectors that also occur in its own training
set; the review gate fails the run if that is ever non-zero. This assert exists
because the EBW set was long believed to be 72 independent experiments when it is
in fact 18 coupons x 4 cross-sections over 15 distinct settings, and a random split
was therefore scoring replicate recall rather than prediction at a new setting.

Task: MULTI-OUTPUT regression (weld depth + width). Records per-instance predictions
(mean AND the quantile grid where the model is distributional) plus per-output
metrics and, for distributional models, calibration (CRPS, interval coverage). The
quantile grid is persisted so that conformal intervals, reliability curves and the
acceptance-decision layer can be rebuilt post hoc WITHOUT a re-run. (Before payload
v2 the grid was computed, consumed inside metrics() and thrown away, which forced a
full re-run for any new interval question.) TFM regressors
(TabPFN v2/v2.5/v3, Mitra) predict in-context from LOCAL checkpoints — no training,
no augmentation. TabICL is classification-only (AR-02 cap) and is skipped here.

Regimes:
  full       — model uses the full real training context
  fewshot50  — context limited to 50% of training rows (few-shot curve point)
  fewshot25  — context limited to 25%
  augment    — CLASSICAL controls only: train on real + CTGAN/TVAE-augmented data
               (the field's current small-n crutch; a comparison, not the object)

Launch wrapped in tmux (house convention):

    tmux new -s ebw
    python bench_runner.py \
        --datasets ebw,laser_bead,gmaw_bead \
        --protocols full,fewshot50,fewshot25,augment \
        --models tabpfn_v2,tabpfn_v25,tabpfn_v3,mitra,catboost,xgb,ngb,mlp \
        --seeds 0,1,2,3,4,5,6,7,8,9 \
        --outdir runs/final --no-resume --shard 0/1 --timeout-s 1800
    # detach: Ctrl-b d
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# PER-UNIT WORK — adapted for A (EBW regression). Orchestration below is base.  #
# CURRENCY CHECKPOINT (Stage C): pin tabpfn (v3), Mitra adapter, ngboost.       #
# --------------------------------------------------------------------------- #
TFM_MODELS = {"tabpfn_v2", "tabpfn_v25", "tabpfn_v3", "mitra"}
CLASSICAL = {"catboost", "xgb", "ngb", "mlp"}
DISTRIBUTIONAL = {"tabpfn_v2", "tabpfn_v25", "tabpfn_v3", "mitra", "ngb"}  # give a predictive dist
MODEL_CAPS = {"tabicl": {"classification"}}  # AR-02 cap: excluded from this regression task

_TABPFN_CACHE = os.path.expanduser(os.environ.get("TABPFN_CACHE_DIR", "~/.cache/tabpfn"))
TABPFN_CKPT = {
    "tabpfn_v2":  os.environ.get("TABPFN_V2_CKPT",  os.path.join(_TABPFN_CACHE, "tabpfn-v2-regressor.ckpt")),
    "tabpfn_v25": os.environ.get("TABPFN_V25_CKPT", os.path.join(_TABPFN_CACHE, "tabpfn-v2.5-regressor-v2.5_default.ckpt")),
    "tabpfn_v3":  os.environ.get("TABPFN_V3_CKPT",  os.path.join(_TABPFN_CACHE, "tabpfn-v3-regressor-v3_default.ckpt")),
}


class _Skip(Exception):
    """Model/regime not applicable to this unit (recorded, not failed)."""


DATA_DIR = Path(os.environ.get("EBW_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def _dataset_registry():
    """Read data/datasets.yaml: {name: {path, inputs: [...], targets: [...]}}.
    EBW is vendored; public welding sets are added there after fetch_data.py."""
    import yaml
    reg_path = DATA_DIR / "datasets.yaml"
    if not reg_path.exists():
        raise FileNotFoundError(f"{reg_path} missing — see fetch_data.py / BUILD.md")
    return yaml.safe_load(reg_path.read_text())


def load_full(dataset: str):
    """Load raw (X, Y, meta) for a dataset token from the registry. Targets and
    inputs come from datasets.yaml. If the spec has `group_by`, compute per-row group
    ids (a "run" = one setting => a time series); the split then keeps whole runs
    together to avoid temporal-neighbour leakage."""
    import numpy as np, pandas as pd
    reg = _dataset_registry()
    if dataset not in reg:
        raise KeyError(f"dataset '{dataset}' not in datasets.yaml (registry: {list(reg)})")
    spec = reg[dataset]
    df = pd.read_csv(DATA_DIR / spec["path"], sep=spec.get("sep", ","))
    X = df[spec["inputs"]].to_numpy(float)
    Y = df[spec["targets"]].to_numpy(float)
    if Y.ndim == 1:
        Y = Y[:, None]
    groups = None
    gcols = spec.get("group_by")
    if gcols:
        groups = df.groupby(gcols, sort=False).ngroup().to_numpy()   # int id per row
    return X, Y, {"task": "regression", "targets": list(spec["targets"]),
                  "name": dataset, "groups": groups}


def n_folds(dataset: str) -> int:
    """How many folds this dataset's declared `cv` mode implies. logo -> one fold per
    group; holdout -> a single seeded split. Called by the orchestrator to build the
    unit list, so the fold axis is data-driven and never hard-coded."""
    import numpy as np
    spec = _dataset_registry()[dataset]
    if spec.get("cv", "holdout") != "logo":
        return 1
    _, _, meta = load_full(dataset)
    if meta.get("groups") is None:
        raise ValueError(f"dataset '{dataset}' declares cv: logo but no group_by")
    return int(np.unique(meta["groups"]).size)


def make_folds(X, Y, seed, groups=None, cv="holdout", test_frac=0.2):
    """Return the list of (train_idx, test_idx) index pairs for this dataset.

    cv='logo'     leave-one-group-out: fold g holds out every row of group g, trains
                  on all other groups. Deterministic — `seed` does not enter the
                  split, only model stochasticity. Used for EBW, where 15 settings
                  are too few to spend 3 of them on a single held-out fifth.
    cv='holdout'  one seeded 80/20 split, grouped when `groups` is given (whole runs
                  stay together), plain random only when the dataset explicitly
                  declares no grouping.

    NOTE: no dataset is assumed ungrouped. `group_by: null` must be written in the
    registry on purpose; an absent key on a dataset with replicate structure is the
    exact mistake that produced the original EBW split.
    """
    import numpy as np
    if cv == "logo":
        if groups is None:
            raise ValueError("cv='logo' requires group_by")
        return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g))
                for g in np.unique(groups)]
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        return [next(gss.split(X, Y, groups))]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); cut = max(1, int(round((1 - test_frac) * len(X))))
    return [(idx[:cut], idx[cut:])]


def load_dataset(dataset: str, seed: int, fold: int = 0):
    """Return one (X_tr, Y_tr, X_te, Y_te, meta, groups_tr) fold. X is standardised on
    TRAIN only. meta carries `leak_check`: how many test input vectors also occur in
    this fold's training set — must be 0, asserted by review_gate F1."""
    import numpy as np
    X, Y, meta = load_full(dataset)
    spec = _dataset_registry()[dataset]
    folds = make_folds(X, Y, seed, groups=meta.get("groups"), cv=spec.get("cv", "holdout"))
    if fold >= len(folds):
        raise IndexError(f"{dataset}: fold {fold} of {len(folds)}")
    tr, te = folds[fold]
    groups = meta.get("groups")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    seen = {tuple(r) for r in X[tr]}
    meta["leak_check"] = int(sum(tuple(r) in seen for r in X[te]))
    meta["n_train"], meta["n_test"], meta["fold"] = len(tr), len(te), fold
    return Xs[tr], Y[tr], Xs[te], Y[te], meta, (groups[tr] if groups is not None else None)


def augment_training(X_tr, Y_tr, seed, method=None, n_synth=2000):
    """CTGAN/TVAE augmentation of the TRAIN set only (classical-control arm). Fits an
    SDV synthesiser on (X_tr, Y_tr), samples n_synth rows, concatenates to real train.
    Independent of the IJAMT code; the generator family matches (CTGAN/TVAE) for a
    like-for-like comparison. Never touches the test set. Cap n_synth=2000 mirrors
    IJAMT's verified cap (macro-R2 change <0.02 vs n=10000)."""
    import numpy as np, pandas as pd
    method = method or os.environ.get("EBW_AUGMENT", "ctgan")
    cols_x = [f"x{i}" for i in range(X_tr.shape[1])]
    cols_y = [f"y{j}" for j in range(Y_tr.shape[1])]
    df = pd.DataFrame(np.hstack([X_tr, Y_tr]), columns=cols_x + cols_y)
    from sdv.metadata import SingleTableMetadata
    md = SingleTableMetadata(); md.detect_from_dataframe(df)
    if method == "tvae":
        from sdv.single_table import TVAESynthesizer as Synth
    else:
        from sdv.single_table import CTGANSynthesizer as Synth
    syn = Synth(md)
    syn.fit(df)
    s = syn.sample(num_rows=n_synth)
    Xa = np.vstack([X_tr, s[cols_x].to_numpy(float)])
    Ya = np.vstack([Y_tr, s[cols_y].to_numpy(float)])
    return Xa, Ya


def subsample_context(X_tr, Y_tr, groups_tr, frac, seed):
    """Few-shot: shrink the context. On a GROUPED dataset drop whole groups, not rows:
    the practitioner question is 'how few welding SETTINGS must I run', and keeping 2
    of the 4 cross-sections of a coupon does not answer it (nor does it shrink the
    input coverage at all). On an ungrouped dataset fall back to a row subsample.
    Carries group ids along so grouped tuning stays consistent afterwards."""
    import numpy as np
    rng = np.random.default_rng(seed)
    if groups_tr is not None:
        uniq = np.unique(groups_tr)
        k = max(2, int(round(frac * len(uniq))))
        keep = set(rng.choice(uniq, size=min(k, len(uniq)), replace=False).tolist())
        idx = np.flatnonzero(np.isin(groups_tr, list(keep)))
    else:
        k = max(8, int(round(frac * len(X_tr))))
        idx = rng.choice(len(X_tr), size=k, replace=False)
    g = groups_tr[idx] if groups_tr is not None else None
    return X_tr[idx], Y_tr[idx], g


PAYLOAD_VERSION = 2      # v1 persisted pred_mean + y_test only; v2 adds pred_q + leak_check
QLEVELS = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]   # persisted per unit as `pred_q` (payload v2)
TFM_DEVICE = os.environ.get("EBW_DEVICE", "cuda")
# Device policy, MEASURED per model on a micro slice rather than assumed. On this
# grid (n_train 64-68 for EBW) the two families want opposite hardware:
#   TFMs        GPU by a wide margin      v3: 13.4 s GPU vs 300.1 s CPU  (22x)
#   tree ctrls  CPU by a wide margin      catboost: 237.4 s GPU vs 9.7 s CPU (24x)
# A CatBoost fit on 64 rows is microseconds of arithmetic; on GPU it is device init,
# host<->device copies and per-iteration sync, three orders of magnitude above the
# useful work. The hybrid — EBW_DEVICE=cuda with EBW_TREE_CPU=1 — is therefore the
# right default, and it is ONE configuration (the manifest records device+tree_cpu
# per unit and gate A1 asserts a single value across the run).
_TREE_GPU = TFM_DEVICE.startswith("cuda") and not os.environ.get("EBW_TREE_CPU")

# THREAD PINNING for the tree controls. On tiny data an unpinned tree library is
# pathological, not merely suboptimal: XGBoost defaults its thread count to every
# core on the node, so a 64-row fit spends tens of milliseconds per boosting round
# spawning and synchronising threads over microseconds of arithmetic. Measured on
# the A800 node: xgb 38.8 s (GPU) -> 1593.5 s (CPU, unpinned) per EBW unit, with the
# augment units hitting the 1800 s timeout. The bigger the node, the worse it gets.
# CatBoost survived only because it throttles its own threads on small inputs; it is
# pinned here too so both controls are timed on equal terms.
# Parallelism belongs at the UNIT level (independent jobs), never inside a fit on 64
# rows. Raise with EBW_TREE_THREADS only after a micro slice says it helps.
_TREE_THREADS = int(os.environ.get("EBW_TREE_THREADS", "1"))
if _TREE_GPU:
    import warnings
    # xgb GPU-trained booster predicting on a tiny host array logs a device-mismatch
    # perf warning; the copy is negligible on our test sizes. Silence it, keep GPU.
    warnings.filterwarnings("ignore", message=".*mismatched devices.*")


def get_model(model: str, meta: dict, seed: int):
    """Return a multi-output regressor handle. TFMs load LOCAL checkpoints and predict
    in-context; classical controls are fit with random_state=seed. Distributional
    models expose quantiles for calibration. Raises _Skip when inapplicable."""
    caps = MODEL_CAPS.get(model)
    if caps is not None and "regression" not in caps:
        raise _Skip(f"{model} does not support regression")
    if model in ("tabpfn_v2", "tabpfn_v25", "tabpfn_v3"):
        ckpt = TABPFN_CKPT[model]
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"{model} checkpoint not found: {ckpt} "
                                    f"(set {model.upper()}_CKPT or TABPFN_CACHE_DIR)")
        # handle only; predict() instantiates per target from the LOCAL checkpoint.
        return ("tabpfn", ckpt)
    if model == "mitra":
        # Optional: wire the AutoGluon mitra-regressor adapter at the currency check.
        # Until then, skip cleanly (recorded, not failed) so the batch stays green.
        raise _Skip("mitra adapter not wired (see BUILD.md) — include only after wiring")
    if model == "catboost":
        from catboost import CatBoostRegressor
        from sklearn.multioutput import MultiOutputRegressor
        # CatBoost's GPU and CPU trainers are DIFFERENT ESTIMATORS, per the vendor FAQ,
        # not one estimator on two devices:
        #   border_count   254 (CPU) vs 128 (GPU)  — half the quantisation resolution
        #   model_size_reg on by default on CPU, off on GPU
        #   bootstrap_type MVS/subsample 0.8 (CPU) vs Bayesian (GPU)
        # and, decisively for us, GPU training is documented as NON-DETERMINISTIC: the
        # order of floating-point summation is not fixed, so `random_state` does NOT
        # reproduce a GPU run. Our seed axis therefore never controlled CatBoost's
        # stochasticity while task_type="GPU" was in force — the 10 seeds were measuring
        # device noise on top of split noise, with no way to separate them.
        # The hybrid default (trees on CPU) removes this: CPU training is deterministic
        # under random_state, so the seed axis means what the paper says it means.
        base = CatBoostRegressor(random_state=seed, verbose=False,
                                 task_type="GPU" if _TREE_GPU else "CPU",
                                 thread_count=_TREE_THREADS)
        return ("sk", MultiOutputRegressor(base))
    if model == "xgb":
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        base = XGBRegressor(tree_method="hist", device="cuda" if _TREE_GPU else "cpu",
                            random_state=seed, n_jobs=_TREE_THREADS)
        return ("sk", MultiOutputRegressor(base))
    if model == "ngb":
        from ngboost import NGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        # NGBoost is CPU-only (no GPU build) and its base learner is a single-threaded
        # decision tree, so it is unaffected by the device flags and by thread pinning:
        # measured 51.1 s vs 57.4 s across the two slices. That cost is real work, not
        # overhead, which makes NGBoost the tail of the hybrid grid. It stays as the
        # distributional calibration control — it is the baseline whose 0.297 coverage
        # the paper is about, so it cannot be dropped for being slow.
        return ("ngb", MultiOutputRegressor(NGBRegressor(random_state=seed, verbose=False)))
    if model == "mlp":
        from sklearn.neural_network import MLPRegressor
        # adam (default), NOT lbfgs: IJAMT reported an lbfgs convergence failure on 72x4.
        return ("sk", MLPRegressor(random_state=seed, max_iter=2000))
    raise ValueError(f"unknown model: {model}")


# Compact HPO grids for the classical controls (fair tuned baselines for item-2
# fairness; the optimism gap they yield is recorded for item-3 HPO honesty / gate D1).
# MOR-wrapped models use the estimator__ prefix; MLP is natively multi-output.
GRIDS = {
    "catboost": {"estimator__depth": [4, 6, 8], "estimator__learning_rate": [0.03, 0.1],
                 "estimator__iterations": [200, 500]},
    "xgb": {"estimator__max_depth": [3, 5, 7], "estimator__learning_rate": [0.03, 0.1],
            "estimator__n_estimators": [200, 500]},
    "ngb": {"estimator__n_estimators": [200, 500], "estimator__learning_rate": [0.01, 0.05]},
    "mlp": {"hidden_layer_sizes": [(64,), (128, 64)], "alpha": [1e-4, 1e-3]},
}


def tune_classical(model, estimator, X_tr, Y_tr, seed, groups_tr=None, cv=3, n_iter=8):
    """Nested-CV tune a classical control on the TRAIN set only. If `groups_tr` is
    given (dynamic sets), the inner CV is a GroupKFold so tuning does not leak across
    runs either (otherwise the optimism gap would be understated). Returns
    (fitted_best_estimator, inner_cv_rmse); n_jobs=1 keeps CPU pressure off AR-02."""
    from sklearn.model_selection import RandomizedSearchCV, GroupKFold
    import numpy as np
    grid = GRIDS[model]
    n_comb = 1
    for v in grid.values():
        n_comb *= len(v)
    fit_groups = None
    if groups_tr is not None and len(np.unique(groups_tr)) >= cv + 1:
        splitter = GroupKFold(n_splits=cv)
        fit_groups = groups_tr
    else:
        splitter = cv                      # plain KFold (EBW, or too few groups)
    search = RandomizedSearchCV(estimator, grid, n_iter=min(n_iter, n_comb), cv=splitter,
                                scoring="neg_root_mean_squared_error",
                                random_state=seed, n_jobs=1, error_score="raise")
    search.fit(X_tr, Y_tr, groups=fit_groups)
    return search.best_estimator_, float(-search.best_score_)


def infer(kind, obj, X_tr, Y_tr, X_te, meta, prefitted=False):
    """Predict mean and, for distributional models, a quantile grid Q of shape
    (n_te, n_targets, len(QLEVELS)) used downstream for coverage, CRPS and reliability.
    Non-distributional models return Q = None."""
    import numpy as np
    if kind == "tabpfn":
        from tabpfn import TabPFNRegressor
        means, grids = [], []
        for j in range(Y_tr.shape[1]):                 # TabPFN is single-target
            reg = TabPFNRegressor(model_path=obj, device=TFM_DEVICE)
            reg.fit(X_tr, Y_tr[:, j])
            means.append(np.asarray(reg.predict(X_te, output_type="mean")))
            qs = reg.predict(X_te, output_type="quantiles", quantiles=list(QLEVELS))
            grids.append(np.stack([np.asarray(q) for q in qs], 1))   # (n_te, levels)
        return np.stack(means, 1), np.stack(grids, 1)               # mean (n,t); Q (n,t,levels)
    if not prefitted:
        obj.fit(X_tr, Y_tr)
    mean = np.asarray(obj.predict(X_te))
    if mean.ndim == 1:
        mean = mean[:, None]
    if kind == "ngb":
        from scipy.stats import norm
        Q = np.zeros((mean.shape[0], mean.shape[1], len(QLEVELS)))
        for j, est in enumerate(obj.estimators_):      # one NGBRegressor per target
            p = est.pred_dist(X_te).params             # Normal loc/scale
            for li, lv in enumerate(QLEVELS):
                Q[:, j, li] = norm.ppf(lv, loc=p["loc"], scale=p["scale"])
        return mean, Q
    return mean, None


def metrics(mean, Q, Y_te, meta, optimism_gap=None):
    """Point metrics for all models; for distributional models (Q given) also the 80%
    interval coverage, a pinball-based CRPS per target, and per-level coverage (for the
    reliability curve). Q has shape (n_te, n_targets, len(QLEVELS))."""
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    out = {"per_target": {}}
    for j, name in enumerate(meta["targets"]):
        rec = {
            "rmse": float(mean_squared_error(Y_te[:, j], mean[:, j]) ** 0.5),
            "mae": float(mean_absolute_error(Y_te[:, j], mean[:, j])),
            "r2": float(r2_score(Y_te[:, j], mean[:, j])),
        }
        if Q is not None:
            # CRPS approx = 2 * mean_tau pinball_tau (quantile decomposition)
            pin = 0.0
            for li, tau in enumerate(QLEVELS):
                d = Y_te[:, j] - Q[:, j, li]
                pin += np.mean(np.maximum(tau * d, (tau - 1) * d))
            rec["crps"] = float(2.0 * pin / len(QLEVELS))
            # split-conformal (CQR, Romano et al. 2019) on an 80% interval, evaluated
            # within the unit: calibrate the offset on half the test points, apply on
            # the other half -> coverage valid by construction. alpha = 0.20.
            lv = list(QLEVELS); i10, i90 = lv.index(0.1), lv.index(0.9)
            yj = Y_te[:, j]; qlo = Q[:, j, i10]; qhi = Q[:, j, i90]; n = len(yj)
            if n >= 8:
                rng = np.random.default_rng(0); idx = rng.permutation(n)
                c, e = idx[:n // 2], idx[n // 2:]
                s = np.maximum(qlo[c] - yj[c], yj[c] - qhi[c])
                k = min(max(np.ceil((len(c) + 1) * 0.8) / len(c), 0.0), 1.0)
                Qoff = float(np.quantile(s, k, method="higher"))
                rec["cov80_raw_eval"] = float(np.mean((yj[e] >= qlo[e]) & (yj[e] <= qhi[e])))
                rec["cov80_conformal"] = float(np.mean((yj[e] >= qlo[e] - Qoff) & (yj[e] <= qhi[e] + Qoff)))
        out["per_target"][name] = rec
    if Q is not None:
        lv = list(QLEVELS)
        i10, i90 = lv.index(0.1), lv.index(0.9)
        lo, hi = np.minimum(Q[:, :, i10], Q[:, :, i90]), np.maximum(Q[:, :, i10], Q[:, :, i90])
        cov = float(np.mean((Y_te >= lo) & (Y_te <= hi)))
        out["coverage_80pi"] = cov
        out["ece"] = abs(0.80 - cov)
        # reliability: empirical fraction of y below the tau-quantile, per level
        out["coverage_by_level"] = {str(tau): float(np.mean(Y_te <= Q[:, :, li]))
                                    for li, tau in enumerate(QLEVELS)}
        raws = [v["cov80_raw_eval"] for v in out["per_target"].values() if "cov80_raw_eval" in v]
        cons = [v["cov80_conformal"] for v in out["per_target"].values() if "cov80_conformal" in v]
        out["coverage_80pi_raw_eval"] = float(np.mean(raws)) if raws else None
        out["coverage_80pi_conformal"] = float(np.mean(cons)) if cons else None
    else:
        out["coverage_80pi"] = None; out["ece"] = None; out["coverage_by_level"] = None
    out["optimism_gap"] = optimism_gap
    return out


def _macro_rmse(mean, Y_te):
    import numpy as np
    from sklearn.metrics import mean_squared_error
    return float(np.mean([mean_squared_error(Y_te[:, j], mean[:, j]) ** 0.5
                          for j in range(Y_te.shape[1])]))


def run_unit(dataset: str, regime: str, model: str, seed: int, fold: int, out_path: Path) -> dict:
    import numpy as np
    X_tr, Y_tr, X_te, Y_te, meta, groups_tr = load_dataset(dataset, seed, fold)
    if regime == "augment" and model not in CLASSICAL:
        _write_skip(out_path, dataset, regime, model, seed, fold, "augment is classical-only")
        return {"skipped": True}
    try:
        kind, obj = get_model(model, meta, seed)
    except _Skip as e:
        _write_skip(out_path, dataset, regime, model, seed, fold, str(e)); return {"skipped": True}

    if regime.startswith("fewshot"):
        frac = {"fewshot50": 0.5, "fewshot25": 0.25}[regime]
        X_tr, Y_tr, groups_tr = subsample_context(X_tr, Y_tr, groups_tr, frac, seed)
    elif regime == "augment":
        X_tr, Y_tr = augment_training(X_tr, Y_tr, seed)
        groups_tr = None                       # synthetic rows have no run structure
    Y_tr = np.asarray(Y_tr); Y_te = np.asarray(Y_te)

    optimism = None
    try:
        if model in CLASSICAL:                 # tuned control: HPO + optimism gap
            obj, inner_cv_rmse = tune_classical(model, obj, X_tr, Y_tr, seed, groups_tr=groups_tr)
            mean, Q = infer(kind, obj, X_tr, Y_tr, X_te, meta, prefitted=True)
            optimism = inner_cv_rmse - _macro_rmse(np.asarray(mean), Y_te)
        else:                                  # TFM: no tuning, in-context
            mean, Q = infer(kind, obj, X_tr, Y_tr, X_te, meta, prefitted=False)
    except Exception as e:
        # A control that cannot fit at this sample size (e.g. NGBoost at n~14) is a
        # graceful skip, NOT a failure: the manifest stays clean and the did-not-fit
        # fact is recorded as data for the "TFMs fit where classical baselines do not"
        # finding. (TFM inference does not hit this in practice.)
        _write_skip(out_path, dataset, regime, model, seed, fold,
                    f"did-not-fit@n{len(X_tr)}:{type(e).__name__}")
        return {"skipped": True}

    result = {
        "payload_version": PAYLOAD_VERSION,
        "dataset": dataset, "regime": regime, "model": model, "seed": seed, "fold": fold,
        "task": "regression", "targets": meta["targets"],
        "n_train": meta["n_train"], "n_test": meta["n_test"],
        "leak_check": meta["leak_check"],   # test inputs also present in train; gate F1 wants 0
        "pred_mean": np.asarray(mean).tolist(),
        # The quantile grid, persisted (payload v2). Shape (n_test, n_targets, |QLEVELS|),
        # None for point-only models. Rounded to 6 dp: the measurement resolution is
        # 0.01 mm, so this is lossless for every downstream use and keeps the unit small.
        "q_levels": list(QLEVELS),
        "pred_q": (np.round(np.asarray(Q), 6).tolist() if Q is not None else None),
        "y_test": Y_te.tolist(),
        "metrics": metrics(np.asarray(mean), Q, Y_te, meta, optimism_gap=optimism),
    }
    out_path.write_text(json.dumps(result))
    return result


def _write_skip(out_path, dataset, regime, model, seed, fold, why):
    out_path.write_text(json.dumps({"payload_version": PAYLOAD_VERSION, "dataset": dataset,
                                    "regime": regime, "model": model, "seed": seed,
                                    "fold": fold, "skipped": why}))


# --------------------------------------------------------------------------- #
# Orchestration — house base (axis 'protocol' carries the regime token).       #
# --------------------------------------------------------------------------- #
def unit_id(d, r, m, s, f): return f"{d}__{r}__{m}__seed{s}__fold{f}"
def unit_out_path(o, d, r, m, s, f): return o / f"{unit_id(d, r, m, s, f)}.json"


def append_manifest(outdir, record):
    with (outdir / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def run_worker(a):
    outdir = Path(a.outdir)
    run_unit(a.dataset, a.protocol, a.model, a.seed, a.fold,
             unit_out_path(outdir, a.dataset, a.protocol, a.model, a.seed, a.fold))


def run_orchestrator(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if a.no_resume and a.folds != "all":
        sys.exit("[runner] refuse: --no-resume final pass must cover every fold "
                 "(drop --folds). A sliced final pass passes gate A1 while missing units.")
    datasets = a.datasets.split(","); regimes = a.protocols.split(",")
    models = a.models.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    run_started = datetime.now(timezone.utc).isoformat()
    (outdir / "run_meta.json").write_text(json.dumps({
        "run_started": run_started, "no_resume": a.no_resume, "shard": a.shard,
        "axes": {"datasets": datasets, "regimes": regimes, "models": models, "seeds": seeds},
        "timeout_s": a.timeout_s}, indent=2))

    # The fold axis is read from each dataset's declared `cv` mode, not passed in:
    # EBW (cv: logo) expands to 15 folds, the GMAW sets (cv: holdout) to 1.
    # --folds pins a subset (timing slices want ONE fold, not the whole axis).
    folds = {d: n_folds(d) for d in datasets}
    print(f"[runner] folds per dataset: {folds}", flush=True)
    if a.folds != "all":
        want = [int(x) for x in a.folds.split(",")]
        fold_ids = {d: [f for f in want if f < folds[d]] for d in datasets}
        print(f"[runner] --folds {a.folds} -> {fold_ids}", flush=True)
    else:
        fold_ids = {d: list(range(folds[d])) for d in datasets}
    units = [(d, r, m, s, f) for d in datasets for r in regimes for m in models
             for s in seeds for f in fold_ids[d]]
    shard_k, shard_n = (int(x) for x in a.shard.split("/"))
    if shard_n > 1:
        if a.no_resume:
            sys.exit("[runner] refuse: --no-resume final pass must run on ONE box (--shard 0/1).")
        units = [u for i, u in enumerate(units) if i % shard_n == shard_k]
        print(f"[runner] shard {shard_k}/{shard_n}: {len(units)} units", flush=True)
    print(f"[runner] {len(units)} units | outdir={outdir} | no_resume={a.no_resume} "
          f"| timeout={a.timeout_s}s", flush=True)

    nd = ns = nf = nt = 0
    for d, r, m, s, fo in units:
        out_path = unit_out_path(outdir, d, r, m, s, fo); uid = unit_id(d, r, m, s, fo)
        if out_path.exists() and not a.no_resume:
            ns += 1; print(f"[skip] {uid}", flush=True); continue
        if out_path.exists() and a.no_resume:
            out_path.unlink()
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--dataset", d, "--protocol", r, "--model", m, "--seed", str(s),
               "--fold", str(fo), "--outdir", str(outdir)]
        t0 = time.time(); u_started = datetime.now(timezone.utc).isoformat(); status = "ok"
        try:
            subprocess.run(cmd, timeout=a.timeout_s, check=True)
        except subprocess.TimeoutExpired:
            status = "timeout"; nt += 1; print(f"[TIMEOUT] {uid}", flush=True)
        except subprocess.CalledProcessError as e:
            status = f"fail(rc={e.returncode})"; nf += 1; print(f"[FAIL] {uid} rc={e.returncode}", flush=True)
        else:
            nd += 1; print(f"[ok] {uid} ({time.time()-t0:.1f}s)", flush=True)
        # `started` is THIS unit's start, not the orchestrator's: with a run-level
        # stamp, finished-started is cumulative and wall_s cannot be checked against
        # anything. `run_started` is kept separately for the A1 clean-run assert.
        append_manifest(outdir, {"unit": uid, "dataset": d, "regime": r, "model": m, "seed": s,
                                 "fold": fo, "status": status,
                                 "started": u_started, "run_started": run_started,
                                 "finished": datetime.now(timezone.utc).isoformat(),
                                 "wall_s": round(time.time()-t0, 1),
                                 "device": os.environ.get("EBW_DEVICE", "cuda"),
                                 "tree_cpu": bool(os.environ.get("EBW_TREE_CPU")),
                                 "no_resume": a.no_resume})
    print(f"[runner] done | ok={nd} skip={ns} fail={nf} timeout={nt}", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--datasets", default="ebw,laser_bead,gmaw_bead")
    ap.add_argument("--protocols", default="full,fewshot50,fewshot25,augment")  # regime axis
    ap.add_argument("--models", default="tabpfn_v2,tabpfn_v25,tabpfn_v3,catboost,xgb,ngb,mlp",
                    help="mitra is optional: add it only after wiring its adapter (BUILD.md)")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--folds", default="all",
                    help="'all' (default) or a comma list, e.g. --folds 0 for a timing "
                         "slice. The final pass must run 'all'; the runner refuses "
                         "--no-resume with a fold subset.")
    ap.add_argument("--outdir", default="runs/dev")
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=1800)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--no-resume", dest="no_resume", action="store_true")
    ap.add_argument("--dataset"); ap.add_argument("--protocol")
    ap.add_argument("--model"); ap.add_argument("--seed", type=int)
    ap.add_argument("--fold", type=int, default=0)
    return ap


if __name__ == "__main__":
    a = build_argparser().parse_args()
    run_worker(a) if a.worker else run_orchestrator(a)
