"""
Baselines, metrics, the scoring table, and the rolling-origin backtest. Every
model is scored by the same code on the same rows.

Two baselines matter. Seasonal naive says whether the model learned anything;
the ENTSO-E-derived day-ahead forecast that OPSD ships says whether the answer
is in the right ballpark. That one is published around 10:00 on D-1, earlier
than this model's assumed midnight, so beating it is not a like-for-like win.
"""

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def seasonal_naive(load):
    """Same hour, same weekday, last week. The one to beat first."""
    return load.shift(168)


def yesterday(load):
    return load.shift(24)


def mean_last_4_weeks(load):
    return sum(load.shift(168 * (w + 1)) for w in range(4)) / 4


def baseline_preds(frame, index):
    """The shift-based baselines plus the official forecast, cut to the scored rows.

    `frame` is what data.load_frame returns: load_mw and benchmark_mw.
    """
    load = frame["load_mw"]
    out = {
        "yesterday": yesterday(load).reindex(index),
        "seasonal_naive": seasonal_naive(load).reindex(index),
        "mean_last_4_weeks": mean_last_4_weeks(load).reindex(index),
    }
    if "benchmark_mw" in frame:
        bench = frame["benchmark_mw"].reindex(index)
        gaps = int(bench.isna().sum())
        if gaps:
            # drop the column rather than invent forecast values nobody published
            print(f"  benchmark has {gaps} missing hours in the scored window "
                  f"({gaps / len(index):.2%}) - excluded from the table")
        else:
            out["entsoe_benchmark"] = bench
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a, b):
    return float(np.mean(np.abs((a - b) / a)) * 100)


def bias(y, pred):
    """Mean signed error. Positive means the forecast runs high - which MAE
    hides completely, and which matters if you buy generation against it."""
    return float(np.mean(pred - y))


def skill(y, pred, base):
    """Fraction of the baseline's error removed. 0 = no better than the baseline."""
    return 1.0 - mae(y, pred) / mae(y, base)


def predict(model, X):
    return pd.Series(model.predict(X), index=X.index)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def assert_same_rows(y, preds):
    """Every model and baseline scored on the same timestamps in the same order.
    A model that quietly dropped rows would average over different hours, and the
    table would compare nothing."""
    for name, pr in preds.items():
        if len(pr) != len(y):
            raise AssertionError(
                f"{name}: {len(pr)} predictions vs {len(y)} target rows")
        if not pr.index.equals(y.index):
            raise AssertionError(f"{name}: scored on a different index to the target")
        if pr.isna().any():
            raise AssertionError(f"{name}: {int(pr.isna().sum())} NaN predictions")


def score_table(y, preds, base="seasonal_naive"):
    assert_same_rows(y, preds)
    b = preds[base]
    rows = {n: {"MAE_MW": mae(y, pr), "RMSE_MW": rmse(y, pr), "MAPE_%": mape(y, pr),
                "bias_MW": bias(y, pr), "skill_vs_naive": skill(y, pr, b)}
            for n, pr in preds.items()}
    return pd.DataFrame(rows).T.sort_values("MAE_MW")


def mae_by_target_hour(y, preds):
    """Error against the target's local clock hour.

    Not lead-time verification: with a single midnight origin, clock hour and
    horizon are the same variable. What it shows is which hours are hard.
    """
    lead = pd.Index(y.index.tz_convert(TZ).hour, name="local_hour")
    return pd.DataFrame({n: pd.Series((pr - y).abs().to_numpy()).groupby(lead).mean()
                         for n, pr in preds.items()})


def worst_days(y, pred, n=5):
    err = (pred - y).abs()
    local_date = pd.Index(y.index.tz_convert(TZ).date, name="date")
    return err.groupby(local_date).mean().sort_values(ascending=False).head(n)


# --------------------------------------------------------------------------
# rolling-origin backtest
#
# One split gives one number per model and no way to tell a real gap from the
# window you happened to pick - on the single split the booster and the MLP
# finish under 2% apart. So walk the origin forward: each fold trains up to its
# own cutoff, tunes on the year before its test block, and scores the block.
#
# Folds share training data and load is serially correlated, so a fold majority
# is a stability signal rather than four independent trials.
# --------------------------------------------------------------------------
def backtest_folds(index, first_test_start, block_months=6, val_months=12):
    """Expanding-window folds: (val_start, test_start, test_end) per fold."""
    start = pd.Timestamp(first_test_start, tz="UTC")
    last = index.max()
    folds = []
    while start < last:
        end = start + pd.DateOffset(months=block_months)
        val_start = start - pd.DateOffset(months=val_months)
        if end > last:
            end = last + pd.Timedelta("1h")
        if (index >= start).sum() < 24 * 30:   # skip a stub final block
            break
        folds.append((val_start, start, end))
        start = end
    return folds


def backtest_run(X, y, models, folds, verbose=True):
    """Fit every model on every fold. Returns tidy MAE per (fold, model)."""
    from .features import chronological_split

    rows = []
    for k, (val_start, test_start, test_end) in enumerate(folds, 1):
        Xk, yk = X[X.index < test_end], y[y.index < test_end]
        (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(
            Xk, yk, val_start, test_start)
        Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

        if verbose:
            print(f"  fold {k}: train {len(ytr):,}h  val {len(yva):,}h  "
                  f"test {yte.index[0]:%Y-%m-%d}..{yte.index[-1]:%Y-%m-%d} "
                  f"({len(yte):,}h)")

        preds = {}
        for fit in models:
            fn, info = fit(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=False)
            preds[info["name"]] = fn(Xte)
        assert_same_rows(yte, preds)   # same guarantee as the main run

        for name, pr in preds.items():
            rows.append({"fold": k,
                         "test_start": yte.index[0].date(),
                         "model": name,
                         "MAE_MW": mae(yte, pr)})

    return pd.DataFrame(rows)


def backtest_summary(tidy):
    """Mean MAE per model, plus how many folds each one won."""
    wide = tidy.pivot(index="fold", columns="model", values="MAE_MW")
    wins = wide.idxmin(axis=1).value_counts()
    out = pd.DataFrame({
        "mean_MAE_MW": wide.mean(),
        "worst_fold_MAE_MW": wide.max(),
        "folds_won": wins.reindex(wide.columns).fillna(0).astype(int),
    }).sort_values("mean_MAE_MW")
    return wide, out
