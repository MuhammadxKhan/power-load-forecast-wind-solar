"""
The ablation tables in ANALYSIS.md, in one pass.

Three questions, all answered with gradient boosting on the same split so the
rows are comparable:

  1. what each weather mode is worth, for demand and for solar
  2. what reducing the ERA5 grid properly is worth
  3. what the clear-sky physics is worth when there is no generation history -
     which is the situation a new asset is in

Prints tables. Nothing here is committed; the numbers go into ANALYSIS.md and
this is how they are regenerated.
"""

import pandas as pd

from src.data import ZONE_CENTROIDS, load_frame, load_solar, load_temperature
from src.evaluate import backtest_folds, backtest_run, mae
from src.features import (build_features, build_solar_features,
                          chronological_split, solar_history_features)
from src.models import fit_gbm
from src.solar import fleet_clear_sky

VAL_START, TEST_START = "2018-01-01", "2019-01-01"


def _fit_score(X, y, mask=None):
    """Fit and score with the same protocol run_comparison.py uses.

    When a mask is given - solar, where it is the daylight hours - the model is
    fitted and tuned on the masked rows as well as scored on them. Tuning on the
    unmasked set would select hyperparameters on a metric half of which is night,
    which changes which model wins.
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(X, y, VAL_START, TEST_START)

    if mask is None:
        fn, _ = fit_gbm(Xtr, ytr, Xva, yva,
                        pd.concat([Xtr, Xva]), pd.concat([ytr, yva]), verbose=False)
        return mae(yte, fn(Xte))

    def _keep(obj):
        m = mask.reindex(obj.index).fillna(False).to_numpy().astype(bool)
        return obj[m]

    Xtr, ytr, Xva, yva = _keep(Xtr), _keep(ytr), _keep(Xva), _keep(yva)
    fn, _ = fit_gbm(Xtr, ytr, Xva, yva,
                    pd.concat([Xtr, Xva]), pd.concat([ytr, yva]), verbose=False)
    Xte_d, yte_d = _keep(Xte), _keep(yte)
    return mae(yte_d, fn(Xte_d).clip(lower=0.0))


def demand_modes(load, weighting="population"):
    rows = []
    for mode in ("none", "lagged", "noisy", "perfect"):
        temp = None if mode == "none" else load_temperature(load.index, weighting=weighting)
        X, y = build_features(load, temp, weather_mode=mode, seed=0)
        rows.append({"mode": mode, "MAE_MW": round(_fit_score(X, y), 1)})
    out = pd.DataFrame(rows).set_index("mode")
    out["vs none"] = (out["MAE_MW"] - out.loc["none", "MAE_MW"]).round(1)
    return out


def weighting_ablation(load):
    rows = []
    for w in ("box", "land", "population"):
        temp = load_temperature(load.index, weighting=w)
        X, y = build_features(load, temp, weather_mode="perfect")
        rows.append({"weighting": w, "MAE_MW": round(_fit_score(X, y), 1)})
    out = pd.DataFrame(rows).set_index("weighting")
    out["vs box"] = (out["MAE_MW"] - out.loc["box", "MAE_MW"]).round(1)
    return out


def solar_modes(cf, cs, temp, daylight, drop_history=False):
    rows = []
    for mode in ("none", "clearsky", "lagged", "perfect"):
        X, y = build_solar_features(cf, None if mode == "none" else cs, temp, mode)
        if drop_history:
            # from features.py, so a new target-derived column cannot escape it
            hist = solar_history_features(mode)
            X = X.drop(columns=[c for c in hist if c in X])
        rows.append({"mode": mode, "MAE_cf": round(_fit_score(X, y, daylight), 4)})
    out = pd.DataFrame(rows).set_index("mode")
    base = out.loc["none", "MAE_cf"]
    out["vs none"] = ((out["MAE_cf"] - base) / base * 100).round(1)
    return out


def fold_weather_delta(load, weighting="population"):
    """The weather gain across rolling-origin folds, gradient boosting only.

    A single test window gives one number and no way to tell a real effect from
    the window. Folds share training data and demand is serially correlated, so a
    majority across them is a stability signal rather than four independent
    trials - but a sign that flips between folds is worth knowing about.
    """
    out, starts = {}, None
    for mode in ("none", "noisy"):
        temp = None if mode == "none" else load_temperature(load.index, weighting=weighting)
        X, y = build_features(load, temp, weather_mode=mode, seed=0)
        folds = backtest_folds(X.index, TEST_START, block_months=6)
        tidy = backtest_run(X, y, [fit_gbm], folds, verbose=False).set_index("fold")
        out[mode] = tidy["MAE_MW"]
        if starts is None:
            starts = tidy["test_start"]
        elif not starts.equals(tidy["test_start"]):
            raise AssertionError("folds differ between modes - not comparable")

    tbl = pd.DataFrame(out).round(1)
    tbl.insert(0, "test_start", starts)
    tbl["delta"] = (tbl["noisy"] - tbl["none"]).round(1)
    return tbl


def seed_study(load, n=10, weighting="population"):
    """`noisy` across n draws of the synthetic error.

    One draw is one realisation, not an uncertainty estimate. If the spread
    across seeds is the size of the effect, the single-run number was noise.
    """
    temp = load_temperature(load.index, weighting=weighting)
    scores = []
    for seed in range(n):
        X, y = build_features(load, temp, weather_mode="noisy", seed=seed)
        scores.append(round(_fit_score(X, y), 1))
    s = pd.Series(scores, index=range(n), name="MAE_MW")
    s.index.name = "seed"
    return s


def monthly_breakdown(load, weighting="population"):
    """Demand MAE by calendar month, with and without weather."""
    out = {}
    for mode in ("none", "noisy"):
        temp = None if mode == "none" else load_temperature(load.index, weighting=weighting)
        X, y = build_features(load, temp, weather_mode=mode, seed=0)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(
            X, y, VAL_START, TEST_START)
        fn, _ = fit_gbm(Xtr, ytr, Xva, yva,
                        pd.concat([Xtr, Xva]), pd.concat([ytr, yva]), verbose=False)
        err = (fn(Xte) - yte).abs()
        out[mode] = err.groupby(yte.index.tz_convert("Europe/Berlin").month).mean()

    tbl = pd.DataFrame(out)
    tbl["delta_%"] = ((tbl["noisy"] - tbl["none"]) / tbl["none"] * 100).round(1)
    tbl.index.name = "month"
    return tbl.round(0)


def main():
    pd.set_option("display.width", 100)

    frame = load_frame()
    load = frame["load_mw"]

    print("\n== demand: what each weather mode is worth ==")
    print(demand_modes(load).to_string())

    print("\n== demand: what reducing the grid properly is worth (mode=perfect) ==")
    print(weighting_ablation(load).to_string())

    df = load_solar()
    cf = df["capacity_factor"]
    shares = {z: float(df[f"solar_{z}_mw"].mean()) for z in ZONE_CENTROIDS}
    temp = load_temperature(cf.index, weighting="population")
    cs = fleet_clear_sky(cf.index, ZONE_CENTROIDS, shares, air_c=temp)
    daylight = cs["daylight"]

    print("\n== solar: what each mode is worth, with generation history ==")
    print(solar_modes(cf, cs, temp, daylight).to_string())

    print("\n== solar: the same, with no generation history ==")
    print(solar_modes(cf, cs, temp, daylight, drop_history=True).to_string())

    poa = cs["cs_poa"]
    drift = (poa - poa.shift(24)).abs()[daylight].mean() / poa[daylight].mean()
    print(f"\nclear-sky irradiance drift over 24h: {drift:.2%} of the daylight mean")
    print(f"daylight hours: {daylight.mean():.1%} of the year")

    print("\n== demand: the weather gain across rolling folds ==")
    ft = fold_weather_delta(load)
    print(ft.to_string())
    print(f"mean {ft['delta'].mean():.1f} MW, "
          f"helps in {(ft['delta'] < 0).sum()} of {len(ft)} folds, "
          f"fold-to-fold range {ft['delta'].max() - ft['delta'].min():.1f}")

    print("\n== demand: does the weather gain survive reseeding? ==")
    s = seed_study(load)
    print(s.to_string())
    print(f"mean {s.mean():.1f}, std {s.std():.1f}, "
          f"range {s.min():.1f} to {s.max():.1f} (spread {s.max() - s.min():.1f})")

    print("\n== demand: the effect by month ==")
    print(monthly_breakdown(load).to_string())


if __name__ == "__main__":
    main()
