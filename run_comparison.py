"""
Run ridge, gradient boosting and the MLP on identical ground, against the
seasonal-naive baseline and a published ENTSO-E-derived day-ahead benchmark.

    pip install -r requirements.txt
    python run_comparison.py                        # no weather, the old model
    python run_comparison.py --weather lagged       # yesterday's temperature only
    python run_comparison.py --weather noisy        # + synthetic temperature error (sensitivity)
    python run_comparison.py --weather perfect      # perfect prognosis upper bound
    python run_comparison.py --weather noisy --backtest
    python run_comparison.py --weather noisy --seed 7   # a different noise draw
    python selfcheck.py                             # checks, no download

Identical ground: features and split come from src/features.py, every model runs
the same protocol (small grid on validation, one refit on train+val, one score
on test), and src/evaluate.py scores them all on an index it asserts is shared.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # no display in a plain terminal
import matplotlib.pyplot as plt
import pandas as pd

from src.data import ZONE_CENTROIDS, load_frame, load_solar, load_temperature
from src.evaluate import (backtest_folds, backtest_run, backtest_summary,
                          baseline_preds, daylight_rows, mae,
                          mae_by_target_hour, score_table, skill,
                          solar_baseline_preds, solar_score_table, worst_days)
from src.features import (build_features, build_solar_features,
                          chronological_split)
from src.models import ALL_MODELS
from src.solar import fleet_clear_sky

RESULTS = "results"
OUT = os.path.join(RESULTS, "figures")
TZ = "Europe/Berlin"


LOAD_MODES = ("none", "lagged", "noisy", "perfect")
SOLAR_MODES = ("none", "clearsky", "lagged", "perfect")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="load", choices=["load", "solar"])
    p.add_argument("--weather", default=None,
                   choices=sorted(set(LOAD_MODES) | set(SOLAR_MODES)),
                   help=f"load: {'/'.join(LOAD_MODES)};  solar: {'/'.join(SOLAR_MODES)}")
    p.add_argument("--weighting", default="population",
                   choices=["box", "land", "population"],
                   help="how the ERA5 grid is reduced to one national number")
    p.add_argument("--val-start", default="2018-01-01")
    p.add_argument("--test-start", default="2019-01-01")
    p.add_argument("--backtest", action="store_true",
                   help="rolling-origin folds as well as the single split (slow)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=0,
                   help="noise draw for --weather noisy; sweep it rather than "
                        "trusting one run")
    args = p.parse_args()

    allowed = SOLAR_MODES if args.target == "solar" else LOAD_MODES
    if args.weather is None:
        args.weather = "clearsky" if args.target == "solar" else "none"
    if args.weather not in allowed:
        p.error(f"--weather {args.weather} is not available for --target "
                f"{args.target}; choose from {'/'.join(allowed)}")

    os.makedirs(RESULTS, exist_ok=True)
    if args.target == "solar":
        return run_solar(args)
    return run_load(args)


def run_load(args):
    print("Loading German hourly load...")
    frame = load_frame()
    load = frame["load_mw"]
    print(f"{len(load):,} hours, {load.index[0]:%Y-%m-%d} to {load.index[-1]:%Y-%m-%d}")

    temp = None
    if args.weather != "none":
        temp = load_temperature(load.index, weighting=args.weighting)
        print(f"temperature: {temp.min():.1f} to {temp.max():.1f} C, "
              f"mean {temp.mean():.1f} C   "
              f"(mode: {args.weather}, weighting: {args.weighting})")
    print()

    X, y = build_features(load, temp, weather_mode=args.weather, seed=args.seed)
    print(f"{X.shape[1]} features, {len(X):,} usable rows (first 3 weeks go to lags)\n")

    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(
        X, y, args.val_start, args.test_start)
    for nm, s in (("train", ytr), ("val", yva), ("test", yte)):
        print(f"{nm:<6}{s.index[0]:%Y-%m-%d}..{s.index[-1]:%Y-%m-%d}  ({len(s):,}h)")
    print()

    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    print("Tuning on validation (test untouched):")
    preds, chosen = baseline_preds(frame, yte.index), []
    for fit in ALL_MODELS:
        fn, info = fit(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=True)
        preds[info["name"]] = fn(Xte)
        chosen.append(info)
    print("\n  chose " + ", ".join(f"{i['name']} {i['params']}" for i in chosen) + "\n")

    table = score_table(yte, preds)
    print("Test-set results (scored once, never tuned on):\n")
    print(table.round(3).to_string())

    base, best = preds["seasonal_naive"], table.index[0]
    print(f"\nBest model: {best}  ({skill(yte, preds[best], base):.1%} of the "
          f"seasonal-naive error removed)")

    if "entsoe_benchmark" in preds:
        off, bm = mae(yte, preds["entsoe_benchmark"]), mae(yte, preds[best])
        print(f"vs the published ENTSO-E-derived benchmark: {bm:,.0f} vs "
              f"{off:,.0f} MW MAE")
        print("  Not like-for-like: the benchmark is published around 10:00 on "
              "D-1, so this\n  model has ~14 hours more demand data. A lower MAE "
              "here is not a better forecast.")

    gbm_mae, mlp_mae = mae(yte, preds["gbm"]), mae(yte, preds["mlp"])
    gap = abs(gbm_mae - mlp_mae)
    print(f"\nGBM vs MLP: {gbm_mae:,.0f} vs {mlp_mae:,.0f} MW "
          f"({gap / min(gbm_mae, mlp_mae):.1%} apart)")
    if gap / min(gbm_mae, mlp_mae) < 0.02:
        print("  Under 2% on one window - a tie. Use --backtest to see whether "
              "the ordering holds.")

    print("\nMAE by local target hour, MW (NOT lead time - see evaluate.py):")
    lead = mae_by_target_hour(yte, preds)
    print(lead[[c for c in ("gbm", "mlp", "entsoe_benchmark") if c in lead]]
          .round(0).to_string())

    print(f"\nWorst 5 days for {best}:")
    print(worst_days(yte, preds[best]).round(0).to_string())

    if args.backtest:
        print("\nRolling-origin backtest (same protocol, origin walked forward):")
        folds = backtest_folds(X.index, args.test_start, block_months=6)
        tidy = backtest_run(X, y, ALL_MODELS, folds)
        wide, summary = backtest_summary(tidy)
        print("\nMAE per fold, MW:")
        print(wide.round(0).to_string())
        print("\nAcross folds:")
        print(summary.round(1).to_string())
        tidy.insert(0, "weather", args.weather)
        tidy.to_csv(os.path.join(RESULTS, "backtest.csv"), index=False)

    if not args.no_plots:
        made = all_plots(yte, preds, temp)
        print("\nWrote " + ", ".join(made))

    table.to_csv(os.path.join(RESULTS, "scores.csv"))
    pd.DataFrame(preds).assign(actual=yte).to_csv(
        os.path.join(RESULTS, "predictions.csv"))
    print("Wrote results/scores.csv and results/predictions.csv")


def run_solar(args):
    print("Loading German hourly solar generation...")
    df = load_solar()
    cf, cap = df["capacity_factor"], df["capacity_mw"]
    print(f"{len(cf):,} hours, {cf.index[0]:%Y-%m-%d} to {cf.index[-1]:%Y-%m-%d}")
    print(f"installed capacity {cap.iloc[0] / 1000:.1f} -> {cap.iloc[-1] / 1000:.1f} GW "
          f"({cap.iloc[-1] / cap.iloc[0] - 1:+.0%}) - which is why the target is "
          "capacity factor, not MW")

    temp = None
    if args.weather in ("lagged", "perfect"):
        temp = load_temperature(cf.index, weighting=args.weighting)
        print(f"temperature: mean {temp.mean():.1f} C  (weighting: {args.weighting})")

    shares = {z: float(df[f"solar_{z}_mw"].mean()) for z in ZONE_CENTROIDS}
    total = sum(shares.values())
    cs = fleet_clear_sky(cf.index, ZONE_CENTROIDS, shares, air_c=temp)
    print("fleet clear sky, generation-weighted over the control zones: "
          + ", ".join(f"{z} {s / total:.0%}" for z, s in shares.items()))
    print()

    X, y = build_solar_features(cf, None if args.weather == "none" else cs,
                                temp, weather_mode=args.weather)
    print(f"{X.shape[1]} features, {len(X):,} usable rows   (mode: {args.weather})\n")

    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(
        X, y, args.val_start, args.test_start)
    for nm, s in (("train", ytr), ("val", yva), ("test", yte)):
        print(f"{nm:<6}{s.index[0]:%Y-%m-%d}..{s.index[-1]:%Y-%m-%d}  ({len(s):,}h)")
    print()

    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    print("Tuning on validation (test untouched):")
    preds, chosen = solar_baseline_preds(cf, cs, yte.index), []
    for fit in ALL_MODELS:
        fn, info = fit(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=True)
        preds[info["name"]] = fn(Xte).clip(lower=0.0)
        chosen.append(info)
    print("\n  chose " + ", ".join(f"{i['name']} {i['params']}" for i in chosen) + "\n")

    day = daylight_rows(cs, yte.index)
    night_share = 1.0 - len(day) / len(yte)
    print(f"Scoring {len(day):,} daylight hours of {len(yte):,} "
          f"({night_share:.0%} of the test period is night and is dropped).")

    all_hours = solar_score_table(yte, preds, capacity=cap)
    table = solar_score_table(yte.loc[day], {k: v.loc[day] for k, v in preds.items()},
                              capacity=cap)
    print("\nTest-set results, daylight hours only:\n")
    print(table.round(4).to_string())

    best = table.index[0]
    inflate = all_hours.loc[best, "MAE_cf"]
    print(f"\nKeeping the night in would report {inflate:.4f} instead of "
          f"{table.loc[best, 'MAE_cf']:.4f} for {best} - a {1 - inflate / table.loc[best, 'MAE_cf']:.0%} "
          "flattering of a number nobody forecast.")

    cs_p = table.loc["clearsky_persistence", "MAE_cf"]
    print(f"\nBest model: {best}. Against clear-sky persistence, the standard "
          f"solar baseline:\n  {table.loc[best, 'MAE_cf']:.4f} vs {cs_p:.4f} "
          f"({1 - table.loc[best, 'MAE_cf'] / cs_p:+.1%})")

    if not args.no_plots:
        # every hour for the plots, not just the scored ones - see plot_solar_week
        made = [p for p in solar_plots(yte, preds, cs, cf) if p]
        print("\nWrote " + ", ".join(made))

    table.to_csv(os.path.join(RESULTS, "solar_scores.csv"))
    pd.DataFrame(preds).assign(actual=yte).to_csv(
        os.path.join(RESULTS, "solar_predictions.csv"))
    print("Wrote results/solar_scores.csv and results/solar_predictions.csv")


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_actual_vs_forecast(y, preds, days=7, start=None):
    """One week of actual demand with the forecasts on top - whether the model
    tracks the shape and sits slightly off, or misses the peaks."""
    idx = y.index.tz_convert(TZ)
    start = pd.Timestamp(start, tz=TZ) if start else idx[0]
    m = (idx >= start) & (idx < start + pd.Timedelta(days=days))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx[m], y[m] / 1000, color="black", lw=2, label="actual")
    for name in ("gbm", "mlp", "entsoe_benchmark"):
        if name in preds:
            ax.plot(idx[m], preds[name][m] / 1000, lw=1.2, alpha=0.85, label=name)
    ax.set_ylabel("GW")
    ax.set_xlabel(f"local time, {days} days from {start:%Y-%m-%d}")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "actual_vs_forecast.png")


def plot_error_by_target_hour(y, preds):
    """MAE against the target's local clock hour. Not lead time, with a single
    midnight origin the two are the same variable - but it shows which hours
    cost the most."""
    tbl = mae_by_target_hour(y, preds)
    keep = [c for c in ("gbm", "mlp", "ridge", "entsoe_benchmark", "seasonal_naive")
            if c in tbl]

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in keep:
        ax.plot(tbl.index, tbl[c], marker="o", ms=3, lw=1.2, label=c)
    ax.set_xlabel("local hour of target")
    ax.set_ylabel("MAE, MW")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "error_by_target_hour.png")


def plot_load_vs_temperature(y, temp):
    """Demand against temperature. Expect a V - heating at the cold end, cooling
    at the warm end - which is why the degree-hour features exist."""
    t = temp.reindex(y.index)
    hour = y.index.tz_convert(TZ).hour

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(t, y / 1000, c=hour, s=3, alpha=0.35, cmap="twilight")
    binned = (y / 1000).groupby(pd.cut(t, bins=30)).mean()
    ax.plot([iv.mid for iv in binned.index], binned.to_numpy(),
            color="black", lw=2, label="binned mean")
    ax.set_xlabel("temperature, deg C")
    ax.set_ylabel("demand, GW")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="local hour")
    return _save(fig, "load_vs_temperature.png")


def plot_worst_days(y, preds, n=12):
    """The days the model got most wrong, ranked."""
    name = "gbm" if "gbm" in preds else list(preds)[0]
    err = (preds[name] - y).abs()
    daily = err.groupby(y.index.tz_convert(TZ).date).mean().sort_values(ascending=False)
    top = daily.head(n)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([str(d) for d in top.index][::-1], top.to_numpy()[::-1] / 1000,
            color="tab:red", alpha=0.8)
    ax.set_xlabel(f"{name} mean absolute error, GW")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, "worst_days.png")


def plot_solar_week(y, preds, clear_sky, days=7, start=None):
    """A week of capacity factor under its own clear-sky ceiling.

    The ceiling is plane-of-array irradiance as a fraction of standard test
    conditions, so it is an upper bound on capacity factor rather than a tight
    one: a real fleet reaches roughly half of it on a clear day, because the
    installed base is a mixture of orientations and loses more to inverters,
    soiling and heat. The day-to-day variation under it is cloud.

    Plotted over every hour, not the daylight rows the scoring uses: dropping
    the night and then drawing a line joins dusk straight to dawn and fills the
    gap with a shape that never happened.
    """
    idx = y.index.tz_convert(TZ)
    if start is None:
        # midsummer, when there is something to see. November is mostly cloud.
        june = idx[(idx.month == 6) & (idx.day >= 10)]
        start = june[0].normalize() if len(june) else idx[len(idx) // 2]
    else:
        start = pd.Timestamp(start, tz=TZ)
    m = (idx >= start) & (idx < start + pd.Timedelta(days=days))
    if m.sum() < 24:
        return None

    fig, ax = plt.subplots(figsize=(11, 4))
    ceiling = (clear_sky["cs_poa"].reindex(y.index) / 1000.0)[m]
    ax.fill_between(idx[m], 0, ceiling, color="orange", alpha=0.3,
                    label="clear-sky ceiling")
    ax.plot(idx[m], y[m], color="black", lw=2, label="actual")
    for name in ("ridge", "gbm", "clearsky_persistence"):
        if name in preds:
            ax.plot(idx[m], preds[name][m], lw=1.1, alpha=0.85, label=name)
    ax.set_ylabel("capacity factor")
    ax.set_xlabel(f"local time, {days} days from {start:%Y-%m-%d}")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "solar_week.png")


def plot_clearsky_vs_actual(cf, clear_sky):
    """Capacity factor against the clear-sky ceiling, daylight hours.

    Almost every point sits under the diagonal, because cloud can only subtract.
    The upper edge is the clear-sky model, and how tightly the points hug it is
    how much of the problem astronomy has already solved.
    """
    poa = clear_sky["cs_poa"].reindex(cf.index) / 1000.0
    day = clear_sky["daylight"].reindex(cf.index).fillna(False).to_numpy()
    month = cf.index.tz_convert(TZ).month

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(poa[day], cf[day], c=month[day], s=2, alpha=0.25, cmap="twilight")
    lim = float(poa[day].max())
    ax.plot([0, lim], [0, lim], color="black", lw=1.5, ls="--",
            label="clear sky (no cloud)")
    ax.set_xlabel("clear-sky plane-of-array, kW/m2")
    ax.set_ylabel("capacity factor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="month")
    return _save(fig, "solar_clearsky_vs_actual.png")


def solar_plots(y, preds, clear_sky, cf):
    return [plot_solar_week(y, preds, clear_sky),
            plot_clearsky_vs_actual(cf, clear_sky)]


def all_plots(y, preds, temp=None):
    made = [plot_actual_vs_forecast(y, preds),
            plot_error_by_target_hour(y, preds),
            plot_worst_days(y, preds)]
    if temp is not None:
        made.append(plot_load_vs_temperature(y, temp))
    return made


if __name__ == "__main__":
    main()
