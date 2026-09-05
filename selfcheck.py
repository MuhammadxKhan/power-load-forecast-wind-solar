"""
Self-checks on synthetic data. No download, no network.

    python selfcheck.py

The printed numbers are meaningless - the data is fake. What matters is that the
assertions hold: nothing reaches forward in time, calendars are on German
clocks, each weather mode does what it claims, the MLP's scalers never see the
test period, and every model is scored on the same rows.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import (ZONE_CENTROIDS, _from_netcdf, fake_frame, fake_solar,
                      fake_temperature)
from src.evaluate import (assert_same_rows, baseline_preds, clearsky_persistence,
                          daylight_rows, mae, mae_by_target_hour, predict,
                          seasonal_naive, skill, solar_yesterday)
from src.features import (build_features, build_solar_features,
                          chronological_split, degree_hours, usable_temperature)
from src.geo import (Grid, area_weights, grid_weights, inside_outline,
                     population_weights)
from src.models import ALL_MODELS, fit_mlp
from src.solar import (air_mass, cell_temperature, clear_sky_dni, clear_sky_ghi,
                       clear_sky_index, fleet_clear_sky, is_daylight,
                       plane_of_array, solar_position, temperature_derate)

VAL_START, TEST_START = "2016-09-01", "2016-11-01"


def _poke(series, pos, amount):
    out = series.copy()
    out.iloc[pos] += amount
    return out, out.index[pos]


def _changed_rows(before, after):
    common = before.index.intersection(after.index)
    return common[(before.loc[common] != after.loc[common]).any(axis=1)]


def check_no_load_leakage(load):
    # 1) no feature may react to a load change less than 24h before it. Note
    # the >= t: a feature reading the value AT t would be the worst leak of all.
    Xb, _ = build_features(load)
    poked, t = _poke(load, 3000, 50000)
    Xa, _ = build_features(poked)

    changed = _changed_rows(Xb, Xa)
    too_soon = changed[(changed >= t) & (changed < t + pd.Timedelta("24h"))]
    assert len(too_soon) == 0, f"LEAKAGE: features reacted within 24h at {list(too_soon)[:3]}"
    assert (changed >= t + pd.Timedelta("24h")).any(), "lags look broken - nothing reacted"
    print("  [ok] no feature uses load data newer than 24h")


def check_local_time(load):
    # 2) calendar features follow German clocks, not UTC.
    # 23:00 UTC on 31 Dec is already New Year's Day in Germany.
    X, _ = build_features(load)
    loc = X.index.tz_convert("Europe/Berlin")

    assert (X["hour"].to_numpy() == loc.hour.to_numpy()).all(), "hour is not local"
    assert (X["dayofweek"].to_numpy() == loc.dayofweek.to_numpy()).all(), "dow is not local"
    assert (X["hour"].to_numpy() != X.index.hour.to_numpy()).any(), \
        "local and UTC hours are identical here, so this check proves nothing"

    nye = pd.Timestamp("2016-12-31 23:00", tz="UTC")
    if nye in X.index:
        assert X.loc[nye, "is_holiday"] == 1, \
            "23:00 UTC on 31 Dec is 1 Jan in Germany and should be a holiday"
        assert X.loc[nye, "hour"] == 0, "local hour should be 0"
    print("  [ok] calendar features are on Europe/Berlin, not UTC")


def check_feature_table(load):
    # 3) target not among the features, no NaNs, aligned
    X, y = build_features(load)
    assert "load_mw" not in X.columns and len(X) == len(y) and (X.index == y.index).all()
    assert not X.isna().any().any() and not y.isna().any()
    print("  [ok] feature table is clean and aligned")


def check_baseline_and_skill(frame):
    # 4) seasonal naive is a 168h shift, and the skill score has the right signs
    load = frame["load_mw"]
    assert seasonal_naive(load).iloc[168] == load.iloc[0]
    yv = pd.Series([10.0, 20.0, 30.0])
    b = pd.Series([12.0, 18.0, 33.0])
    assert abs(skill(yv, yv, b) - 1.0) < 1e-9 and abs(skill(yv, b, b)) < 1e-9
    print("  [ok] seasonal naive is a 168h shift, skill score behaves")


def check_beats_naive(frame):
    # 5) a model beats naive on clean synthetic data
    load = frame["load_mw"]
    X, y = build_features(load)
    (Xtr, ytr), _, (Xte, yte) = chronological_split(X, y, VAL_START, TEST_START)
    m = HistGradientBoostingRegressor(max_iter=100, early_stopping=False,
                                      random_state=0).fit(Xtr, ytr)
    nv = seasonal_naive(load).reindex(yte.index)
    assert mae(yte, predict(m, Xte)) < mae(yte, nv)
    print("  [ok] model beats the baseline on synthetic data")


def check_weather_modes(load):
    """6) each weather mode does what it claims, and 'lagged' cannot leak.

    Poke the temperature series and see which modes react. 'lagged' must not
    react inside 24h; 'perfect' must react at the poked hour. Asserting both
    directions stops the two modes quietly collapsing into each other.
    """
    temp = fake_temperature(load.index, seed=3)

    hdh, cdh = degree_hours(pd.Series([-5.0, 18.0, 30.0]))
    assert hdh.tolist() == [20.0, 0.0, 0.0] and cdh.tolist() == [0.0, 0.0, 8.0]

    assert usable_temperature(temp, "none") is None
    assert usable_temperature(temp, "perfect").equals(temp)
    assert usable_temperature(temp, "lagged").equals(temp.shift(24))
    fc = usable_temperature(temp, "noisy", seed=0)
    assert not fc.equals(temp), "noisy mode must not be the exact truth"
    assert usable_temperature(temp, "noisy", seed=0).equals(fc), "noisy mode not seeded"

    poked, t = _poke(temp, 4000, 25.0)

    Xb, _ = build_features(load, temp, weather_mode="lagged")
    Xa, _ = build_features(load, poked, weather_mode="lagged")
    changed = _changed_rows(Xb, Xa)
    too_soon = changed[(changed >= t) & (changed < t + pd.Timedelta("24h"))]
    assert len(too_soon) == 0, f"LEAKAGE: lagged weather reacted within 24h at {list(too_soon)[:3]}"
    assert len(changed) > 0, "lagged weather never reacted at all - features look dead"
    print("  [ok] weather_mode='lagged' uses no temperature newer than 24h")

    Xb, _ = build_features(load, temp, weather_mode="perfect")
    Xa, _ = build_features(load, poked, weather_mode="perfect")
    changed = _changed_rows(Xb, Xa)
    assert t in changed, "perfect mode should react at the poked hour - it isn't perfect prog"
    print("  [ok] weather_mode='perfect' does use target-hour temperature, as documented")

    n_none = build_features(load, temp, weather_mode="none")[0].shape[1]
    n_wx = build_features(load, temp, weather_mode="lagged")[0].shape[1]
    assert n_wx == n_none + 5, f"expected 5 weather features, got {n_wx - n_none}"
    print(f"  [ok] weather adds exactly 5 features ({n_none} -> {n_wx})")


def check_netcdf_reader():
    """7) the ERA5 NetCDF reader actually reads NetCDF.

    Everything else here runs on a plain pandas Series and never touches xarray.
    This builds a real two-file fixture in ERA5's layout, reads it through the
    same _from_netcdf the pipeline uses, and deletes it. Two files because the
    downloader writes one per year, and the multi-file path is the one that
    breaks.
    """
    try:
        import xarray as xr
    except ImportError:
        raise AssertionError(
            "xarray is a pinned dependency but isn't installed, so the NetCDF "
            "reader is untested. Failing is better than skipping quietly.")

    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        files, expected = [], []
        for k, yr in enumerate((2016, 2017)):
            idx = pd.date_range(f"{yr}-01-01", periods=36, freq="h")
            lats = np.arange(55.0, 53.9, -0.25)
            lons = np.arange(5.5, 6.6, 0.25)
            # every cell in hour i holds 273.15 + i + k, so the mean is i + k in C
            base = np.arange(len(idx), dtype="float32") + k + 273.15
            data = np.repeat(np.repeat(base[:, None, None], len(lats), 1), len(lons), 2)
            f = f"{tmp}/era5_t2m_{yr}.nc"
            xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"), data)},
                       coords={"valid_time": idx, "latitude": lats,
                               "longitude": lons}).to_netcdf(f)
            files.append(f)
            expected.extend((np.arange(len(idx)) + k).tolist())

        s = _from_netcdf(files)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert len(s) == 72, f"expected 72 hours across two files, got {len(s)}"
    assert s.index.tz is not None and str(s.index.tz) == "UTC", "ERA5 index must be UTC"
    assert np.allclose(s.to_numpy(), expected), \
        "spatial mean or the Kelvin->Celsius conversion is wrong"
    assert s.index.is_monotonic_increasing, "concatenated files are out of order"
    print("  [ok] NetCDF reader handles multiple files, no dask, K->C correct")


def check_mlp_scalers_and_determinism(load):
    """8) the MLP's scalers only ever see the fold it is fitted on.

    The fit functions are never handed the test set, so structurally they cannot
    scale by test statistics. This checks it the hard way: wreck the load series
    inside the test period, refit, and require the model that comes out to be
    bit-for-bit the one from the clean run.
    """
    X, y = build_features(load)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(X, y, VAL_START, TEST_START)
    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    fn_a, info_a = fit_mlp(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=False)
    fn_b, _ = fit_mlp(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=False)
    pa = fn_a(Xte)
    assert (pa.to_numpy() == fn_b(Xte).to_numpy()).all(), "MLP is not deterministic"
    print("  [ok] MLP is bit-identical across runs (fixed seed, fixed batch order)")

    assert np.array_equal(info_a["scaler_x_mean"], Xfit.to_numpy(dtype=np.float64).mean(axis=0))
    assert float(yfit.to_numpy().mean()) == info_a["scaler_y_mean"]
    assert not np.allclose(X.to_numpy(dtype=np.float64).mean(axis=0),
                           info_a["scaler_x_mean"]), \
        "fit-fold and full-series stats are identical here, so this proves nothing"

    cut = pd.Timestamp(TEST_START, tz="UTC") + pd.Timedelta("504h")
    wrecked = load.copy()
    wrecked.loc[cut:] = wrecked.loc[cut:] * 7.5
    Xw, yw = build_features(wrecked)
    (Xtr_w, ytr_w), (Xva_w, yva_w), _ = chronological_split(Xw, yw, VAL_START, TEST_START)
    assert Xtr_w.equals(Xtr) and Xva_w.equals(Xva), "the wrecking touched the training folds"

    fn_w, info_w = fit_mlp(
        Xtr_w, ytr_w, Xva_w, yva_w,
        pd.concat([Xtr_w, Xva_w]), pd.concat([ytr_w, yva_w]), verbose=False)
    assert np.array_equal(info_a["scaler_x_mean"], info_w["scaler_x_mean"])
    assert (pa.to_numpy() == fn_w(Xte).to_numpy()).all(), \
        "test-period values changed the fitted MLP - something leaked"
    print("  [ok] wrecking the test period leaves the fitted MLP bit-identical")


def check_same_rows(frame):
    """9) every model and baseline scored on identical rows."""
    load = frame["load_mw"]
    X, y = build_features(load)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(X, y, VAL_START, TEST_START)
    Xfit, yfit = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    preds = baseline_preds(frame, yte.index)
    assert "entsoe_benchmark" in preds, "the published benchmark should be in the baselines"
    for fit in ALL_MODELS:
        fn, info = fit(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=False)
        preds[info["name"]] = fn(Xte)

    assert_same_rows(yte, preds)
    print(f"  [ok] all {len(preds)} models scored on the same {len(yte):,} rows")

    lead = mae_by_target_hour(yte, preds)
    assert list(lead.index) == list(range(24)), "local hour should run 0..23"

    broken = dict(preds)
    broken["gbm"] = broken["gbm"].iloc[:-1]
    try:
        assert_same_rows(yte, broken)
    except AssertionError:
        print("  [ok] the same-rows check actually fails when rows differ")
    else:
        raise AssertionError("assert_same_rows passed a mismatched set - it is useless")


def check_land_mask():
    """10) the German outline masks the right cells.

    The bounding box is a rectangle over the North Sea, the Baltic and four
    neighbours. The area the mask keeps is checkable against a number everyone
    knows: Germany is 357,000 km2. Cell coordinates are built here rather than
    read, so this needs no NetCDF.
    """
    lats = np.arange(47.0, 55.01, 0.25)
    lons = np.arange(5.5, 15.51, 0.25)
    mask = inside_outline(lats, lons)

    # a 0.25 deg cell is 0.25*111 km tall and that times cos(lat) wide
    km2 = (0.25 * 111.0) ** 2 * np.cos(np.deg2rad(lats))[:, None]
    area = float((km2 * mask).sum())
    assert 0.90 < area / 357_000 < 1.10, \
        f"masked area {area:,.0f} km2 is not within 10% of Germany's 357,000"

    assert not mask[np.argmin(abs(lats - 54.5)), np.argmin(abs(lons - 6.5))], \
        "a North Sea cell is inside the mask"
    assert not mask[np.argmin(abs(lats - 52.5)), np.argmin(abs(lons - 15.25))], \
        "a cell well into Poland is inside the mask"
    assert mask[np.argmin(abs(lats - 52.5)), np.argmin(abs(lons - 13.5))], \
        "Berlin is outside the mask"
    assert mask[np.argmin(abs(lats - 48.25)), np.argmin(abs(lons - 11.5))], \
        "Munich is outside the mask"
    print(f"  [ok] land mask keeps {mask.sum()} cells, {area:,.0f} km2 against 357,000")


def check_weights():
    """11) every weighting is a mean, and population weighting moves west."""
    lats = np.arange(47.0, 55.01, 0.25)
    lons = np.arange(5.5, 15.51, 0.25)

    times = pd.date_range("2016-01-01", periods=5, freq="h", tz="UTC")
    flat = np.full((len(times), len(lats), len(lons)), 7.5)
    g = Grid(times, lats, lons, flat)

    for kind in ("box", "land", "population"):
        w = grid_weights(lats, lons, kind)
        assert (w >= 0).all(), f"{kind} produced a negative weight"
        got = g.reduce(w)
        assert np.allclose(got.to_numpy(), 7.5), \
            f"{kind} does not average a constant field back to the constant"

    # cos(lat) must fall from south to north
    a = area_weights(lats, lons)
    assert a[0, 0] > a[-1, 0], "area weight should shrink towards the pole"

    pop = population_weights(lats, lons)
    nrw = pop[np.argmin(abs(lats - 51.5)), np.argmin(abs(lons - 7.5))]
    mv = pop[np.argmin(abs(lats - 53.8)), np.argmin(abs(lons - 12.5))]
    assert nrw > 3 * mv, \
        f"North Rhine-Westphalia ({nrw:.2f}) should far outweigh Mecklenburg ({mv:.2f})"
    print("  [ok] all three weightings average a constant field, population leans west")


def check_bilinear():
    """12) series_at is bilinear: exact on nodes, linear between them."""
    lats = np.array([50.0, 50.25, 50.5])
    lons = np.array([9.0, 9.25, 9.5])
    times = pd.date_range("2016-01-01", periods=3, freq="h", tz="UTC")

    # a plane in lat and lon; bilinear interpolation reproduces a plane exactly
    lon_g, lat_g = np.meshgrid(lons, lats)
    plane = 2.0 * lat_g + 3.0 * lon_g
    vals = np.stack([plane + t for t in range(len(times))])
    g = Grid(times, lats, lons, vals)

    assert np.allclose(g.at(50.25, 9.25).to_numpy(), vals[:, 1, 1]), "not exact at a node"

    got = g.at(50.1, 9.4).to_numpy()
    want = 2.0 * 50.1 + 3.0 * 9.4 + np.arange(len(times))
    assert np.allclose(got, want), f"plane not reproduced: {got} vs {want}"

    # a descending-latitude file must give the same answer as an ascending one
    flipped = Grid(times, lats[::-1], lons, vals[:, ::-1, :])
    assert np.allclose(flipped.at(50.1, 9.4).to_numpy(), want), \
        "latitude order changes the answer"

    for bad in ((49.0, 9.25), (50.25, 12.0)):
        try:
            g.at(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"at{bad} is outside the grid and should refuse")
    print("  [ok] bilinear interpolation is exact on a plane, order-independent")


def check_solar_geometry():
    """13) the sun is where astronomy says it is.

    None of this is fitted, so it can be checked against numbers that come from
    outside the repo: the earth's axial tilt is 23.44 degrees, the equation of
    time runs from about -14 to +16 minutes, and noon elevation at a known
    latitude on a solstice is 90 - latitude +/- the tilt.
    """
    idx = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h", tz="UTC")
    pos = solar_position(idx, 52.52, 13.40)          # Berlin

    d = pos["declination"]
    assert abs(d.max() - 23.44) < 0.05 and abs(d.min() + 23.44) < 0.05, \
        f"declination runs {d.min():.2f}..{d.max():.2f}, axial tilt is 23.44"

    eot = pos["eot_minutes"]
    assert -15 < eot.min() < -13 and 15 < eot.max() < 17, \
        f"equation of time runs {eot.min():.1f}..{eot.max():.1f}, expected ~-14..+16"

    for day, sign in (("2019-06-21", 1.0), ("2019-12-21", -1.0)):
        peak = pos.loc[day, "elevation"].max()
        want = 90.0 - 52.52 + sign * 23.44
        assert abs(peak - want) < 0.3, \
            f"{day} noon elevation {peak:.2f}, geometry says {want:.2f}"

    # At its highest the sun bears due south from Germany. Checked at minute
    # resolution: azimuth moves about a quarter of a degree a minute near noon,
    # so an hourly grid misses the crossing by several degrees. This is also what
    # tests the equation of time - drop that correction and solar noon lands up
    # to a quarter of an hour out.
    fine = pd.date_range("2019-06-21 09:00", "2019-06-21 13:00", freq="min", tz="UTC")
    fine_pos = solar_position(fine, 52.52, 13.40)
    noon = fine_pos["elevation"].idxmax()
    assert abs(fine_pos.loc[noon, "azimuth"] - 180.0) < 0.5, \
        f"sun bears {fine_pos.loc[noon, 'azimuth']:.2f} at solar noon, should be 180"

    # Berlin is 13.4E, so solar noon runs about 54 minutes ahead of noon UTC
    offset = (noon - pd.Timestamp("2019-06-21 12:00", tz="UTC")).total_seconds() / 60
    assert -62 < offset < -45, f"solar noon is {offset:.0f} min from noon UTC, expected ~-54"

    # and below the horizon in the middle of a December night
    assert pos.loc["2019-12-21 01:00", "zenith"] > 90.0, "sun is up at 1am in December"

    assert air_mass(0.0) < 1.001, "air mass overhead should be 1"
    assert air_mass(60.0) > 1.9, "air mass at 60 degrees should be about 2"
    print("  [ok] solar position matches axial tilt, equation of time, solstice noon")


def check_clear_sky():
    """14) the clear-sky model is internally consistent.

    A panel laid flat must collect exactly the global horizontal irradiance -
    that is the definition, and it is the check that catches an inconsistent
    beam/diffuse split, because a beam larger than the global it belongs to
    would force a negative diffuse term.
    """
    idx = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h", tz="UTC")
    pos = solar_position(idx, 51.2, 10.4)
    z, az = pos["zenith"].to_numpy(), pos["azimuth"].to_numpy()

    ghi = clear_sky_ghi(z)
    dni = clear_sky_dni(idx, z)

    night = z >= 90.0
    assert (ghi[night] == 0).all() and (dni[night] == 0).all(), \
        "clear-sky irradiance is non-zero at night"
    assert (ghi >= 0).all() and (dni >= 0).all(), "negative irradiance"
    assert 800 < ghi.max() < 1000, f"peak clear-sky GHI {ghi.max():.0f} W/m2 is implausible"

    flat = plane_of_array(ghi, dni, z, az, tilt=0.0)
    assert np.allclose(flat, ghi), "a flat panel does not collect GHI"

    # the beam's horizontal component cannot exceed the global
    cos_z = np.cos(np.radians(np.clip(z, 0, 90)))
    assert (dni * cos_z <= ghi + 1e-9).all(), "beam exceeds the global it is part of"

    tilted = plane_of_array(ghi, dni, z, az)
    assert (tilted >= 0).all(), "negative plane-of-array irradiance"
    assert 1.1 < tilted.sum() / ghi.sum() < 1.4, \
        "a 30 degree south roof should gain 10-40% over flat under clear skies"

    # pointing the panel north must be worse than pointing it south
    north = plane_of_array(ghi, dni, z, az, panel_azimuth=0.0)
    assert north.sum() < tilted.sum(), "a north-facing roof out-collects a south-facing one"
    print("  [ok] flat panel collects GHI exactly, beam stays inside the global")


def check_cell_temperature():
    """15) panels run hot, and hot panels lose output."""
    assert abs(temperature_derate(25.0) - 1.0) < 1e-12, "derate at 25 C should be 1"
    assert temperature_derate(45.0) < 1.0, "a hot panel should lose output"
    assert temperature_derate(5.0) > 1.0, "a cold panel should gain"

    hot = cell_temperature(25.0, 1000.0)
    assert 50 < hot < 60, f"cell at 25 C air and full sun should be 50-60 C, got {hot:.1f}"
    assert cell_temperature(25.0, 0.0) == 25.0, "no sun means no rise above air"

    # a summer afternoon loses several percent to heat
    loss = 1.0 - temperature_derate(cell_temperature(30.0, 900.0))
    assert 0.05 < loss < 0.20, f"expected a 5-20% heat loss, got {loss:.1%}"
    print(f"  [ok] cells reach {hot:.0f} C in full sun, costing {loss:.0%} at 30 C air")


def check_clear_sky_index():
    """16) the clear-sky index is a ratio, and undefined at night."""
    idx = pd.date_range("2019-06-01", periods=48, freq="h", tz="UTC")
    cs = pd.Series(np.tile(np.clip(np.sin(np.arange(24) / 24 * np.pi), 0, None) * 800, 2),
                   index=idx)
    gen = cs * 0.6

    kt = clear_sky_index(gen, cs)
    day = cs > 1e-3
    assert np.allclose(kt[day], 0.6), "clear-sky index should recover the ratio"
    assert kt[~day].isna().all(), "night hours should be NaN, not a number"

    assert clear_sky_index(cs * 5, cs).max() <= 2.0, "clear-sky index should be capped"
    print("  [ok] clear-sky index recovers the ratio and is NaN at night")


def check_daylight_mask():
    """17) the daylight mask drops the hours every model gets right for free."""
    idx = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h", tz="UTC")
    z = solar_position(idx, 51.2, 10.4)["zenith"].to_numpy()
    day = is_daylight(z)

    frac = day.mean()
    assert 0.45 < frac < 0.60, f"{frac:.1%} of hours in daylight, expected about half"

    # longest day in June, shortest in December
    per_day = pd.Series(day, index=idx).groupby(idx.date).sum()
    assert per_day.max() >= 16, f"longest day only {per_day.max()}h at 51N"
    assert per_day.min() <= 9, f"shortest day still {per_day.min()}h at 51N"
    print(f"  [ok] daylight is {frac:.0%} of the year, {per_day.min()}-{per_day.max()}h a day")


def _fake_solar_inputs(n_days=400, seed=2):
    idx = pd.date_range("2016-01-01", periods=n_days * 24, freq="h", tz="UTC")
    df = fake_solar(idx, seed=seed)
    shares = {z: 1.0 for z in ZONE_CENTROIDS}
    temp = fake_temperature(idx, seed=seed)
    cs = fleet_clear_sky(idx, ZONE_CENTROIDS, shares, air_c=temp)
    return df["capacity_factor"], cs, temp


def check_solar_no_leakage():
    """18) no solar feature reads generation newer than 24 hours.

    Same poke as the demand check. The interesting half is the other direction:
    the clear-sky columns must NOT move at all, because they are astronomy and
    know nothing about what the fleet produced.
    """
    cf, cs, temp = _fake_solar_inputs()

    for mode in ("none", "clearsky", "lagged", "perfect"):
        Xb, _ = build_solar_features(cf, None if mode == "none" else cs, temp, mode)
        poked, t = _poke(cf, 3000, 0.4)
        Xa, _ = build_solar_features(poked, None if mode == "none" else cs, temp, mode)

        changed = _changed_rows(Xb, Xa)
        too_soon = changed[(changed >= t) & (changed < t + pd.Timedelta("24h"))]
        assert len(too_soon) == 0, \
            f"LEAKAGE in mode {mode!r}: reacted within 24h at {list(too_soon)[:3]}"
        assert (changed >= t + pd.Timedelta("24h")).any(), \
            f"mode {mode!r}: nothing reacted at all, the lags look dead"

        if mode != "none":
            sky = [c for c in ("cs_poa", "cs_ghi", "elevation", "azimuth") if c in Xb]
            common = Xb.index.intersection(Xa.index)
            assert (Xb.loc[common, sky] == Xa.loc[common, sky]).all().all(), \
                f"mode {mode!r}: a clear-sky column moved when generation changed"
    print("  [ok] no solar feature reads generation newer than 24h")


def check_solar_modes():
    """19) each solar mode adds what it says, and temperature timing is the only
    difference between lagged and perfect."""
    cf, cs, temp = _fake_solar_inputs()
    cols = {m: set(build_solar_features(cf, None if m == "none" else cs, temp, m)[0])
            for m in ("none", "clearsky", "lagged", "perfect")}

    assert not (cols["none"] & {"cs_poa", "temp_c"}), "mode 'none' picked up weather"
    assert {"cs_poa", "cs_ghi", "elevation", "azimuth"} <= cols["clearsky"], \
        "mode 'clearsky' is missing solar geometry"
    assert "temp_c" not in cols["clearsky"], "mode 'clearsky' should use no temperature"
    assert cols["lagged"] == cols["perfect"], \
        "lagged and perfect must differ only in when temperature is read"
    assert {"temp_c", "cell_c", "derate", "kt_yesterday"} <= cols["lagged"]

    # and they must actually differ in value, or the two modes are the same thing
    Xl, _ = build_solar_features(cf, cs, temp, "lagged")
    Xp, _ = build_solar_features(cf, cs, temp, "perfect")
    common = Xl.index.intersection(Xp.index)
    assert not np.allclose(Xl.loc[common, "temp_c"], Xp.loc[common, "temp_c"]), \
        "lagged and perfect read the same temperature"

    # perfect must react to a temperature poke at the target hour; lagged must not
    poked, t = _poke(temp, 4000, 20.0)
    for mode, should_react in (("lagged", False), ("perfect", True)):
        Xb, _ = build_solar_features(cf, cs, temp, mode)
        Xa, _ = build_solar_features(cf, fleet_clear_sky(
            cf.index, ZONE_CENTROIDS, {z: 1.0 for z in ZONE_CENTROIDS}, air_c=poked),
            poked, mode)
        reacted = t in _changed_rows(Xb, Xa)
        assert reacted == should_react, \
            f"mode {mode!r}: reacted={reacted} at the poked hour, expected {should_react}"
    print("  [ok] solar modes add what they claim, lagged and perfect differ only in timing")


def check_solar_baselines():
    """20) the solar baselines behave, and smart persistence ties with plain
    persistence at a 24-hour horizon because the sun barely moves in a day."""
    cf, cs, _ = _fake_solar_inputs()

    assert solar_yesterday(cf).iloc[24] == cf.iloc[0], "persistence is not a 24h shift"

    smart = clearsky_persistence(cf, cs)
    day = cs["daylight"].to_numpy()
    plain = solar_yesterday(cf)
    both = day & smart.notna().to_numpy() & plain.notna().to_numpy()
    gap = abs((smart[both] - cf[both]).abs().mean() - (plain[both] - cf[both]).abs().mean())
    assert gap < 0.01, f"smart and plain persistence differ by {gap:.4f} at 24h"

    poa = cs["cs_poa"]
    drift = (poa - poa.shift(24)).abs()[day].mean() / poa[day].mean()
    assert drift < 0.05, f"clear-sky irradiance moved {drift:.1%} in 24h, expected under 5%"

    rows = daylight_rows(cs, cf.index)
    frac = len(rows) / len(cf)
    assert 0.40 < frac < 0.65, f"daylight_rows kept {frac:.0%} of hours"
    assert (cs["daylight"].reindex(rows)).all(), "daylight_rows kept a night hour"
    print(f"  [ok] smart persistence ties plain at 24h (sky moves {drift:.1%}), "
          f"daylight keeps {frac:.0%}")


def main():
    print("Self-check on synthetic data (numbers are meaningless)...\n")
    frame = fake_frame(400, seed=1)
    load = frame["load_mw"]

    check_no_load_leakage(load)
    check_local_time(load)
    check_feature_table(load)
    check_baseline_and_skill(frame)
    check_beats_naive(frame)
    check_weather_modes(load)
    check_netcdf_reader()
    check_mlp_scalers_and_determinism(load)
    check_same_rows(frame)
    check_land_mask()
    check_weights()
    check_bilinear()
    check_solar_geometry()
    check_clear_sky()
    check_cell_temperature()
    check_clear_sky_index()
    check_daylight_mask()
    check_solar_no_leakage()
    check_solar_modes()
    check_solar_baselines()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
