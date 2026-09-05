"""
Getting data in. Demand from OPSD, temperature from ERA5. No features here.

Two targets come out of the OPSD file.

Demand: load_mw, plus benchmark_mw, OPSD's aggregation of the ENTSO-E
Transparency day-ahead load forecast. Under Regulation 543/2013 that forecast is
published at least two hours before gate closure, around 10:00 on D-1 for
Germany, so its information cutoff is earlier than this model's assumed midnight.

Solar: generation, installed capacity, and the four TSO control zones. The
German fleet grew from 37 GW to 50 GW over these six years, so the modelling
target is capacity factor rather than MW - generation in MW is not stationary
and a model fitted on 2015 would under-predict 2020 by a third.

Temperature is ERA5, ECMWF's reanalysis: their best after-the-fact estimate of
what the weather was, hourly on a 0.25 degree grid.

Germany only. Adding a country needs a column name, a timezone and a holiday
list, all three.
"""

import glob
import os

import numpy as np
import pandas as pd

# Pinned to a dated package, not /latest/ - OPSD republishes, and a moving URL
# would make the committed numbers stop reproducing silently.
OPSD_VERSION = "2020-10-06"
OPSD_URL = (f"https://data.open-power-system-data.org/time_series/{OPSD_VERSION}/"
            "time_series_60min_singleindex.csv")
OPSD_CACHE = "opsd_60min.csv"        # the full 94MB file, gitignored
OPSD_EXTRACT = "data/de_hourly.csv"  # three columns of it, ~2MB, committed

ERA5_CACHE = "data/era5_temp_de.csv"            # unweighted box mean, committed
ERA5_WEIGHTED = "data/era5_temp_de_weighted.csv"  # box/land/population, committed
ERA5_GLOB = "data/era5_raw/*.nc"                # raw download, ~1GB, gitignored

# A rectangle loosely around Germany, not a border. It includes sea and parts of
# four neighbours; masking it to land is the first thing worth doing.
BBOX = {"north": 55.0, "south": 47.0, "west": 5.5, "east": 15.5}

ACTUAL_COL = "DE_load_actual_entsoe_transparency"
BENCH_COL = "DE_load_forecast_entsoe_transparency"

SOLAR_EXTRACT = "data/de_solar_hourly.csv"   # ~2MB, committed
SOLAR_COLS = {
    "DE_solar_generation_actual": "solar_mw",
    "DE_solar_capacity": "capacity_mw",
    "DE_solar_profile": "opsd_profile",       # OPSD's own capacity factor
    "DE_50hertz_solar_generation_actual": "solar_50hertz_mw",
    "DE_amprion_solar_generation_actual": "solar_amprion_mw",
    "DE_tennet_solar_generation_actual": "solar_tennet_mw",
    "DE_transnetbw_solar_generation_actual": "solar_transnetbw_mw",
}

# Approximate centroids of the four German TSO control zones. Clear-sky
# irradiance depends on latitude, so a zone needs its own coordinates: 50Hertz
# in the east sits about 3.4 degrees north of TransnetBW in the south-west,
# which is roughly 25 minutes of daylight at midsummer.
ZONE_CENTROIDS = {
    "50hertz": (52.0, 12.5),
    "amprion": (50.5, 7.5),
    "tennet": (51.5, 9.8),
    "transnetbw": (48.6, 9.0),
}


# --------------------------------------------------------------------------
# demand
# --------------------------------------------------------------------------
def load_frame():
    """Actual German load and the published benchmark, hourly, indexed by UTC."""
    # The committed extract is three columns of the full OPSD file, so cloning
    # is enough to run this. The download path below is what generated it, kept
    # so the extract is reproducible rather than a magic artefact.
    if os.path.exists(OPSD_EXTRACT):
        df = pd.read_csv(OPSD_EXTRACT, parse_dates=["utc_timestamp"])
        df = df.set_index("utc_timestamp")
    else:
        if not os.path.exists(OPSD_CACHE):
            print(f"Downloading OPSD {OPSD_VERSION} (~94MB, one-off)...")
            pd.read_csv(OPSD_URL, low_memory=False).to_csv(OPSD_CACHE, index=False)
            print(f"Cached to {OPSD_CACHE}")
        df = pd.read_csv(OPSD_CACHE, usecols=["utc_timestamp", ACTUAL_COL, BENCH_COL],
                         parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
        df = df.rename(columns={ACTUAL_COL: "load_mw", BENCH_COL: "benchmark_mw"})
        os.makedirs(os.path.dirname(OPSD_EXTRACT), exist_ok=True)
        df.to_csv(OPSD_EXTRACT)
        print(f"Wrote {OPSD_EXTRACT} - commit this and nobody needs the download")

    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")

    s = df["load_mw"]
    df = df.loc[s.first_valid_index():s.last_valid_index()]

    # every hour must exist, or "168 rows back" stops meaning "168 hours back"
    # and every lag misaligns
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC"))

    missing = int(df["load_mw"].isna().sum())
    if missing:
        print(f"{missing} missing load hours ({missing / len(df):.3%}) - "
              "filling from earlier values")
    df["load_mw"] = _fill_gaps(df["load_mw"])

    # The benchmark is deliberately not filled: that would invent forecast values
    # nobody published and then score models against them. evaluate.py drops the
    # column if the scored window has holes, and says so.
    gaps = int(df["benchmark_mw"].isna().sum())
    if gaps:
        print(f"{gaps} missing benchmark hours ({gaps / len(df):.3%}) - left as NaN")

    df.index.name = "timestamp"
    return df


def _fill_gaps(s):
    """Fill interior gaps forward only; a leading gap is filled backwards.

    Not interpolate(), which fills from both sides and would carry information
    backwards from the future. The pinned OPSD package has no missing hours once
    trimmed, so this changes nothing - it exists so the code does not depend on
    the data being clean.
    """
    return s.ffill().bfill()


# --------------------------------------------------------------------------
# solar
# --------------------------------------------------------------------------
def load_solar(index=None):
    """Hourly German solar generation, capacity and capacity factor, UTC.

    Columns: solar_mw, capacity_mw, capacity_factor, and one column per TSO
    control zone.

    OPSD reports installed capacity once a day, so it is forward-filled to
    hourly. Capacity factor is generation over capacity, which is the target the
    model actually fits: the fleet grew 36% across these six years and raw MW is
    not comparable end to end. OPSD ships its own profile column, and
    selfcheck.py requires the two to agree.
    """
    if os.path.exists(SOLAR_EXTRACT):
        df = pd.read_csv(SOLAR_EXTRACT, parse_dates=["utc_timestamp"])
        df = df.set_index("utc_timestamp")
    else:
        if not os.path.exists(OPSD_CACHE):
            print(f"Downloading OPSD {OPSD_VERSION} (~125MB, one-off)...")
            pd.read_csv(OPSD_URL, low_memory=False).to_csv(OPSD_CACHE, index=False)
            print(f"Cached to {OPSD_CACHE}")
        df = pd.read_csv(OPSD_CACHE, usecols=["utc_timestamp"] + list(SOLAR_COLS),
                         parse_dates=["utc_timestamp"], low_memory=False)
        df = df.set_index("utc_timestamp").rename(columns=SOLAR_COLS)
        os.makedirs(os.path.dirname(SOLAR_EXTRACT), exist_ok=True)
        df.to_csv(SOLAR_EXTRACT)
        print(f"Wrote {SOLAR_EXTRACT} - commit this and nobody needs the download")

    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")

    s = df["solar_mw"]
    df = df.loc[s.first_valid_index():s.last_valid_index()]
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC"))

    # capacity is a daily figure on an hourly index, so most hours are blank
    df["capacity_mw"] = df["capacity_mw"].ffill().bfill()

    gen_cols = [c for c in df.columns if c.endswith("_mw") and c != "capacity_mw"]
    missing = int(df[gen_cols].isna().any(axis=1).sum())
    if missing:
        print(f"{missing} hours miss a solar generation figure "
              f"({missing / len(df):.3%}) - filling from earlier values")
        for c in gen_cols:
            df[c] = _fill_gaps(df[c])

    df["capacity_factor"] = df["solar_mw"] / df["capacity_mw"]

    df.index.name = "timestamp"
    if index is not None:
        df = df.reindex(index)
    return df


def zone_capacity_factor(df, zone):
    """A capacity factor for one control zone, from its own rolling annual peak.

    OPSD publishes generation per zone but not capacity per zone, so there is no
    published denominator. The peak generation over a trailing year is a proxy
    for it: solar plants reach close to their rated output on the clearest days,
    so the annual maximum tracks installed capacity and grows with it.

    It is an estimate, not a measurement. Zone-level numbers are comparable with
    each other and with their own history; they are not capacity factors in the
    sense the national column is.
    """
    gen = df[f"solar_{zone}_mw"]
    peak = gen.rolling(24 * 365, min_periods=24 * 30).max()
    return (gen / peak.bfill()).clip(0, 1)


def fake_solar(index, seed=0):
    """Synthetic solar, so the checks run without the download.

    Zero at night, a smooth bell through the day, seasonal amplitude, cloud as a
    slow random wander, and capacity growing steadily. Meaningless numbers with
    the right shape.
    """
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(index)
    loc = idx.tz_convert("Europe/Berlin")
    hour = loc.hour.to_numpy() + loc.minute.to_numpy() / 60.0
    doy = loc.dayofyear.to_numpy()

    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    season = 0.35 + 0.65 * np.clip(np.sin((doy - 80) / 365 * 2 * np.pi) * 0.5 + 0.5, 0, 1)
    cloud = np.clip(pd.Series(rng.normal(0.75, 0.25, len(idx)))
                    .rolling(12, min_periods=1).mean().to_numpy(), 0.05, 1.0)

    capacity = np.linspace(37000, 50000, len(idx))
    cf = np.clip(daylight * season * cloud * 0.85, 0, 1)
    gen = cf * capacity

    out = pd.DataFrame({"solar_mw": gen, "capacity_mw": capacity,
                        "capacity_factor": cf}, index=idx)
    for k, z in enumerate(ZONE_CENTROIDS):
        out[f"solar_{z}_mw"] = gen * (0.15 + 0.1 * k) * (1 + rng.normal(0, 0.05, len(idx)))
    return out.rename_axis("timestamp")


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------
def load_temperature(index=None, weighting="population"):
    """National hourly 2m temperature in Celsius, indexed by UTC.

    Three ways of turning the grid into one number, all reduced from the same
    files by src/geo.py:

      "box"         unweighted mean of the whole rectangle - 45% of it is sea or
                    a neighbouring country
      "land"        area-weighted, masked to the German outline
      "population"  land, then weighted by where people are

    Reads the committed CSV if it is there, otherwise builds all three from the
    NetCDF in era5_raw/ and writes them out, so the slow path happens once.
    """
    from .geo import WEIGHTINGS
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")

    if os.path.exists(ERA5_WEIGHTED):
        df = pd.read_csv(ERA5_WEIGHTED, parse_dates=["timestamp"]).set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        s = df[weighting]
    elif weighting == "box" and os.path.exists(ERA5_CACHE):
        s = pd.read_csv(ERA5_CACHE, parse_dates=["timestamp"]).set_index("timestamp")["temp_c"]
        s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    else:
        if not sorted(glob.glob(ERA5_GLOB)):
            raise FileNotFoundError(
                f"no {ERA5_WEIGHTED} and nothing matching {ERA5_GLOB}.\n"
                "Run  python -m src.download_era5  first (free Copernicus account "
                "needed), or run without --weather.")
        from .geo import all_weightings
        df = all_weightings()
        df.to_csv(ERA5_WEIGHTED)
        print(f"Wrote {ERA5_WEIGHTED} ({len(df):,} hours) - the raw NetCDF isn't "
              "needed again")
        s = df[weighting]

    s.name = "temp_c"
    if index is not None:
        s = s.reindex(index)
        gaps = int(s.isna().sum())
        if gaps:
            print(f"{gaps} hours have no temperature - filling from earlier values")
            s = _fill_gaps(s)
    return s


def _from_netcdf(files):
    """Average the ERA5 grid down to one number per hour.

    Not xarray.open_mfdataset, which needs dask - not a dependency here, and
    download_era5.py writes one file per year. Reducing each file to a 1-D series
    as it opens keeps a year at 8,760 numbers rather than a full grid.
    """
    import xarray as xr

    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            if "t2m" not in ds:
                raise KeyError(
                    f"{f}: no 't2m' variable, found {list(ds.data_vars)}. "
                    "Refusing to guess which field is temperature.")
            # CDS has used both 'time' and 'valid_time' over the years
            tname = "valid_time" if "valid_time" in ds["t2m"].dims else "time"
            space = [d for d in ds["t2m"].dims if d != tname]
            parts.append(ds["t2m"].mean(dim=space).to_series())

    s = pd.concat(parts).sort_index()
    dupes = int(s.index.duplicated().sum())
    if dupes:
        # yearly files can overlap at the seam; a big count means the download is
        # wrong rather than the seam
        print(f"{dupes} duplicate ERA5 timestamps, keeping the first of each")
        s = s[~s.index.duplicated(keep="first")]

    s = s - 273.15                            # ERA5 ships Kelvin
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")  # ERA5 timestamps are UTC
    return s.sort_index()


# --------------------------------------------------------------------------
# synthetic, for the self-checks only
# --------------------------------------------------------------------------
def fake_frame(n_days=1500, seed=0):
    """Synthetic load, so the checks run without the download. The numbers are
    meaningless."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n_days * 24, freq="h", tz="UTC")
    loc = idx.tz_convert("Europe/Berlin")
    hour, dow, doy = loc.hour.to_numpy(), loc.dayofweek.to_numpy(), loc.dayofyear.to_numpy()

    daily = 8000 * np.sin((hour - 3) / 24 * 2 * np.pi) + 3000 * np.sin(hour / 12 * 2 * np.pi)
    weekly = np.where(dow >= 5, -6000, 0)
    yearly = 5000 * np.cos((doy - 15) / 365 * 2 * np.pi)
    load = 50000 + daily + weekly + yearly + rng.normal(0, 900, len(idx))

    # a fake benchmark that's decent but beatable, so the comparison machinery
    # has something to chew on
    return pd.DataFrame({"load_mw": load,
                         "benchmark_mw": load + rng.normal(0, 1800, len(idx))},
                        index=idx).rename_axis("timestamp")


def fake_temperature(index, seed=0):
    """Synthetic German-ish temperature: seasonal swing, daily swing, and a slow
    random wander so consecutive days correlate the way real weather does."""
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(index)
    loc = idx.tz_convert("Europe/Berlin")

    seasonal = 9.5 - 9.0 * np.cos((loc.dayofyear.to_numpy() - 20) / 365 * 2 * np.pi)
    diurnal = 3.5 * np.sin((loc.hour.to_numpy() - 9) / 24 * 2 * np.pi)
    wander = (pd.Series(rng.normal(0, 1.0, len(idx)))
              .rolling(72, min_periods=1).mean() * 6.0).to_numpy()

    return pd.Series(seasonal + diurnal + wander, index=idx,
                     name="temp_c").rename_axis("timestamp")
