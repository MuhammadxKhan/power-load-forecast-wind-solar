"""
The ERA5 grid, kept as a grid.

data.py reduces the NetCDF to one number per hour by averaging the whole
bounding box. That box is a rectangle, not Germany: it takes in the North Sea,
the Baltic and parts of four neighbours, and it weights open water the same as
Berlin.

Keeping latitude and longitude lets the same files support:

  - a land mask, from a simplified outline of the German border
  - area weighting, correcting for grid cells narrowing towards the pole
  - population weighting, so the regions that actually use power count for more
  - series_at(lat, lon), bilinear interpolation to an arbitrary point

The last of those is what the solar model needs. Clear-sky irradiance depends on
latitude, so a national solar figure has to be built from per-location values
rather than from one national number.

One year is opened and reduced at a time, so memory stays at one year of grid
rather than six.
"""

import glob

import numpy as np
import pandas as pd

from .data import ERA5_GLOB

# A simplified outline of the German border, (lon, lat), closed by repeating the
# first point. Roughly 60 vertices against a 0.25 degree grid whose cells are
# about 28 km across, so the polygon is finer than the thing it is masking.
GERMANY_OUTLINE = np.array([
    (8.40, 55.00), (9.00, 54.85), (9.60, 54.83), (10.00, 54.60), (10.80, 54.40),
    (11.30, 54.00), (12.10, 54.20), (13.00, 54.40), (13.80, 54.10),
    (14.40, 53.90), (14.40, 53.30), (14.15, 52.90), (14.60, 52.50),
    (14.75, 52.07), (14.60, 51.80), (15.04, 51.30), (14.80, 50.87),
    (14.40, 50.90), (13.50, 50.70), (12.90, 50.40), (12.10, 50.30),
    (12.20, 50.00), (12.50, 49.90), (13.00, 49.30), (13.80, 48.90),
    (13.40, 48.60), (13.00, 48.30), (12.80, 48.00), (13.00, 47.70),
    (12.80, 47.50), (12.20, 47.70), (11.60, 47.60), (11.00, 47.40),
    (10.50, 47.55), (10.20, 47.40), (10.10, 47.30), (9.80, 47.55),
    (9.50, 47.55), (8.90, 47.65), (8.60, 47.80), (8.40, 47.70), (7.90, 47.60),
    (7.60, 47.60), (7.55, 47.90), (7.60, 48.30), (8.00, 48.90), (8.20, 49.00),
    (7.10, 49.10), (6.70, 49.20), (6.40, 49.50), (6.10, 49.50), (6.40, 49.80),
    (6.10, 50.10), (6.20, 50.50), (6.00, 50.75), (5.90, 51.05), (6.20, 51.40),
    (5.95, 51.80), (6.70, 51.90), (7.00, 52.40), (6.70, 52.60), (7.00, 53.30),
    (7.20, 53.50), (8.00, 53.70), (8.50, 53.90), (8.90, 54.00), (8.60, 54.40),
    (8.90, 54.90), (8.40, 55.00),
])

# Population in millions and an approximate centroid, per federal state, 2020.
# Sixteen public numbers are enough to weight the grid by where people are: a
# cell takes the density of the state whose centroid is nearest.
BUNDESLAENDER = {
    "Baden-Wuerttemberg":     (11.10, 48.60, 9.00),
    "Bayern":                 (13.14, 48.90, 11.40),
    "Berlin":                 (3.66, 52.52, 13.40),
    "Brandenburg":            (2.53, 52.40, 13.20),
    "Bremen":                 (0.68, 53.08, 8.80),
    "Hamburg":                (1.85, 53.55, 10.00),
    "Hessen":                 (6.29, 50.60, 9.00),
    "Mecklenburg-Vorpommern": (1.61, 53.80, 12.50),
    "Niedersachsen":          (8.00, 52.80, 9.30),
    "Nordrhein-Westfalen":    (17.93, 51.50, 7.50),
    "Rheinland-Pfalz":        (4.10, 49.90, 7.50),
    "Saarland":               (0.98, 49.40, 7.00),
    "Sachsen":                (4.06, 51.10, 13.20),
    "Sachsen-Anhalt":         (2.18, 51.90, 11.70),
    "Schleswig-Holstein":     (2.91, 54.20, 9.70),
    "Thueringen":             (2.12, 50.90, 11.00),
}

WEIGHTINGS = ("box", "land", "population")


class Grid:
    """One file's worth of ERA5: values[time, lat, lon], with its coordinates.

    Latitudes are stored ascending whichever way the file had them, so the
    interpolation does not have to care.
    """

    def __init__(self, times, lats, lons, values):
        order = np.argsort(lats)
        self.lats = np.asarray(lats, dtype=float)[order]
        self.lons = np.asarray(lons, dtype=float)
        self.values = np.asarray(values, dtype=float)[:, order, :]
        self.times = pd.DatetimeIndex(times)
        if self.times.tz is None:
            self.times = self.times.tz_localize("UTC")

    @property
    def shape(self):
        return self.values.shape

    def reduce(self, weights):
        """Weighted mean over the grid, one number per hour.

        `weights` is a (lat, lon) array; it is renormalised here so a mask with
        any number of live cells still averages to a mean rather than a sum.
        """
        w = np.asarray(weights, dtype=float)
        if w.shape != self.values.shape[1:]:
            raise ValueError(f"weights are {w.shape}, grid is {self.values.shape[1:]}")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights sum to zero - the mask kept no cells")
        flat = self.values.reshape(len(self.times), -1) @ w.reshape(-1)
        return pd.Series(flat / total, index=self.times)

    def at(self, lat, lon):
        """Bilinear interpolation to one point, one value per hour.

        Exact at a grid node, linear along a grid line, and refuses points
        outside the box rather than silently clamping to the edge.
        """
        if not (self.lats[0] <= lat <= self.lats[-1]):
            raise ValueError(f"lat {lat} outside {self.lats[0]}..{self.lats[-1]}")
        if not (self.lons[0] <= lon <= self.lons[-1]):
            raise ValueError(f"lon {lon} outside {self.lons[0]}..{self.lons[-1]}")

        i = int(np.clip(np.searchsorted(self.lats, lat) - 1, 0, len(self.lats) - 2))
        j = int(np.clip(np.searchsorted(self.lons, lon) - 1, 0, len(self.lons) - 2))

        dy = (lat - self.lats[i]) / (self.lats[i + 1] - self.lats[i])
        dx = (lon - self.lons[j]) / (self.lons[j + 1] - self.lons[j])

        v = self.values
        top = v[:, i, j] * (1 - dx) + v[:, i, j + 1] * dx
        bot = v[:, i + 1, j] * (1 - dx) + v[:, i + 1, j + 1] * dx
        return pd.Series(top * (1 - dy) + bot * dy, index=self.times)


# --------------------------------------------------------------------------
# masks and weights
# --------------------------------------------------------------------------
def inside_outline(lats, lons, outline=GERMANY_OUTLINE):
    """Boolean (lat, lon) mask: which cell centres fall inside the polygon."""
    from matplotlib.path import Path

    lon_g, lat_g = np.meshgrid(np.asarray(lons, float), np.asarray(lats, float))
    pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])
    return Path(outline).contains_points(pts).reshape(lon_g.shape)


def area_weights(lats, lons):
    """cos(latitude), the area a cell covers as the meridians converge.

    A 0.25 degree cell at 55N is about 12% narrower than one at 47N, so an
    unweighted mean over-counts the north.
    """
    w = np.cos(np.deg2rad(np.asarray(lats, float)))
    return np.repeat(w[:, None], len(lons), axis=1)


def population_weights(lats, lons):
    """Population density per cell, from the state whose centroid is nearest.

    Each state's population is spread evenly over the cells assigned to it, so a
    cell in North Rhine-Westphalia counts for far more than one in Mecklenburg.
    State resolution is coarse - it cannot see that a city is denser than the
    farmland around it - but it captures the west-heavy distribution that a flat
    average misses entirely.
    """
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    lon_g, lat_g = np.meshgrid(lons, lats)

    pops = np.array([v[0] for v in BUNDESLAENDER.values()])
    clat = np.array([v[1] for v in BUNDESLAENDER.values()])
    clon = np.array([v[2] for v in BUNDESLAENDER.values()])

    # nearest centroid, with longitude degrees shrunk by cos(lat) so the
    # distance is roughly metric rather than degrees-as-if-square
    scale = np.cos(np.deg2rad(lat_g))[..., None]
    d2 = ((lat_g[..., None] - clat) ** 2
          + ((lon_g[..., None] - clon) * scale) ** 2)
    nearest = d2.argmin(axis=-1)

    counts = np.bincount(nearest.ravel(), minlength=len(pops)).astype(float)
    counts[counts == 0] = 1.0
    return (pops / counts)[nearest]


def grid_weights(lats, lons, kind="population"):
    """The weight field for one of the three weightings, land-masked as needed."""
    if kind not in WEIGHTINGS:
        raise ValueError(f"kind must be one of {WEIGHTINGS}")

    area = area_weights(lats, lons)
    if kind == "box":
        return area

    mask = inside_outline(lats, lons)
    if not mask.any():
        raise ValueError("the land mask kept no cells - outline and grid disagree")
    if kind == "land":
        return area * mask
    return area * mask * population_weights(lats, lons)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def each_year(pattern=ERA5_GLOB):
    """Yield a Grid per NetCDF file, oldest first, in Celsius."""
    import xarray as xr

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"nothing matching {pattern}. Run  python -m src.download_era5  "
            "first (free Copernicus account needed).")

    for f in files:
        with xr.open_dataset(f) as ds:
            if "t2m" not in ds:
                raise KeyError(f"{f}: no 't2m' variable, found {list(ds.data_vars)}")
            da = ds["t2m"]
            tname = "valid_time" if "valid_time" in da.dims else "time"
            extra = [d for d in da.dims if d not in (tname, "latitude", "longitude")]
            if extra:
                da = da.isel({d: 0 for d in extra})
            da = da.transpose(tname, "latitude", "longitude")
            yield Grid(da[tname].values, da["latitude"].values,
                       da["longitude"].values, da.values - 273.15)


def national_temperature(kind="population", pattern=ERA5_GLOB):
    """Weighted national hourly temperature in Celsius, indexed by UTC."""
    parts, weights = [], None
    for g in each_year(pattern):
        if weights is None:
            weights = grid_weights(g.lats, g.lons, kind)
        parts.append(g.reduce(weights))

    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.rename(kind).rename_axis("timestamp")


def series_at(lat, lon, pattern=ERA5_GLOB):
    """Hourly temperature interpolated to one point, indexed by UTC."""
    parts = [g.at(lat, lon) for g in each_year(pattern)]
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.rename("temp_c").rename_axis("timestamp")


def all_weightings(pattern=ERA5_GLOB):
    """All three national series in one frame, so the ablation is one pass over
    the NetCDF rather than three."""
    cols, weights = {k: [] for k in WEIGHTINGS}, None
    for g in each_year(pattern):
        if weights is None:
            weights = {k: grid_weights(g.lats, g.lons, k) for k in WEIGHTINGS}
        for k in WEIGHTINGS:
            cols[k].append(g.reduce(weights[k]))

    out = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in cols.items()})
    return out[~out.index.duplicated(keep="first")].rename_axis("timestamp")
