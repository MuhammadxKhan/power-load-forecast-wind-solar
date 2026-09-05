"""
Solar geometry and a clear-sky model.

Solar generation splits into two parts that behave completely differently:

    generation = capacity x clear-sky irradiance x clear-sky index x derate
                            \\_ deterministic _/   \\_ cloud _/

The first part is astronomy. Where the sun is, and how much power it would
deliver through a cloudless atmosphere, follow from latitude, longitude and the
timestamp - no measurement, no forecast, no data at all. The second part, cloud,
is the only piece a weather model is needed for.

That is the opposite of the demand problem in the rest of this repo, where the
weather signal is small because demand history already carries it. Here the free
deterministic component is most of the answer.

Everything below is standard and none of it is fitted:

  solar_position    NOAA's algorithm, good to about 0.01 degrees
  clear_sky_ghi     Haurwitz, a one-parameter clear-sky model
  plane_of_array    isotropic sky (Liu and Jordan), which is where tilt and
                    azimuth enter
  cell_temperature  NOCT, and the linear efficiency loss above 25 C

Angles are degrees. Azimuth is measured clockwise from north, so 180 is south.
"""

import numpy as np
import pandas as pd

SOLAR_CONSTANT = 1361.0     # W/m2 at mean earth-sun distance

# A representative German rooftop array. Most domestic PV in Germany sits on a
# pitched roof at roughly 30 degrees, and installers point it south where the
# roof allows. A national fleet is a mixture of orientations, so this is the
# centre of a distribution rather than any particular array.
PANEL_TILT = 30.0
PANEL_AZIMUTH = 180.0
ALBEDO = 0.2               # ground reflectance, typical mixed land

NOCT = 45.0                # nominal operating cell temperature, C
GAMMA = -0.004             # power lost per degree above 25 C, /C
STC_IRRADIANCE = 1000.0    # W/m2, the rating condition


def _julian_day(index):
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    seconds = (idx.tz_localize(None) - pd.Timestamp("1970-01-01")).total_seconds()
    return seconds.to_numpy() / 86400.0 + 2440587.5


def solar_position(index, lat, lon):
    """Sun zenith, elevation and azimuth for a place and a set of times.

    NOAA's algorithm. The equation of time correction is what stops solar noon
    being clock noon: it swings by about half an hour across the year, which
    matters because a solar forecast that is 30 minutes out is wrong at exactly
    the hours generation is changing fastest.

    Returns a frame indexed like `index`, in degrees.
    """
    idx = pd.DatetimeIndex(index)
    jd = _julian_day(idx)
    t = (jd - 2451545.0) / 36525.0

    # geometric mean longitude and anomaly of the sun
    l0 = np.radians((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0)
    m = np.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    # equation of centre, then the apparent longitude
    c = (np.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + np.sin(2 * m) * (0.019993 - 0.000101 * t)
         + np.sin(3 * m) * 0.000289)
    omega = np.radians(125.04 - 1934.136 * t)
    lam = np.radians(np.degrees(l0) + c - 0.00569 - 0.00478 * np.sin(omega))

    # obliquity of the ecliptic, and the declination that follows from it
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = np.radians(eps0 + 0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(eps) * np.sin(lam))

    # equation of time, in minutes
    y = np.tan(eps / 2.0) ** 2
    eot = 4.0 * np.degrees(
        y * np.sin(2 * l0) - 2 * e * np.sin(m)
        + 4 * e * y * np.sin(m) * np.cos(2 * l0)
        - 0.5 * y * y * np.sin(4 * l0) - 1.25 * e * e * np.sin(2 * m))

    utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    minutes = utc.hour.to_numpy() * 60.0 + utc.minute.to_numpy() + utc.second.to_numpy() / 60.0
    true_solar = minutes + eot + 4.0 * lon
    hour_angle = np.radians((true_solar / 4.0 - 180.0 + 180.0) % 360.0 - 180.0)

    phi = np.radians(lat)
    cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    cos_z = np.clip(cos_z, -1.0, 1.0)
    zenith = np.degrees(np.arccos(cos_z))

    azimuth = (180.0 + np.degrees(np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(phi) - np.tan(decl) * np.cos(phi)))) % 360.0

    return pd.DataFrame({"zenith": zenith, "elevation": 90.0 - zenith,
                         "azimuth": azimuth, "declination": np.degrees(decl),
                         "eot_minutes": eot}, index=idx)


def is_daylight(zenith, horizon=90.0):
    """True where the sun is above the horizon.

    Solar output is exactly zero for 43% of the hours in the year, and every
    model predicts those correctly. Scoring them inflates every skill number, so
    the evaluation drops them.
    """
    return np.asarray(zenith) < horizon


def air_mass(zenith):
    """Kasten and Young: the path length through the atmosphere, 1.0 overhead.

    Grows to about 38 at the horizon, which is why the last hour of daylight
    delivers so little.
    """
    z = np.clip(np.asarray(zenith, dtype=float), 0.0, 90.0)
    cos_z = np.cos(np.radians(z))
    am = 1.0 / (cos_z + 0.50572 * (96.07995 - z) ** -1.6364)
    return np.where(z >= 90.0, np.inf, am)


def extraterrestrial(index):
    """Irradiance at the top of the atmosphere, W/m2.

    Varies by 3.3% across the year because the orbit is an ellipse - and the
    earth is closest to the sun in January, which is not the intuitive answer.
    """
    doy = pd.DatetimeIndex(index).dayofyear.to_numpy()
    return SOLAR_CONSTANT * (1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.25))


def clear_sky_ghi(zenith):
    """Haurwitz: global horizontal irradiance under a cloudless sky, W/m2.

    One parameter, no tuning, and about as good as anything more elaborate for a
    temperate climate. Peaks near 1,000 W/m2 with the sun overhead and falls to
    zero at the horizon.
    """
    cos_z = np.cos(np.radians(np.clip(np.asarray(zenith, dtype=float), 0.0, 90.0)))
    ghi = 1098.0 * cos_z * np.exp(-0.059 / np.maximum(cos_z, 1e-6))
    return np.where(np.asarray(zenith) >= 90.0, 0.0, np.maximum(ghi, 0.0))


def clear_sky_dni(index, zenith):
    """Direct beam under a cloudless sky, W/m2, by atmospheric attenuation.

    0.7 raised to the air mass is the standard rule of thumb for a clear
    atmosphere; the exponent softens it for the long slant paths near sunrise.

    Capped so the beam's horizontal component cannot exceed the Haurwitz global.
    Attenuation and Haurwitz are two independent models, and near sunrise the
    uncapped beam overshoots the global it is supposed to be part of, which would
    make the diffuse term negative. The cap binds only at low sun.
    """
    z = np.asarray(zenith, dtype=float)
    am = air_mass(z)
    dni = extraterrestrial(index) * 0.7 ** (np.where(np.isfinite(am), am, 0.0) ** 0.678)

    cos_z = np.cos(np.radians(np.clip(z, 0.0, 90.0)))
    ceiling = np.where(cos_z > 1e-6, clear_sky_ghi(z) / np.maximum(cos_z, 1e-6), 0.0)
    dni = np.minimum(dni, ceiling)
    return np.where(z >= 90.0, 0.0, np.maximum(dni, 0.0))


def plane_of_array(ghi, dni, zenith, azimuth, tilt=PANEL_TILT,
                   panel_azimuth=PANEL_AZIMUTH, albedo=ALBEDO):
    """Irradiance on a tilted panel, W/m2 - the isotropic sky model.

    Three terms: the beam, reduced by the angle between the sun and the panel
    normal; the diffuse sky, of which a panel tilted by `tilt` sees the fraction
    (1 + cos tilt) / 2; and ground reflection, which is the mirror of that.

    This is where tilt and azimuth earn their place. Under clear skies a
    south-facing 30 degree roof in Germany collects about a quarter more over a
    year than the same panels laid flat, and the gain is largest in winter when
    the sun is low. The real-world gain is smaller, because much of the German
    year is diffuse and diffuse light does not care which way a panel points.
    """
    z = np.radians(np.clip(np.asarray(zenith, dtype=float), 0.0, 90.0))
    beta = np.radians(tilt)
    delta_az = np.radians(np.asarray(azimuth, dtype=float) - panel_azimuth)

    # angle of incidence between the beam and the panel normal
    cos_aoi = (np.cos(z) * np.cos(beta)
               + np.sin(z) * np.sin(beta) * np.cos(delta_az))
    cos_aoi = np.maximum(cos_aoi, 0.0)

    ghi = np.asarray(ghi, dtype=float)
    dni = np.asarray(dni, dtype=float)
    dhi = np.maximum(ghi - dni * np.cos(z), 0.0)

    beam = dni * cos_aoi
    sky = dhi * (1.0 + np.cos(beta)) / 2.0
    ground = ghi * albedo * (1.0 - np.cos(beta)) / 2.0

    poa = beam + sky + ground
    return np.where(np.asarray(zenith) >= 90.0, 0.0, poa)


def cell_temperature(air_c, poa, noct=NOCT):
    """Panel temperature from air temperature and irradiance, Celsius.

    The NOCT model. Panels run well above air temperature in sun - 25 C air and
    full irradiance puts the cells near 56 C - and this is the whole reason the
    ERA5 temperature series is a solar feature and not just a demand feature.
    """
    return np.asarray(air_c, dtype=float) + (noct - 20.0) / 800.0 * np.asarray(poa, dtype=float)


def temperature_derate(cell_c, gamma=GAMMA):
    """Efficiency relative to the 25 C rating.

    Silicon loses about 0.4% of its output per degree, so the hottest hours of
    the year are not the most productive: a cell at 56 C delivers roughly 88% of
    its rated efficiency. It is why solar output peaks in June rather than in the
    hottest weeks of July and August.
    """
    return 1.0 + gamma * (np.asarray(cell_c, dtype=float) - 25.0)


def clear_sky_output(index, lat, lon, air_c=None, tilt=PANEL_TILT,
                     panel_azimuth=PANEL_AZIMUTH):
    """Everything above, in one pass, for one location.

    Returns the solar position, clear-sky irradiance on the horizontal and on
    the panel, and - if air temperature is given - the derated output a
    cloudless sky would produce, as a fraction of rated capacity.

    The last column is the deterministic ceiling: what the fleet would make with
    no cloud. Dividing actual generation by it gives the clear-sky index, which
    is the part of the problem that actually needs a weather forecast.
    """
    pos = solar_position(index, lat, lon)
    z, az = pos["zenith"].to_numpy(), pos["azimuth"].to_numpy()

    ghi = clear_sky_ghi(z)
    dni = clear_sky_dni(index, z)
    poa = plane_of_array(ghi, dni, z, az, tilt, panel_azimuth)

    out = pos.copy()
    out["cs_ghi"] = ghi
    out["cs_dni"] = dni
    out["cs_poa"] = poa
    out["daylight"] = is_daylight(z)

    if air_c is not None:
        air = pd.Series(air_c).reindex(out.index).to_numpy()
        cell = cell_temperature(air, poa)
        out["cell_c"] = cell
        out["derate"] = temperature_derate(cell)
        out["cs_output"] = poa / STC_IRRADIANCE * out["derate"].to_numpy()
    return out


def fleet_clear_sky(index, centroids, shares, air_c=None, tilt=PANEL_TILT,
                    panel_azimuth=PANEL_AZIMUTH):
    """Clear-sky output for a fleet spread over several places.

    Solar geometry depends on latitude, so a national figure is built from
    per-location values rather than from one point. `centroids` maps a name to
    (lat, lon); `shares` weights them, and generation share is the sensible
    weight because it is what the fleet actually produces.

    The shift is small but real. The four German control-zone centroids span
    3.4 degrees of latitude, about 25 minutes of midsummer daylight between the
    northernmost and the southernmost.

    Air temperature stays national. Latitude is a first-order effect on
    irradiance and costs nothing to resolve; temperature enters only through a
    derate of a few percent, so resolving it per zone would not pay for itself.
    """
    total = float(sum(shares[k] for k in centroids))
    if total <= 0:
        raise ValueError("fleet shares sum to zero")

    out = None
    for name, (lat, lon) in centroids.items():
        w = shares[name] / total
        part = clear_sky_output(index, lat, lon, air_c, tilt, panel_azimuth)
        cols = [c for c in part.columns if c != "daylight"]
        scaled = part[cols] * w
        out = scaled if out is None else out + scaled

    # daylight anywhere in the fleet, not on average
    out["daylight"] = False
    for lat, lon in centroids.values():
        out["daylight"] |= is_daylight(solar_position(index, lat, lon)["zenith"])
    return out


def clear_sky_index(generation, clear_sky, floor=1e-3):
    """Actual over clear-sky: the fraction of a cloudless sky that got through.

    Around 1 on a clear day, near 0 under thick cloud, and undefined at night -
    the denominator is zero, so those hours come back as NaN rather than as a
    number that would then be averaged into something meaningless.
    """
    gen = pd.Series(generation, dtype=float)
    cs = pd.Series(clear_sky, dtype=float).reindex(gen.index)
    return (gen / cs.where(cs > floor)).clip(upper=2.0)
