# Analysis

The workings behind the summary in [README.md](README.md). Everything here comes
from `python ablations.py`, which regenerates all of it in one pass, with
gradient boosting on a single split so the rows are comparable.

Test period 2019-01-01 to 2020-09-30 throughout. Temperature is
population-weighted unless a table says otherwise.

---

## Demand: what each weather mode is worth

| mode | MAE (MW) | vs `none` |
|---|---:|---:|
| `none` | 1,200.6 | — |
| `lagged` | 1,197.9 | −2.7 |
| `noisy` (seed 0) | 1,175.5 | −25.1 |
| `perfect` | 1,163.1 | **−37.5** |

Perfect foreknowledge of temperature — an upper bound no weather model can reach
— buys 3.1%. That is the ceiling on what any improvement in temperature
forecasting could be worth to this model, and it is small because the demand
history already carries most of it.

`lagged` is worth almost nothing (−2.7 MW). Yesterday's temperature adds little
that the model has not already extracted from yesterday's demand.

---

## Demand: does the gain survive reseeding?

`noisy` draws one realisation of a synthetic error, so a single run is one draw
and not an uncertainty estimate. Across ten seeds:

| | |
|---|---:|
| mean | 1,171.6 |
| standard deviation | 6.0 |
| range | 1,164.0 to 1,181.9 |
| **spread** | **17.9** |
| mean gain against `none` | **29.0** |

The effect survives: the spread is smaller than the gain, but only by a factor
of 1.6. Seed 0, the default a single run reports, lands 1,175.5 — mid-pack, so
the headline number is representative rather than flattering.

Had the spread exceeded the gain, the single-window number would have been noise
reported as a finding.

---

## Demand: across rolling folds

Expanding-window folds, six-month test blocks, gradient boosting:

| fold | test period | `none` | `noisy` | delta |
|---|---|---:|---:|---:|
| 1 | 2019 H1 | 1,276.2 | 1,305.2 | **+29.0** |
| 2 | 2019 H2 | 1,024.1 | 967.7 | −56.4 |
| 3 | 2020 H1 | 1,301.4 | 1,295.6 | −5.8 |
| 4 | 2020 H2 | 959.8 | 859.0 | **−100.8** |

Mean −33.5 MW, helping in three folds of four, with a fold-to-fold range of
129.8 MW — around four times the mean effect. The two folds where it helps
clearly are both second halves of the year; the one where it hurts is a first
half. That is the seasonal result showing up in a different cut.

Folds share training data and demand is serially correlated, so three of four is
a stability signal rather than four independent trials. The proper test is
Diebold-Mariano on the paired errors, which is not done here.

---

## Demand: the effect is seasonal

MAE by calendar month, `none` against `noisy`:

| month | `none` | `noisy` | delta |
|---|---:|---:|---:|
| Jan | 1,214 | 1,417 | **+17%** |
| Feb | 1,031 | 1,015 | −2% |
| Mar | 1,428 | 1,441 | +1% |
| Apr | 1,578 | 1,516 | −4% |
| May | 1,395 | 1,383 | −1% |
| Jun | 1,254 | 1,230 | −2% |
| Jul | 922 | 785 | **−15%** |
| Aug | 1,061 | 901 | **−15%** |
| Sep | 970 | 939 | −3% |
| Oct | 1,042 | 1,083 | +4% |
| Nov | 1,065 | 1,062 | −0% |
| Dec | 1,379 | 1,271 | −8% |

**June–August: −107 MW, or −9.9%. November–March: +18 MW.**

Temperature is worth 15% in July and August and nothing across winter on net.
January runs 17% worse with weather, which sits inside the fold-to-fold spread
above rather than being a separate effect.

---

## Why weather adds so little on average

Temperature is 95% autocorrelated at 24 hours:

| lag | corr with now |
|---|---:|
| 1h | +0.995 |
| **24h** | **+0.949** |
| 168h | +0.807 |

The strongest feature is `lag_24h`. Yesterday's demand already contains
yesterday's weather, and yesterday's weather is 95% of today's, so an explicit
temperature feature largely re-delivers what the model has. On daily means:

```
corr(load today, temp today)      = -0.354
corr(load today, temp yesterday)  = -0.355
```

Yesterday's temperature predicts today's demand as well as today's does.

In winter the calendar carries the rest — it is cold most days, so "January at
18:00" is most of the answer. In summer the calendar cannot tell a cool August
from a hot one, so temperature earns its keep. Cooling degree hours are also
active in only 7.2% of hours against 70.4% for heating, which is the thin right
arm of the V in the README's scatter plot.

---

## Reducing the grid: what the geography is worth

ERA5 is a 33 × 41 grid at 0.25°. Collapsing it three ways, `perfect` mode:

| weighting | cells | MAE (MW) | vs `box` |
|---|---:|---:|---:|
| `box` — the whole rectangle | 1,353 | 1,179.6 | — |
| `land` — masked to the German outline | 740 | 1,175.7 | −3.9 |
| `population` — land, weighted by people | 740 | 1,163.1 | **−16.5** |

1.4%, and worth having, but the more useful consequence is what it does to the
weather result itself. Under the unweighted box the ceiling on temperature was
2.5% and `lagged` was actively harmful; population-weighted it is 3.1% and
`lagged` is mildly positive. **The crude national average was understating what
weather is worth**, which is a reason to be careful reading any weather-value
study that reduces a grid without saying how.

The mask is checkable against a number from outside the repo. Germany is
357,000 km²; summing the area of the cells the outline keeps gives 357,451, or
0.13% out. `selfcheck.py` asserts it stays within 10%.

---

## Solar: what each mode is worth

Capacity factor, daylight hours, gradient boosting.

**With generation history:**

| mode | MAE (cf) | vs `none` |
|---|---:|---:|
| `none` | 0.0438 | — |
| `clearsky` | 0.0436 | −0.5% |
| `lagged` | 0.0435 | −0.7% |
| `perfect` | 0.0424 | **−3.2%** |

**Without it** — the same table with every lag and rolling feature dropped, which
is the position of an asset that has no production record:

| mode | MAE (cf) | vs `none` |
|---|---:|---:|
| `none` | 0.0545 | — |
| `clearsky` | 0.0539 | −1.1% |
| `lagged` | 0.0483 | **−11.4%** |
| `perfect` | 0.0443 | **−18.7%** |

Two things fall out of the pair.

**Explicit clear-sky physics is worth about 1%, either way.** That is not because
the geometry does not matter — it is most of the signal — but because a
gradient booster given hour-of-day and day-of-year re-derives it from three
years of data. The physics is a reparameterisation of the calendar, not new
information. It would stop being so the moment the model had to generalise to a
latitude it had never seen, which is exactly the asset-yield problem.

**Weather is worth six times more without history than with it**: 18.7% against
3.2%. Every channel that is genuinely exogenous — cloud, temperature — pays far
better when there is no lagged observation quietly carrying it.

---

## Solar: why night is dropped

Solar output is exactly zero for 47% of the test period, and every model predicts
those hours correctly. Scored over all hours the best model reports **0.0240**
instead of **0.0425** — a 44% improvement delivered entirely by the planet
rotating.

Clear-sky persistence — yesterday's cloudiness on today's sky — ties plain
persistence at 0.0444. The reason is measurable: over 24 hours the clear-sky
irradiance at a given hour moves **0.56%** of the daylight mean. Knowing exactly
where the sun will be adds nothing that yesterday had not already said. The
benchmark earns its keep at multi-day horizons, not this one.

---

## Bias is almost entirely COVID

The +258 MW annual bias on the gradient boosting model is not a standing
tendency to over-forecast. Split by period:

| period | gbm | MLP | ENTSO-E |
|---|---:|---:|---:|
| 2019 (full year) | **+94** | +109 | −1,334 |
| 2020 Jan–Feb | **−27** | +115 | −130 |
| **2020 Mar–Jun** | **+919** | +1,164 | **+1,294** |
| 2020 Jul–Sep | +218 | +374 | −562 |

Under normal conditions the model is close to unbiased: +94 MW over a full year
is 0.17% of mean demand (54,729 MW). Essentially all of the annual bias comes
from March–June 2020, when demand fell and nothing trained on 2015–2018 could
have known. The TSOs' own forecast over-predicted by more in the same window
(+1,294 MW) despite a much shorter horizon, which is worth knowing before
treating any single bias number as a property of a model rather than of a period.

---

## What was checked

`python selfcheck.py` — 24 assertions, synthetic data, no network, run in CI on
every push.

- **No lookahead, tested rather than asserted.** One value in the load series is
  spiked, the features are rebuilt, and nothing within the next 24 hours may
  move. The comparison is `>=`, not `>`, so a feature reading the value *at* the
  poked hour fails too. The same test runs over all four solar modes.
- **Both weather directions.** `lagged` must not react inside 24h; `perfect`
  *must* react at the target hour, or the two modes could quietly collapse into
  each other.
- **The clear-sky columns must not react at all** to a change in generation. They
  are astronomy and know nothing about what the fleet produced.
- **Solar physics against outside numbers.** Axial tilt 23.44°, equation of time
  −14 to +16 minutes, noon elevation at Berlin on both solstices, solar noon 54
  minutes ahead of noon UTC at 13.4°E. Azimuth is checked at minute resolution,
  because on an hourly grid the sample nearest solar noon is several degrees away
  from it.
- **A panel at zero tilt collects exactly the global horizontal irradiance.** This
  one caught a real fault: the beam and global clear-sky models are independent,
  and near sunrise the uncapped beam exceeded the global it is part of, which
  would have driven the diffuse term negative.
- **The land mask against Germany's area**, 357,451 km² against 357,000.
- **Bilinear interpolation reproduces a plane exactly**, from either latitude
  order, and refuses points outside the grid rather than clamping.
- **Fitting is independent of the test set.** Wrecking the test period by 7.5×
  and refitting leaves the trained MLP bit-identical.
- **Same rows for every model**, and the check that verifies this is itself
  tested against a deliberately mismatched set.
- **Calendar features are Europe/Berlin, not UTC**, and the holiday table matches
  the `holidays` package 55/55 for German national holidays 2015–2020.

---

## Limitations

- **No forecast-origin structure.** A single assumed midnight cutoff, so there is
  no real lead-time verification. That needs issue time, valid time and lead time
  carried explicitly, with the same valid hour forecast from several origins. It
  also means error-by-hour cannot separate "forecast decays with horizon" from
  "afternoon load is harder".
- **The ENTSO-E comparison is not like-for-like.** Under Regulation 543/2013 the
  first day-ahead load forecast is published at least two hours before gate
  closure, around 10:00 on D-1 for Germany. This model assumes midnight, so it
  has roughly fourteen more hours of demand data. OPSD keeps target timestamps
  but no forecast vintage, so a given value may be a later revision. Reported
  because it is the right thing to measure against, not as a win.
- **ERA5 is reanalysis, and `noisy` understates its own uncertainty.** The
  synthetic error is independent hour to hour, so `temp_roll_mean_24h` averages
  it from 1.00 to 0.21 °C (measured; 1/√24) where real forecast error is
  autocorrelated and does not. AR(1) error at φ=0.95 and the same marginal σ
  barely moves the mean gain but roughly doubles the spread across draws. An
  archived operational forecast is the fix.
- **There is no perfect-irradiance mode for solar.** Only the temperature channel
  can be bounded from what is here; bounding cloud needs archived forecast or
  observed irradiance. That is the same gap that keeps wind out of this repo —
  wind power goes as roughly the cube of wind speed, and temperature says
  nothing useful about it.
- **Zone capacity factors use a trailing-year peak as the denominator.** OPSD
  publishes generation per control zone but not capacity per zone. Peak
  generation tracks installed capacity but sits below it, so zone figures are
  comparable with each other and with their own history, and are not capacity
  factors in the sense the national column is.
- **Population weighting is at federal-state resolution.** Sixteen centroids and
  sixteen population figures, assigned nearest-first. It captures the west-heavy
  distribution a flat average misses; it cannot see that a city is denser than
  the farmland around it.
- **One fixed panel geometry for the whole fleet**, 30° facing south. The real
  installed base is a mixture of orientations, which is part of why actual output
  reaches only about half the clear-sky ceiling on a clear day.
- **Fold results are not independent.** Folds share training data and load is
  serially correlated, so three of four is a weak stability signal. A
  Diebold-Mariano test on paired errors would be the proper check.
- **The test period contains COVID.** No model trained on 2015–2018 was going to
  handle spring 2020.
