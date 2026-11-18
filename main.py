"""Forecast GTA VI's launch-day Twitch viewers, hour by hour, as five quantiles.

Method: an ensemble of two statistical forecasters, both fitted at run time
from the case's own `comparables` (past big launches' hourly Twitch curves).
No network, no fitted constants from outside the case file beyond the
bandwidths and shrinkage below, which were picked by leave-one-out backtest
on the kit's comparables and are documented in the README.

For every comparable launch we form the log-ratio curve
    r_h = ln(viewers_h / viewers_1),  h = 2..24,
and three features of its first hour: its scale ln(viewers_1), the launch
ramp ln(viewers_1 / pre-release baseline), and the pre-midnight ratio
ln(pre-midnight viewers / viewers_1).  The target case's features come from
its own history, so the forecast adapts to how launched hour 1 already is.

Component A (local linear): per hour, a ridge regression of r_h on the
standardised log scale, weighted by closeness in scale to the target, with
the target's scale clamped to the comparables' range so a record launch
never extrapolates the slope past the data.  Quantiles come from the
weighted residual distribution, widened by a factor of 1.1.

Component B (kernel quantiles): per hour, the weighted empirical quantiles
of r_h across comparables, weighted by a Gaussian kernel over all three
features, so launches that unlocked the same way count most.

The forecast is the geometric mean of the two components' quantiles,
smoothed over neighbouring hours.  Every parameter that depends on the
launches is estimated from the case file at run time; if a case carries too
few usable comparables, the forecast falls back to persistence (every hour
at hour 1's print, with a random-walk spread).

Interface: reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and
writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output), one JSON object
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for hours
2..24.  Standard library only, deterministic: two runs write equal bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))

# Gaussian-kernel bandwidths in log units, from the leave-one-out backtest.
BANDWIDTH = {"ramp": 2.0, "prer": 1.0, "lnp1": 1.5}
SCALE_BANDWIDTH = 1.5
WEIGHT_FLOOR = 0.02
RIDGE = 1.0
SPREAD_MULT = 1.1
FALLBACK_SIGMA = 0.15
MIN_COMPARABLES = 3
MIN_DAY_HOURS = 20
MAX_VIEWERS = 1.0e9


def parse_time(text):
    """An ISO-8601 timestamp as an aware UTC datetime."""
    moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def median(values):
    """The median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def pstdev(values, mean):
    """The population standard deviation around a given mean."""
    if not values:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def hourly_points(rows, key):
    """(timestamp, viewers) pairs from hourly rows, oldest first, viewers >= 1."""
    points = []
    for row in rows or []:
        try:
            when = parse_time(row[key])
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(viewers):
            points.append((when, max(1.0, viewers)))
    points.sort(key=lambda item: item[0])
    return points


def day_hours(points, day_start):
    """Whole-hour viewer map {1..24: viewers} for the UTC day at day_start."""
    hours = {}
    for when, viewers in points:
        offset = (when - day_start).total_seconds()
        hour = int(offset // 3600) + 1
        if 1 <= hour <= 24 and hour not in hours:
            hours[hour] = viewers
    return hours


def launch_features(points, day_start, p1):
    """Scale, launch ramp and pre-midnight ratio of a launch, from its points.

    The baseline is the median of the points 48 to 14 hours before the day
    starts (before any rolling midnight unlock); when that window is empty,
    every point more than 13 hours before the day start.  The pre-midnight
    level is the median of the 3 hours before the day starts.  A feature is
    None when its window holds no point.
    """
    baseline_rows = [v for t, v in points
                     if day_start - timedelta(hours=48) <= t < day_start - timedelta(hours=14)]
    if not baseline_rows:
        baseline_rows = [v for t, v in points if t < day_start - timedelta(hours=13)]
    baseline = median(baseline_rows) if baseline_rows else None
    pre_rows = [v for t, v in points
                if day_start - timedelta(hours=3) <= t < day_start]
    pre = median(pre_rows) if pre_rows else None
    return {
        "ramp": math.log(p1 / baseline) if baseline else None,
        "prer": math.log(pre / p1) if pre else None,
        "lnp1": math.log(p1),
    }


def prepare_comparables(case):
    """Usable past launches: log-ratio curves plus features, in name order."""
    target_name = str(case.get("game", {}).get("name", "")).strip().lower()
    launches = []
    raw = sorted(case.get("comparables") or [],
                 key=lambda c: (str(c.get("name", "")), str(c.get("day_start", ""))))
    for comp in raw:
        name = str(comp.get("name", ""))
        # The target game's own pre-release category history is not a launch.
        if target_name and target_name in name.strip().lower():
            continue
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        points = hourly_points(comp.get("hourly"), "from")
        hours = day_hours(points, day_start)
        if 1 not in hours or len(hours) < MIN_DAY_HOURS:
            continue
        p1 = hours[1]
        ratios = {h: math.log(hours[h] / p1) for h in HOURS if h in hours}
        if len(ratios) < MIN_DAY_HOURS - 1:
            continue
        launches.append({
            "ratios": ratios,
            "features": launch_features(points, day_start, p1),
        })
    return launches


def target_history_points(case):
    """The target's own hourly history points (whole-hour means preferred)."""
    rows = case.get("history_hourly")
    if rows:
        return hourly_points(rows, "from")
    # Fall back to the minute reads, averaged into whole hours.
    buckets = {}
    for row in case.get("history") or []:
        try:
            when = parse_time(row["at"])
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(viewers):
            hour = when.replace(minute=0, second=0, microsecond=0)
            buckets.setdefault(hour, []).append(max(1.0, viewers))
    return sorted((hour, sum(vals) / len(vals)) for hour, vals in buckets.items())


def kernel_weight(target, other, keys):
    """Gaussian-kernel similarity over the features both sides carry."""
    squared = 0.0
    used = 0
    for key in keys:
        a, b = target.get(key), other.get(key)
        if a is None or b is None:
            continue
        squared += ((a - b) / BANDWIDTH[key]) ** 2
        used += 1
    if not used:
        return WEIGHT_FLOOR
    return math.exp(-0.5 * squared) + WEIGHT_FLOOR


def weighted_quantile(pairs, tau):
    """A weighted quantile, linearly interpolated on the weighted CDF."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    previous_value, previous_share = ordered[0][0], 0.0
    running = 0.0
    for value, weight in ordered:
        running += weight
        share = running / total
        if share >= tau:
            if share <= previous_share:
                return value
            frac = (tau - previous_share) / (share - previous_share)
            return previous_value + frac * (value - previous_value)
        previous_value, previous_share = value, share
    return ordered[-1][0]


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting; None when singular."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r == col:
                continue
            factor = rows[r][col] / rows[col][col]
            for k in range(col, n + 1):
                rows[r][k] -= factor * rows[col][k]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def local_linear_component(p1, launches, weights, target_lnp1):
    """Per-hour ridge fit of log-ratios on log scale, residual quantiles."""
    scales = [launch["features"]["lnp1"] for launch in launches]
    centre = sum(scales) / len(scales)
    spread = pstdev(scales, centre) or 1.0
    lo, hi = min(scales), max(scales)
    target_x = (min(hi, max(lo, target_lnp1)) - centre) / spread
    forecast = {}
    for hour in HOURS:
        kept = [(launch, w) for launch, w in zip(launches, weights)
                if hour in launch["ratios"]]
        if not kept:
            forecast[hour] = [float(p1)] * 5
            continue
        values = [launch["ratios"][hour] for launch, _ in kept]
        ws = [w for _, w in kept]
        xh = [(launch["features"]["lnp1"] - centre) / spread for launch, _ in kept]
        # Ridge-weighted least squares for [intercept, slope].
        s_w = sum(ws)
        s_wx = sum(w * x for w, x in zip(ws, xh))
        s_wxx = sum(w * x * x for w, x in zip(ws, xh)) + RIDGE
        s_wy = sum(w * y for w, y in zip(ws, values))
        s_wxy = sum(w * x * y for w, x, y in zip(ws, xh, values))
        beta = solve_linear([[s_w, s_wx], [s_wx, s_wxx]], [s_wy, s_wxy])
        if beta is None:
            centre_value = sum(w * y for w, y in zip(ws, values)) / s_w
            fitted = [centre_value] * len(values)
            mu = centre_value
        else:
            fitted = [beta[0] + beta[1] * x for x in xh]
            mu = beta[0] + beta[1] * target_x
        mu = min(max(values), max(min(values), mu))
        residuals = [(y - f, w) for y, f, w in zip(values, fitted, ws)]
        med_resid = weighted_quantile(residuals, 0.5)
        centre_q = mu + med_resid * SPREAD_MULT
        quantiles = [centre_q + (weighted_quantile(residuals, tau) - med_resid) * SPREAD_MULT
                     for tau in LEVELS]
        forecast[hour] = [p1 * math.exp(q) for q in quantiles]
    return forecast


def kernel_component(p1, launches, weights):
    """Per-hour weighted empirical quantiles of the log-ratio curves."""
    forecast = {}
    for hour in HOURS:
        pairs = [(launch["ratios"][hour], w)
                 for launch, w in zip(launches, weights) if hour in launch["ratios"]]
        if not pairs:
            forecast[hour] = [float(p1)] * 5
            continue
        forecast[hour] = [p1 * math.exp(weighted_quantile(pairs, tau)) for tau in LEVELS]
    return forecast


def smooth_forecast(forecast):
    """Three-hour centred smoothing of log-quantiles; quantiles stay ordered."""
    smoothed = {}
    for hour in HOURS:
        window = [h for h in (hour - 1, hour, hour + 1) if h in forecast]
        row = []
        for k in range(5):
            mean_log = sum(math.log(forecast[h][k]) for h in window) / len(window)
            row.append(math.exp(mean_log))
        for k in range(1, 5):
            row[k] = max(row[k], row[k - 1])
        smoothed[hour] = row
    return smoothed


def persistence_forecast(p1):
    """Every hour at hour 1's print, with a widening random-walk spread."""
    return {h: [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)) for z in Z]
            for h in HOURS}


def round_half_up(value):
    """Round to 2 decimals, halves away from zero; whole numbers as ints."""
    scaled = value * 100.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    rounded = whole / 100.0
    return int(rounded) if float(rounded).is_integer() else rounded


def finalise_row(values):
    """Round, clamp to (0, 1e9] and keep the five quantiles non-decreasing."""
    row = [round_half_up(min(MAX_VIEWERS, max(0.01, v))) for v in values]
    for k in range(1, 5):
        row[k] = max(row[k], row[k - 1])
    return [min(MAX_VIEWERS, v) for v in row]


def forecast_case(case):
    """The 23-row forecast for one case dict."""
    known = case.get("known") or []
    first = next((row for row in known if row.get("hour") == 1), known[0] if known else None)
    try:
        p1 = max(1.0, float(first["viewers"]))
    except (TypeError, ValueError, KeyError):
        p1 = 1.0
    launches = prepare_comparables(case)
    if len(launches) < MIN_COMPARABLES:
        combined = persistence_forecast(p1)
    else:
        try:
            day_start = parse_time(case["day_start"])
        except (KeyError, TypeError, ValueError):
            day_start = None
        target_features = {"ramp": None, "prer": None, "lnp1": math.log(p1)}
        if day_start is not None:
            points = target_history_points(case)
            if points:
                target_features = launch_features(points, day_start, p1)
        scale_weights = [math.exp(-0.5 * ((target_features["lnp1"]
                                           - launch["features"]["lnp1"]) / SCALE_BANDWIDTH) ** 2)
                         + WEIGHT_FLOOR for launch in launches]
        kernel_weights = [kernel_weight(target_features, launch["features"],
                                        ("ramp", "prer", "lnp1")) for launch in launches]
        part_a = smooth_forecast(local_linear_component(
            p1, launches, scale_weights, target_features["lnp1"]))
        part_b = smooth_forecast(kernel_component(p1, launches, kernel_weights))
        combined = {}
        for hour in HOURS:
            combined[hour] = [math.exp((math.log(part_a[hour][k])
                                        + math.log(part_b[hour][k])) / 2.0) for k in range(5)]
    rows = []
    wanted = [row["hour"] for row in case.get("forecast") or []]
    if not wanted:
        wanted = HOURS
    for hour in wanted:
        values = combined.get(hour, [p1] * 5)
        rows.append({"hour": hour, "q": finalise_row(values)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
