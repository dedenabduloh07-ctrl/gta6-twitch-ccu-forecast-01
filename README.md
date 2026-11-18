# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dedenabduloh07-ctrl/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dedenabduloh07-ctrl/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for Grand Theft Auto VI's launch-day Twitch viewers: given
everything recorded up to the end of the first hour of release day
(19 November 2026, UTC), it predicts how many people will be watching live
on Twitch in each of the other 23 hours, as five quantiles per hour
(5%, 25%, 50%, 75%, 95%). It runs offline on the case file alone, standard
library only, and is deterministic: two runs write identical bytes.

## How it works

The program fits its forecast at run time from the case's own
`comparables` — past big launches' hourly Twitch curves. For each
comparable it forms the log-ratio curve `ln(viewers_h / viewers_1)` and
three features of its first hour: its scale, its launch ramp over the
pre-release baseline, and its pre-midnight ratio. The target's features
come from its own history, so the forecast adapts to how launched hour 1
already is. Two components are combined:

- **Local linear:** per hour, a ridge regression of the log-ratios on log
  scale, weighted by closeness in scale, with the target's scale clamped
  to the comparables' range; quantiles from the weighted residual
  distribution, widened by 1.1.
- **Kernel quantiles:** per hour, weighted empirical quantiles of the
  log-ratios, weighted by similarity over all three features, so launches
  that unlocked the same way count most.

The forecast is the geometric mean of the two components, smoothed over
neighbouring hours. If a case carries too few usable comparables, it falls
back to persistence (every hour at hour 1's print, widening spread).

## Backtest

Leave-one-out over the 37 launch comparables in the kit (each launch
forecast from the other 36, scored by the contest's pinball loss on the
log scale): mean loss **0.399**, median **0.255**, against the house
persistence baseline's **0.930** / **0.325** on the same launches. On the
16 rolling-midnight launches most like GTA VI's unlock, mean loss
**0.253** vs the baseline's **0.351**. Mean 90% coverage is 0.89, close to
the honest 0.90. Simultaneous global-unlock launches (tiny first hours,
explosive spikes) remain the hardest cases; GTA VI unlocks at rolling
local midnight, like the launches the model handles best.

## Validation

Checked locally on the kit's four practice cases: every output has exactly
23 rows (hours 2–24), five finite, non-decreasing quantiles per row within
(0, 1e9], under the 65,536-byte cap; two runs on the same input write
byte-identical files; all four cases run in under half a second total,
far inside the 8-second per-case cap.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 /bot/main.py` with the case in `/input` and
expects `/output/<id>.out`. Each output is one JSON object,
`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours
2 to 24.

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
