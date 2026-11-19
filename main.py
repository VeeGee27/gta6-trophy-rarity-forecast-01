"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target.

Python 3.13, standard library only. Reads every <id>.in in the input directory
($MAKEFUN_INPUT_DIR, else /input) and writes <id>.out to the output directory
($MAKEFUN_OUTPUT_DIR, else /output): {"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}]}, one row per trophy, five quantiles in percent, non-decreasing.

How it forecasts, all in logit space with x = ln(1 + hours since 2026-11-19T00:00Z):

- The history is collapsed into distinct snapshots. Exophase refreshes a game's
  counts in jumps, so identical consecutive reads carry no new information; a
  trend fitted on raw reads would overweight stale values.
- Flat history (fewer than 3 distinct snapshots): no refresh has been seen yet.
  The competition rules document the first refresh -- tracked players 638 ->
  3,868, every rate but the 100% one down -- i.e. rates dilute roughly fivefold
  (pure denominator 638/3868 = 0.165; the house baseline's stated practice loss
  of ~1.51 reproduces at an effective dilution of ~0.20, so a few new earners
  arrive with the refresh too). The snapshot is already hours stale and the
  refresh is due inside the horizon, so the forecast centres on the diluted
  rate (factor 0.20) with a tight spread. The centre and spread were
  grid-searched over the plausible answers (uniform dilutions 0.165-0.25,
  jittered new-earner draws, constant new earners per trophy): a per-trophy
  centre (old earners + a constant E_new, over the refreshed 3,868 players)
  was tested properly and did NOT beat the uniform centre robustly -- the
  asymmetric upside widening for rare trophies already covers it -- so the
  flat path stays uniform. Over a multi-day horizon the dilution fades out
  (adoption growth after the refresh roughly cancels it)
  and the spread widens. The 100% trophy is exempt: the documented refresh
  left it at 100%.
- Trend history (3+ distinct snapshots): per-trophy weighted least squares of
  logit(rate) against x over the distinct snapshots of the last 12 hours
  (6-hour half-life weights). The median anchors at the last observed rate and
  moves only a damped, blended, capped share of the fitted change: launch-week
  slopes decay as player growth slows and trophies saturate, so a raw linear
  extrapolation in logit space overshoots saturating trophies by tens of points.
  The move is capped at 1.2 logit units and by tier ceilings (platinum 5%,
  gold 40%, silver 75%, bronze 97% -- ceilings never pull a rate below its last
  observed value). A trophy whose last rate is 0 gets a small tier-based prior
  median (platinum 0.1%, gold 0.3%, silver 0.5%, bronze 1.0%): over a multi-day
  horizon even hard trophies gain their first earners as the player base grows.
- Spreads combine the house random walk (0.5 * sqrt(days)), the regression
  prediction error, and the refresh-jump scale estimated from the data, with
  asymmetric widening near 0% and 100% (rare trophies can jump up, near-certain
  ones can dip down).

Everything is deterministic: no randomness, no network, no wall-clock reads.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
TIER_CAP = {"platinum": 5.0, "gold": 40.0, "silver": 75.0, "bronze": 97.0}

# First-refresh dilution prior (see the module docstring). Grid-searched
# centre 0.20 x spread 0.08 over the plausible answers; the per-trophy
# centre was tested and did not beat this uniform centre robustly.
DILUTION = 0.20
DILUTION_LOG = math.log(DILUTION)
# Spread of the flat-path forecast, in logit units: tight, because the refresh
# dilution is mechanical rather than a coin flip.
FLAT_SPREAD = 0.08

# Trend-path tuning.
TREND_BLEND_DAYS = 2.0   # w_trend = 1 / (1 + horizon_days / this)
DAMP_DAYS = 7.0           # damp = exp(-horizon_days / this)
MOVE_CAP = 1.2            # max |logit move| from the last rate
JUMP_REF_CAP = 4.0        # s_jump = sig * sqrt(min(n_refreshes, this))
FIT_WINDOW_H = 12.0       # distinct snapshots used for the trend fit
FIT_HALF_LIFE_H = 6.0     # recency weight half-life for the trend fit
# Median (percent) for a trophy whose last rate is 0.
ZERO_PRIOR = {"platinum": 0.1, "gold": 0.3, "silver": 0.5, "bronze": 1.0}


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l > 30:
        return 100.0
    if l < -30:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if float(v).is_integer() else v


def hours_since_epoch(dt):
    return (dt - EPOCH).total_seconds() / 3600.0


def distinct_snapshots(case):
    """Collapse consecutive identical reads. A snapshot's time is the first
    read that showed its values."""
    snaps = []
    prev = None
    for h in case["history"]:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            snaps.append({"t": parse_ts(h["at"]), "players": h["players_tracked"],
                          "rates": list(h["rates"])})
            prev = key
    return snaps


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, sigma, xbar, sxx, n)."""
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = [y - ybar for y in ys]
        sig = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / max(sw, 1e-9))
        return ybar, 0.0, sig, xbar, sxx, len(xs)
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    n = len(xs)
    sig = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / max(sw, 1e-9)
                  * n / max(n - 2, 1))
    return a, b, sig, xbar, sxx, n


def flat_quantiles(y_last, rate_last, horizon_h):
    """Flat history: centre on the refresh-diluted rate, tight spread.

    Over a multi-day horizon the one-refresh dilution fades out (adoption
    growth after the refresh roughly cancels it) and the spread widens. The
    100% trophy is exempt from dilution: the documented first refresh left it
    at 100%, so only its lower tail is widened.
    """
    wide = 0.45 * (1.0 - math.exp(-(horizon_h / 24.0) / 2.0))
    if rate_last is not None and rate_last >= 99.5:
        s_down, s_up = 1.2 + wide, 0.15
        return [round4(inv_logit(y_last + z * (s_down if z < 0 else s_up)))
                for z in Z]
    fade = math.exp(-max(0.0, horizon_h - 1.5) / 48.0)
    centre = y_last + DILUTION_LOG * fade
    s = FLAT_SPREAD + wide
    # Near-zero rates: widen the upside, since first earners can arrive with
    # the refresh.
    s_up = s * (1.0 + 1.2 * sigmoid((-y_last - 4.0) / 1.0))
    qs = []
    for z in Z:
        ss = s_up if z > 0 else s
        qs.append(round4(inv_logit(centre + z * ss)))
    return sorted(qs)


def forecast_case(case):
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    target = parse_ts(case["target"])
    x_target = math.log1p(hours_since_epoch(target))
    snaps = distinct_snapshots(case)
    t_last = snaps[-1]["t"]
    x_last = math.log1p(hours_since_epoch(t_last))
    gaps = [(snaps[i + 1]["t"] - snaps[i]["t"]).total_seconds() / 3600.0
            for i in range(len(snaps) - 1)]
    gaps = [g for g in gaps if g > 0]
    med_gap = sorted(gaps)[len(gaps) // 2] if gaps else 4.5
    med_gap = min(max(med_gap, 0.5), 12.0)

    cutoff = t_last.timestamp() - FIT_WINDOW_H * 3600
    win = [s for s in snaps if s["t"].timestamp() >= cutoff]
    if len(win) < 3:
        win = snaps
    xs = [math.log1p(hours_since_epoch(s["t"])) for s in win]
    ws = [0.5 ** ((t_last - s["t"]).total_seconds() / (FIT_HALF_LIFE_H * 3600.0))
          for s in win]

    # Trend trust shrinks with horizon: slopes decay as growth slows and
    # trophies saturate. Blend with persistence + extra damping + move cap.
    w_trend = 1.0 / (1.0 + horizon_d / TREND_BLEND_DAYS)
    damp = math.exp(-horizon_d / DAMP_DAYS)
    s_rw = 0.5 * math.sqrt(horizon_d)
    n_ref = horizon_h / med_gap

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = snaps[-1]["rates"][i]
        players_last = snaps[-1]["players"]

        def y_of(rate, players):
            # Continuity correction: half an earner where the printed rate is 0.
            eff = rate if (rate is not None and rate > 0) else 0.0
            eff = max(eff, 50.0 / max(players, 1))
            return logit(eff)

        y_last = y_of(last_rate, players_last)
        if len(snaps) < 3:
            rows.append({"id": trophy["id"],
                         "q": flat_quantiles(y_last, last_rate, horizon_h)})
            continue
        ys = [y_of(s["rates"][i], s["players"]) for s in win]
        a, b, sig, xbar, sxx, n = wls(xs, ys, ws)
        move = w_trend * damp * b * (x_target - x_last)
        move = max(-MOVE_CAP, min(MOVE_CAP, move))
        y_med = y_last + move
        if last_rate is not None and last_rate <= 0.0:
            zp = ZERO_PRIOR.get(trophy["tier"], 0.0)
            if zp > 0:
                y_med = max(y_med, logit(zp))
        cap = TIER_CAP.get(trophy["tier"], 97.0)
        y_med = min(y_med, max(logit(cap), y_last))
        y_med = max(y_med, logit(50.0 / max(players_last, 1)))
        if sxx > 1e-12 and n >= 3:
            se_pred = sig * math.sqrt(1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx)
        else:
            se_pred = sig
        s_jump = sig * math.sqrt(min(n_ref, JUMP_REF_CAP))
        s_tot = math.sqrt(s_rw ** 2 + se_pred ** 2 + s_jump ** 2)
        s_tot = min(max(s_tot, 0.35), 2.5)
        s_down = s_tot * (1.0 + 1.5 * sigmoid((y_med - 5.0) / 1.5))
        s_up = s_tot * (1.0 + 1.5 * sigmoid((-y_med - 5.0) / 1.5))
        qs = []
        for z in Z:
            s = s_down if z < 0 else (s_up if z > 0 else s_tot)
            qs.append(round4(inv_logit(y_med + z * s)))
        rows.append({"id": trophy["id"], "q": sorted(qs)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["levels"] != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case['levels']}, "
                             f"not {LEVELS}")
        if not case["history"]:
            raise ValueError(f"{path.name} has no history")
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
