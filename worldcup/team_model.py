"""
team_model.py — Per-team statistical model for the markets the bookmakers DON'T
price (offsides, fouls) and team-vs-team comparisons.

Uses the per-match rates in data/team_stats.csv (StatsBomb WC18/22) + optionally
data/team_stats_extra.csv (gap teams from FBref/results). For each match we compute
an opponent-adjusted expected count for each stat and turn it into the platform's
question types. These are MODEL rows (source="teamstat"); where a sharp market also
covers the same question (corners, cards) the de-dup in predict_wc keeps the sharp
number — so team stats only "win" on the unpriced markets.

Matchup adjustment (classic for/against decomposition):
    expected_home_X = home.X_for × (away.X_against / league_baseline_X)
"""
from __future__ import annotations
import csv
import os
import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# tournament baselines (per team per match) — same priors as fetch_statsbomb.py
BASE = dict(goals=1.35, shots=11.5, sot=4.0, corners=4.8, offsides=1.8,
            fouls=11.5, cards=2.0, pens=0.10)

# Odds-API team name -> name used in the stats tables (StatsBomb spelling)
NAME_TO_STATS = {"USA": "United States"}

# StatsBomb logs more "Foul Committed" events than the official foul stat the platform
# settles on (~14/team vs official ~11). Scale SB foul rates to the official scale.
# Gap teams (team_stats_extra) already use the official-scale prior, so leave them.
FOUL_SB_SCALE = 0.82


# a team_stats_extra row is "prior-only" (goals real, rest = priors) if its prop rates
# exactly equal the prior constants — used to prefer real event data when both exist.
_PRIOR_MARK = {"offsides_for": BASE["offsides"], "fouls_for": BASE["fouls"],
               "corners_for": BASE["corners"]}


def _is_prior(r):
    try:
        return all(abs(float(r.get(k, -9)) - v) < 1e-6 for k, v in _PRIOR_MARK.items())
    except (ValueError, TypeError):
        return False


def load_rates():
    """Merge team_stats.csv (StatsBomb) + team_stats_extra.csv (API-Football/results).
    When a team appears in both, keep the better row: real event data beats prior-filled,
    then more matches wins."""
    merged = {}
    for fn, src in (("team_stats.csv", "sb"), ("team_stats_extra.csv", "extra")):
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            r["_src"] = src
            t = r["team"]
            if t not in merged:
                merged[t] = r
                continue
            cur = merged[t]
            cand_key = (not _is_prior(r), float(r.get("n_matches", 0) or 0))
            cur_key = (not _is_prior(cur), float(cur.get("n_matches", 0) or 0))
            if cand_key > cur_key:
                merged[t] = r
    return merged


def _rate(rates, team, stat, side):
    r = rates.get(NAME_TO_STATS.get(team, team)) or rates.get(team)
    if not r:
        return BASE[stat]
    try:
        v = float(r[f"{stat}_{side}"])
        if not (np.isfinite(v) and v > 0):
            return BASE[stat]
        if stat == "fouls" and r.get("_src") == "sb":
            v *= FOUL_SB_SCALE
        return v
    except (KeyError, ValueError, TypeError):
        return BASE[stat]


def expected(rates, stat, home, away):
    """Opponent-adjusted expected count of `stat` for home and away this match."""
    hf, ha = _rate(rates, home, stat, "for"), _rate(rates, home, stat, "against")
    af, aa = _rate(rates, away, stat, "for"), _rate(rates, away, stat, "against")
    # dampened opponent adjustment (sqrt) — full multiplicative over-amplifies when
    # both teams are above average (e.g. inflated foul/shot totals).
    h = hf * (aa / BASE[stat]) ** 0.5
    a = af * (ha / BASE[stat]) ** 0.5
    return max(0.05, h), max(0.05, a)


def _p_ge(lam, k):                       # P(X >= k)
    return float(1 - poisson.cdf(k - 1, lam))


def _p_over(lam, line):                  # P(X > line), line is x.5
    return float(1 - poisson.cdf(int(np.floor(line)), lam))


def _p_more(lh, la, kmax=40):            # P(H > A), independent Poissons
    H = poisson.pmf(np.arange(kmax), lh)
    A = poisson.pmf(np.arange(kmax), la)
    return float(np.tril(np.outer(H, A), -1).sum())


def team_questions(rates, home, away, weather_adj=1.0):
    """Return list of (market, question, prob, note) for the team-stat markets.

    weather_adj (<=1) scales goal-creation events (shots/SoT/corners) only — those
    rates come from StatsBomb and don't know this match's conditions. Market-derived
    markets and offsides/fouls/cards are left alone (the market already prices weather;
    offsides/fouls aren't meaningfully weather-driven)."""
    out = []
    wx = float(weather_adj) if weather_adj else 1.0

    def q(mkt, question, prob, note=""):
        if prob is not None and np.isfinite(prob):
            out.append((mkt, question, float(np.clip(prob, 0.03, 0.97)), note))

    # ---- OFFSIDES (no bookmaker market — pure model value) ----
    ho, ao = expected(rates, "offsides", home, away)
    q("Offsides", f"Will {home} be caught offside 2 or more times?", _p_ge(ho, 2))
    q("Offsides", f"Will {away} be caught offside 2 or more times?", _p_ge(ao, 2))
    for L in (2.5, 3.5, 4.5):
        q("Offsides", f"Will there be over {L} total offsides?", _p_over(ho + ao, L))
    q("Offsides", f"Will {home} be caught offside more often than {away}?", _p_more(ho, ao))

    # ---- FOULS (no bookmaker market) ----
    hf, af = expected(rates, "fouls", home, away)
    for L in (20.5, 22.5, 24.5):
        q("Fouls", f"Will there be over {L} total fouls?", _p_over(hf + af, L))
    q("Fouls", f"Will {home} commit more fouls than {away}?", _p_more(hf, af))
    q("Fouls", f"Will {home} commit over 11.5 fouls?", _p_over(hf, 11.5))
    q("Fouls", f"Will {away} commit over 11.5 fouls?", _p_over(af, 11.5))

    # ---- SHOTS / SHOTS ON TARGET (weather-scaled) ----
    hs, a_s = expected(rates, "shots", home, away)
    hs, a_s = hs * wx, a_s * wx
    for L in (20.5, 24.5):
        q("Shots", f"Will there be over {L} total shots?", _p_over(hs + a_s, L))
    q("Shots", f"Will {home} have more shots than {away}?", _p_more(hs, a_s))
    hsot, asot = expected(rates, "sot", home, away)
    hsot, asot = hsot * wx, asot * wx
    for L in (7.5, 8.5):
        q("Shots", f"Will there be over {L} total shots on target?", _p_over(hsot + asot, L))
    q("Shots", f"Will {home} have more shots on target than {away}?", _p_more(hsot, asot))

    # ---- CORNERS (weather-scaled; sharp market wins via de-dup where present) ----
    hc, ac = expected(rates, "corners", home, away)
    hc, ac = hc * wx, ac * wx
    q("Corners", f"Will {home} win more corners than {away}?", _p_more(hc, ac))
    for L in (8.5, 9.5, 10.5):
        q("Corners", f"Will there be over {L} total corners?", _p_over(hc + ac, L))

    # ---- CARDS (referee-adjusted; sharp market usually wins via de-dup) ----
    from referee_model import card_factor
    rf, ref = card_factor(home, away)
    hcd, acd = expected(rates, "cards", home, away)
    hcd, acd = hcd * rf, acd * rf          # scale by referee strictness
    note = f"ref {ref} x{rf:.2f}" if ref and abs(rf - 1) > 0.01 else ""
    for L in (2.5, 3.5, 4.5):
        out.append(("Cards", f"Will there be over {L} total cards?",
                    float(np.clip(_p_over(hcd + acd, L), 0.03, 0.97)), note))
    q("Cards", f"Will {home} receive more cards than {away}?", _p_more(hcd, acd))

    return out
