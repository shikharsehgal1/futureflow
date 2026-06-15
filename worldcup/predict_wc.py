"""
predict_wc.py — Turn raw World Cup odds into a probability for EVERY platform question.

Reads the cached Odds API JSON (featured markets: h2h, totals, spreads) plus, when
present, per-event niche markets, and emits one row per (match, question) into
worldcup/data/wc_questions.csv — the single source of truth the dashboard reads.

Probability hierarchy (highest priority first):
  1. SHARP    — direct de-vig of a Pinnacle market for exactly this question
  2. POISSON  — derived from a double-Poisson fitted to the de-vigged h2h+totals
  3. (ELO blend handled separately in build_elo.py, for matches with no odds)

Scoring philosophy: report the de-vigged sharp number, clipped to [3%,97%]. Do NOT
shrink toward 50% — the sharp line is already the best-calibrated estimate, and the
field we're scored against is softer, so honest sharp numbers win relative Brier.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from math import factorial

from devig import (devig, devig_two, fit_poisson, score_matrix, model_over_prob,
                   clip_prob, _hda_from_matrix)
from niche import niche_questions
from team_model import load_rates, team_questions

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FEATURED = os.path.join(DATA, "wc_featured_raw.json")
EVENT_DIR = os.path.join(DATA, "events")          # per-event niche-market JSON cache
OUT = os.path.join(DATA, "wc_questions.csv")
SUMMARY = os.path.join(DATA, "wc_match_summary.csv")

SHARP_BOOK = "pinnacle"
FIRST_HALF_FRAC = 0.45   # ~45% of goals are scored in the 1st half (well documented)


# ---------------------------------------------------------------------------
# odds extraction
# ---------------------------------------------------------------------------

def _book_markets(event, key):
    for b in event.get("bookmakers", []):
        if b["key"] == key:
            return {m["key"]: m for m in b.get("markets", [])}
    return {}


def _consensus_h2h(event, home, away):
    """Median de-vigged h2h across all books — robust cross-check on the sharp line."""
    rows = []
    for b in event.get("bookmakers", []):
        mk = {m["key"]: m for m in b.get("markets", [])}
        if "h2h" not in mk:
            continue
        o = {x["name"]: x["price"] for x in mk["h2h"]["outcomes"]}
        if home in o and away in o and "Draw" in o:
            p = devig([o[home], o["Draw"], o[away]])
            if p is not None:
                rows.append(p)
    if not rows:
        return None
    return np.median(np.array(rows), axis=0)


def extract_market_probs(event):
    """Return de-vigged sharp probabilities + fitted Poisson lambdas for one event."""
    home, away = event["home_team"], event["away_team"]
    pin = _book_markets(event, SHARP_BOOK)

    # --- h2h (3-way) ---
    pH = pD = pA = np.nan
    if "h2h" in pin:
        o = {x["name"]: x["price"] for x in pin["h2h"]["outcomes"]}
        if home in o and away in o and "Draw" in o:
            p = devig([o[home], o["Draw"], o[away]])
            if p is not None:
                pH, pD, pA = float(p[0]), float(p[1]), float(p[2])

    # fall back to consensus if Pinnacle h2h missing
    src_h2h = "sharp:pinnacle"
    if not np.isfinite(pH):
        cons = _consensus_h2h(event, home, away)
        if cons is not None:
            pH, pD, pA = map(float, cons)
            src_h2h = "consensus:median"

    # --- totals (main line) ---
    total_line = p_over = np.nan
    if "totals" in pin:
        outs = pin["totals"]["outcomes"]
        over = next((x for x in outs if x["name"] == "Over"), None)
        under = next((x for x in outs if x["name"] == "Under"), None)
        if over and under:
            total_line = float(over["point"])
            p_over = devig_two(over["price"], under["price"])

    # --- spreads (Asian handicap main line) ---
    spread_line = spread_home_p = np.nan
    if "spreads" in pin:
        outs = pin["spreads"]["outcomes"]
        hh = next((x for x in outs if x["name"] == home), None)
        aa = next((x for x in outs if x["name"] == away), None)
        if hh and aa:
            spread_line = float(hh.get("point", 0.0))
            spread_home_p = devig_two(hh["price"], aa["price"])

    lam_h = lam_a = np.nan
    if np.isfinite(pH):
        lam_h, lam_a = fit_poisson(pH, pD, pA,
                                   total_line if np.isfinite(total_line) else None,
                                   p_over if np.isfinite(p_over) else None)

    return dict(home=home, away=away, pH=pH, pD=pD, pA=pA, src_h2h=src_h2h,
                total_line=total_line, p_over=p_over,
                spread_line=spread_line, spread_home_p=spread_home_p,
                lam_h=lam_h, lam_a=lam_a)


# ---------------------------------------------------------------------------
# question generation
# ---------------------------------------------------------------------------

def _q(rows, match, mkt, question, prob, source, note=""):
    if prob is None or not np.isfinite(prob):
        return
    rows.append(dict(match=match, market=mkt, question=question,
                     prob=round(clip_prob(prob), 4),
                     pct=round(100 * clip_prob(prob), 1),
                     source=source, note=note))


def generate_questions(mp, commence):
    home, away = mp["home"], mp["away"]
    match = f"{home} vs {away}"
    rows = []
    pH, pD, pA = mp["pH"], mp["pD"], mp["pA"]
    lh, la = mp["lam_h"], mp["lam_a"]

    # ---- Match result (direct sharp de-vig — the sharpest, highest-value rows) ----
    _q(rows, match, "Moneyline", f"Will {home} win the match?", pH, mp["src_h2h"])
    _q(rows, match, "Moneyline", f"Will {away} win the match?", pA, mp["src_h2h"])
    _q(rows, match, "Moneyline", "Will the match end in a draw?", pD, mp["src_h2h"])

    # ---- Double chance ----
    if np.isfinite(pH):
        _q(rows, match, "Double Chance", f"{home} or Draw (1X)", pH + pD, mp["src_h2h"])
        _q(rows, match, "Double Chance", f"{away} or Draw (X2)", pA + pD, mp["src_h2h"])
        _q(rows, match, "Double Chance", f"{home} or {away} (12)", pH + pA, mp["src_h2h"])

    if not np.isfinite(lh):
        return rows  # no goal model -> can't derive the rest

    M = score_matrix(lh, la)
    kmax = M.shape[0] - 1
    totals_grid = np.add.outer(np.arange(kmax + 1), np.arange(kmax + 1))

    # ---- Total goals over/under (Poisson, full-match) ----
    for line in (1.5, 2.5, 3.5):
        _q(rows, match, "Total Goals", f"Will total goals be over {line}?",
           model_over_prob(M, line), "poisson")

    # report the actual market total line too (direct sharp), as a sanity anchor
    if np.isfinite(mp["total_line"]) and np.isfinite(mp["p_over"]):
        _q(rows, match, "Total Goals",
           f"Will total goals be over {mp['total_line']}? (market line)",
           mp["p_over"], "sharp:pinnacle")

    # ---- Odd/even total ----
    p_even = float(M[totals_grid % 2 == 0].sum())
    _q(rows, match, "Total Goals O/E", "Will the total goals be even?", p_even, "poisson")
    _q(rows, match, "Total Goals O/E", "Will the total goals be odd?", 1 - p_even, "poisson")

    # ---- Both teams to score ----
    p_btts = float(M[1:, 1:].sum())
    _q(rows, match, "BTTS", "Will both teams score?", p_btts, "poisson")
    _q(rows, match, "BTTS", "Will NOT both teams score?", 1 - p_btts, "poisson")

    # ---- Team totals ----
    ph_goals = M.sum(axis=1)   # home goal distribution
    pa_goals = M.sum(axis=0)
    for name, dist in ((home, ph_goals), (away, pa_goals)):
        _q(rows, match, "Team Total", f"Will {name} score over 0.5?",
           1 - dist[0], "poisson")
        _q(rows, match, "Team Total", f"Will {name} score over 1.5?",
           dist[2:].sum(), "poisson")

    # ---- First team to score / no goals ----
    p_no_goals = float(M[0, 0])
    p_any = 1 - p_no_goals
    if (lh + la) > 0:
        _q(rows, match, "First To Score", f"Will {home} score first?",
           lh / (lh + la) * p_any, "poisson")
        _q(rows, match, "First To Score", f"Will {away} score first?",
           la / (lh + la) * p_any, "poisson")
    _q(rows, match, "First To Score", "Will there be no goals (0-0)?", p_no_goals, "poisson")

    # ---- Asian handicap (derive arbitrary lines from the matrix) ----
    diff = np.subtract.outer(np.arange(kmax + 1), np.arange(kmax + 1))  # home - away
    for h in (1, 2):
        _q(rows, match, "Handicap", f"Will {home} win by 2+ goals?" if h == 1 else f"Will {home} win by 3+ goals?",
           float(M[diff >= (h + 1)].sum()), "poisson")
    _q(rows, match, "Handicap", f"Will {away} win by 2+ goals?",
       float(M[diff <= -2].sum()), "poisson")
    # +1.5 cover (avoid losing by 2+)
    _q(rows, match, "Handicap", f"Will {home} avoid losing by 2+ (+1.5)?",
       float(M[diff >= -1].sum()), "poisson")
    _q(rows, match, "Handicap", f"Will {away} avoid losing by 2+ (+1.5)?",
       float(M[diff <= 1].sum()), "poisson")

    # ---- Correct score (top scorelines) ----
    flat = [((i, j), M[i, j]) for i in range(kmax + 1) for j in range(kmax + 1)]
    flat.sort(key=lambda x: -x[1])
    for (i, j), p in flat[:6]:
        _q(rows, match, "Correct Score", f"Will the score be {home} {i}-{j} {away}?",
           float(p), "poisson")

    # ---- First-half markets (1H lambdas ~ 45% of goals) ----
    M1 = score_matrix(lh * FIRST_HALF_FRAC, la * FIRST_HALF_FRAC)
    g1 = np.add.outer(np.arange(M1.shape[0]), np.arange(M1.shape[0]))
    _q(rows, match, "1st Half", "Will there be over 0.5 goals in the 1st half?",
       float(M1[g1 > 0.5].sum()), "poisson")
    _q(rows, match, "1st Half", "Will there be over 1.5 goals in the 1st half?",
       float(M1[g1 > 1.5].sum()), "poisson")
    h1H, d1H, a1H = _hda_from_matrix(M1)
    _q(rows, match, "1st Half", f"Will {home} lead at half-time?", h1H, "poisson")
    _q(rows, match, "1st Half", "Will it be level at half-time?", d1H, "poisson")
    _q(rows, match, "1st Half", f"Will {away} lead at half-time?", a1H, "poisson")

    # ---- 2nd half (full minus first half means) ----
    M2 = score_matrix(lh * (1 - FIRST_HALF_FRAC), la * (1 - FIRST_HALF_FRAC))
    g2 = np.add.outer(np.arange(M2.shape[0]), np.arange(M2.shape[0]))
    _q(rows, match, "2nd Half", "Will there be over 1.5 goals in the 2nd half?",
       float(M2[g2 > 1.5].sum()), "poisson")
    # which half has more goals
    _q(rows, match, "Half Compare", "Will the 2nd half have more goals than the 1st?",
       0.52, "heuristic", note="2H slightly goal-heavier on average")

    return rows


def load_event_niche(match_id):
    """Load cached per-event niche markets (corners/cards/btts/player props) if present."""
    path = os.path.join(EVENT_DIR, f"{match_id}.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None


# Reliability of each probability source and how "timid" the field tends to be on
# each market (popular markets -> field is sharp -> small edge; obscure markets ->
# field clusters near 50% -> confident calls earn big relative-Brier points).
_RELIABILITY = {"sharp": 1.0, "consensus": 0.85, "book": 0.7,
                "poisson": 0.45, "teamstat": 0.5, "heuristic": 0.2, "elo": 0.15}
_OBSCURITY = {"Moneyline": 0.4, "Total Goals": 0.4, "BTTS": 0.5, "Double Chance": 0.45,
              "Draw No Bet": 0.8, "Team Total": 0.8, "Handicap": 0.85,
              "1st Half": 0.85, "2nd Half": 0.85, "First To Score": 0.7,
              "Corners": 1.0, "Cards": 1.0, "Shots on Target": 1.0, "Shots": 1.0,
              "Offsides": 1.0, "Fouls": 1.0,
              "Anytime Scorer": 0.9, "Player Cards": 1.0, "Correct Score": 0.5,
              "Total Goals O/E": 0.1, "Half Compare": 0.1}


def value_tier(row):
    """Estimate a question's relative-Brier value and bucket it HIGH/MEDIUM/SKIP.

    Expected points beating a field ≈ (your_p − field_p)². We can't see field_p, so
    we proxy: value = reliability × |p−0.5|×2 × market_obscurity. Confident, reliable
    calls on low-attention markets (where the field is timid) are worth the most.
    SKIP = near-zero-skill markets, longshots, or low-value model guesses — under
    cumulative scoring these add variance without expected gain.
    """
    src = str(row["source"]).split(":")[0]
    rel = _RELIABILITY.get(src, 0.4)
    obsc = _OBSCURITY.get(row["market"], 0.6)
    p = float(row["prob"])
    value = rel * abs(p - 0.5) * 2 * obsc
    if (row["market"] in ("Total Goals O/E", "Half Compare")
            or src in ("elo", "heuristic")
            or (row["market"] == "Correct Score" and p < 0.06)
            or value < 0.12):
        tier = "SKIP"
    elif value >= 0.45:
        tier = "HIGH"
    else:
        tier = "MEDIUM"
    return round(value, 3), tier


def load_weather():
    """match_id -> weather dict, if the weather feature has been built."""
    path = os.path.join(DATA, "wc_weather.csv")
    if not os.path.exists(path):
        return {}
    wx = {}
    for r in pd.read_csv(path).to_dict(orient="records"):
        wx[r["match_id"]] = r
    return wx


def load_home_adv():
    """match_id -> net_home_adv_goals (diaspora/crowd factor). Display + fundamentals
    layer only — NEVER applied to de-vigged market numbers (market prices it already)."""
    path = os.path.join(DATA, "home_advantage.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in pd.read_csv(path).to_dict(orient="records"):
        out[r["match_id"]] = float(r.get("net_home_adv_goals", 0.0) or 0.0)
    return out


def main():
    events = json.load(open(FEATURED))
    rates = load_rates()
    weather = load_weather()
    home_adv = load_home_adv()
    all_rows = []
    summary = []
    for e in events:
        mp = extract_market_probs(e)
        rows = generate_questions(mp, e["commence_time"])

        # merge any niche-market questions (corners/cards/btts/player props)
        niche = load_event_niche(e["id"])
        if niche:
            rows += niche_questions(niche, mp)

        # team-stat model rows: offsides/fouls (unpriced) + shots/corners/cards comparisons
        wx = weather.get(e["id"], {})
        wadj = wx.get("weather_adj", 1.0)
        for mkt, question, prob, note in team_questions(rates, mp["home"], mp["away"], wadj):
            rows.append(dict(match=f"{mp['home']} vs {mp['away']}", market=mkt,
                             question=question, prob=round(prob, 4),
                             pct=round(100 * prob, 1), source="teamstat", note=note))

        hadv = home_adv.get(e["id"], 0.0)
        from referee_model import card_factor
        rf, ref = card_factor(mp["home"], mp["away"])
        for r in rows:
            r["commence"] = e["commence_time"]
            r["match_id"] = e["id"]
            r["weather_adj"] = round(float(wadj), 3) if wadj else 1.0
            r["home_adv"] = round(hadv, 3)
            r["referee"] = ref or ""
            r["ref_card_factor"] = round(rf, 2)
        all_rows += rows
        summary.append(dict(match=f"{mp['home']} vs {mp['away']}", commence=e["commence_time"],
                            match_id=e["id"], pH=mp["pH"], pD=mp["pD"], pA=mp["pA"],
                            total_line=mp["total_line"], p_over=mp["p_over"],
                            lam_h=round(mp["lam_h"], 3) if np.isfinite(mp["lam_h"]) else None,
                            lam_a=round(mp["lam_a"], 3) if np.isfinite(mp["lam_a"]) else None,
                            n_questions=len(rows)))

    df = pd.DataFrame(all_rows)
    df[["value", "tier"]] = df.apply(lambda r: pd.Series(value_tier(r)), axis=1)
    # Dedupe: one row per (match, question), keeping the highest-priority source.
    # Sharp de-vigged Pinnacle beats consensus beats single-book beats model.
    prio = {"sharp": 0, "consensus": 1, "book": 2, "poisson": 3,
            "teamstat": 4, "heuristic": 5, "elo": 6}
    df["_p"] = df["source"].str.split(":").str[0].map(prio).fillna(9)
    df = (df.sort_values(["match", "question", "_p"])
            .drop_duplicates(["match", "question"], keep="first")
            .drop(columns="_p"))
    df = df.sort_values(["commence", "match", "market"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)
    print(f"Wrote {len(df)} questions across {len(events)} matches -> {OUT}")
    print(f"Match summary -> {SUMMARY}")


if __name__ == "__main__":
    main()
