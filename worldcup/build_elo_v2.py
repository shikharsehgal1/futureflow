"""
build_elo_v2.py — Upgraded international-football fundamentals rating (SECOND signal).

This replaces the eloratings.net scrape used by build_elo.py with a from-scratch,
goal-based, recency-weighted Elo computed over the *entire* international match
history (martj42/international_results, 1872->present, includes 2024-2026 qualifiers
and friendlies). Building the rating ourselves lets us:

  - weight by goal margin (a 4-0 moves more than a 1-0),
  - weight by tournament importance (World Cup / qualifiers >> friendlies),
  - decay older matches so current form dominates (half-life ~2.5y),
  - handle neutral venues explicitly (no home bonus; World Cup is neutral).

Pipeline:
  1. Download results.csv (cached to data/intl_results.csv for ~1 day).
  2. Iterate chronologically, updating a World-Football-Elo-style rating per team.
     The K applied to each match is the base K * margin-of-victory multiplier *
     tournament-importance weight * recency weight (the recency weight is applied
     by *re-running* the whole history with a time-decayed effective K so that the
     final rating is dominated by recent form -- see _recency_weight()).
  3. Write data/intl_ratings.csv (team,rating,n_matches,last_played).
  4. Map each WC2026 team to its rating (alias map; reuse build_elo's NAME_ALIASES
     where relevant), convert the rating diff into supremacy/total goals, and fit
     the existing double-Poisson (devig.fit_poisson/score_matrix) to derive
     pH/pD/pA, over 1.5/2.5/3.5, BTTS and expected goals. Skip-and-log unmatched.
  5. Write data/wc_elo_questions.csv with the SAME schema build_elo.py emits
     (source="elo:model"): Moneyline (each team + draw), Total Goals over
     1.5/2.5/3.5, BTTS -- per match.

Plain numpy/scipy/pandas + requests; reuses devig.py wholesale. Does NOT touch
build_elo.py or any other existing .py.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pandas as pd
import requests

from devig import (fit_poisson, score_matrix, model_over_prob, clip_prob,
                   _hda_from_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SUMMARY = os.path.join(DATA, "wc_match_summary.csv")

RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")
RESULTS_CACHE = os.path.join(DATA, "intl_results.csv")
RATINGS_OUT = os.path.join(DATA, "intl_ratings.csv")
QUESTIONS_OUT = os.path.join(DATA, "wc_elo_questions.csv")

CACHE_MAX_AGE = 24 * 3600  # re-download results only if cache missing/older than 1 day

# --- Elo update constants (World Football Elo style) -----------------------
START_RATING = 1500.0
K_BASE = 40.0            # base K-factor
HOME_ADV = 65.0          # home-field Elo bonus (skipped on neutral ground)

# Recency: older matches matter less. We apply a decay to the K-factor based on how
# long ago the match was relative to the reference date (today). half_life in years.
RECENCY_HALF_LIFE_YEARS = 2.5
# Floor so very old matches still anchor a team somewhere instead of vanishing.
RECENCY_FLOOR = 0.04

# Tournament importance weights (multiplies K). Matches the World-Football-Elo idea
# that competitive matches carry more signal than friendlies. Keyed by substring.
TOURNAMENT_WEIGHTS = [
    ("FIFA World Cup qualification", 1.30),
    ("FIFA World Cup", 1.60),
    ("UEFA Nations League", 1.20),
    ("CONCACAF Nations League", 1.10),
    ("Nations League", 1.10),
    ("qualification", 1.15),     # generic continental qualifiers
    ("UEFA Euro", 1.40),
    ("Copa América", 1.40),
    ("Copa America", 1.40),
    ("African Cup of Nations", 1.30),
    ("AFC Asian Cup", 1.30),
    ("Gold Cup", 1.20),
    ("Confederations Cup", 1.30),
    ("Friendly", 0.70),
]
DEFAULT_TOURNAMENT_WEIGHT = 1.00  # other competitive cups

# --- rating diff -> goals model (per the build spec) -----------------------
GOALS_PER_RATING = 1.0 / 120.0   # supremacy (goals) ~ ratingdiff / 120
BASE_TOTAL = 2.6                 # league-neutral expected total goals
SUP_CAP = 3.0                    # cap supremacy at +-3 goals

# Odds-API spelling -> results.csv spelling. Most names match directly; only a
# couple differ. (NAME_ALIASES from build_elo.py target eloratings.net spellings,
# which differ from this source, so we keep our own minimal map here.)
NAME_ALIASES = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    # the rest match the source directly: Czech Republic, Turkey, Ivory Coast,
    # Curaçao, South Korea, DR Congo, Cape Verde, etc.
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    """Download (or use cached) international results, return played matches only."""
    fresh = False
    if os.path.exists(RESULTS_CACHE):
        age = time.time() - os.path.getmtime(RESULTS_CACHE)
        if age < CACHE_MAX_AGE:
            fresh = True
            print(f"[elo2] using cached results ({age/3600:.1f}h old)")
    if not fresh:
        print("[elo2] downloading international results history ...")
        r = requests.get(RESULTS_URL, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        os.makedirs(DATA, exist_ok=True)
        with open(RESULTS_CACHE, "wb") as f:
            f.write(r.content)
        print(f"[elo2] cached -> {RESULTS_CACHE}")

    df = pd.read_csv(RESULTS_CACHE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # keep only matches that have actually been played (future WC rows have NA scores)
    df = df[df["home_score"].notna() & df["away_score"].notna() & df["date"].notna()]
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # 'neutral' is TRUE/FALSE strings or bools depending on source
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Elo computation
# ---------------------------------------------------------------------------

def _tournament_weight(name: str) -> float:
    if not isinstance(name, str):
        return DEFAULT_TOURNAMENT_WEIGHT
    for key, w in TOURNAMENT_WEIGHTS:
        if key in name:
            return w
    return DEFAULT_TOURNAMENT_WEIGHT


def _margin_multiplier(goal_diff: int) -> float:
    """World-Football-Elo goal-margin multiplier. 1 goal ->1.0, 2 ->1.5, 3+ damped."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0   # 3 ->1.75, 4 ->1.875, ... (standard WFE form)


def _recency_weight(match_date: pd.Timestamp, ref_date: pd.Timestamp) -> float:
    """Time-decayed weight on the K-factor: exp(-ln2 * age_years / half_life)."""
    age_years = (ref_date - match_date).days / 365.25
    if age_years < 0:
        age_years = 0.0
    w = math.exp(-math.log(2.0) * age_years / RECENCY_HALF_LIFE_YEARS)
    return max(RECENCY_FLOOR, w)


def compute_ratings(df: pd.DataFrame):
    """Iterate chronologically and return (ratings, n_matches, last_played) dicts."""
    ref_date = df["date"].max()  # latest played match anchors recency
    ratings: dict[str, float] = {}
    n_matches: dict[str, int] = {}
    last_played: dict[str, pd.Timestamp] = {}

    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, START_RATING)
        ra = ratings.get(a, START_RATING)

        # expected result (home perspective), with home advantage unless neutral
        adv = 0.0 if row.neutral else HOME_ADV
        dr = (rh + adv) - ra
        exp_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

        gh, ga = row.home_score, row.away_score
        if gh > ga:
            score_h = 1.0
        elif gh < ga:
            score_h = 0.0
        else:
            score_h = 0.5

        k = (K_BASE
             * _margin_multiplier(gh - ga)
             * _tournament_weight(row.tournament)
             * _recency_weight(row.date, ref_date))

        delta = k * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

        n_matches[h] = n_matches.get(h, 0) + 1
        n_matches[a] = n_matches.get(a, 0) + 1
        last_played[h] = row.date
        last_played[a] = row.date

    return ratings, n_matches, last_played


def write_ratings_csv(ratings, n_matches, last_played) -> pd.DataFrame:
    rows = []
    for team, rating in ratings.items():
        lp = last_played.get(team)
        rows.append(dict(team=team, rating=round(rating, 2),
                         n_matches=n_matches.get(team, 0),
                         last_played=lp.date().isoformat() if lp is not None else ""))
    rdf = pd.DataFrame(rows).sort_values("rating", ascending=False).reset_index(drop=True)
    rdf.to_csv(RATINGS_OUT, index=False)
    return rdf


# ---------------------------------------------------------------------------
# rating -> probabilities (neutral venue; reuse devig)
# ---------------------------------------------------------------------------

def resolve_rating(team: str, ratings: dict):
    if team in ratings:
        return ratings[team]
    alias = NAME_ALIASES.get(team)
    if alias and alias in ratings:
        return ratings[alias]
    low = team.lower()
    for name, r in ratings.items():
        if name.lower() == low:
            return r
    return None


def match_probs(rate_h: float, rate_a: float):
    """rating pair -> consistent double-Poisson scoreline (neutral, no home edge)."""
    d = rate_h - rate_a
    sup = float(np.clip(d * GOALS_PER_RATING, -SUP_CAP, SUP_CAP))
    tot = BASE_TOTAL
    lam_h0 = max(0.1, (tot + sup) / 2.0)
    lam_a0 = max(0.1, (tot - sup) / 2.0)

    # Seed pH/pD/pA from the supremacy guess, keep the Poisson-implied draw, then
    # let fit_poisson reconcile -- identical approach to build_elo.py so the markets
    # stay consistent with the rest of the repo.
    M0 = score_matrix(lam_h0, lam_a0)
    pH0, pD0, pA0 = _hda_from_matrix(M0)

    lam_h, lam_a = fit_poisson(pH0, pD0, pA0, total_line=tot, p_over=None)
    M = score_matrix(lam_h, lam_a)
    pH, pD, pA = _hda_from_matrix(M)

    return dict(d=d, sup=sup, lam_h=lam_h, lam_a=lam_a,
                pH=float(pH), pD=float(pD), pA=float(pA),
                p_over_15=float(model_over_prob(M, 1.5)),
                p_over_25=float(model_over_prob(M, 2.5)),
                p_over_35=float(model_over_prob(M, 3.5)),
                p_btts=float(M[1:, 1:].sum()), M=M)


def _q(rows, match, mkt, question, prob, commence, match_id, note=""):
    if prob is None or not np.isfinite(prob):
        return
    rows.append(dict(match=match, market=mkt, question=question,
                     prob=round(clip_prob(prob), 4),
                     pct=round(100 * clip_prob(prob), 1),
                     source="elo:model", note=note,
                     commence=commence, match_id=match_id))


def build():
    df = load_results()
    print(f"[elo2] {len(df):,} played matches "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")
    ratings, n_matches, last_played = compute_ratings(df)
    rdf = write_ratings_csv(ratings, n_matches, last_played)
    print(f"[elo2] wrote {len(rdf)} team ratings -> {RATINGS_OUT}")

    summ = pd.read_csv(SUMMARY)
    all_rows, table, skipped = [], [], []
    for _, m in summ.iterrows():
        match, commence, match_id = m["match"], m["commence"], m["match_id"]
        if " vs " not in match:
            skipped.append((match, "unparseable match name"))
            continue
        home, away = match.split(" vs ", 1)
        rh = resolve_rating(home, ratings)
        ra = resolve_rating(away, ratings)
        if rh is None or ra is None:
            miss = (f"{home} & {away}" if rh is None and ra is None
                    else (home if rh is None else away))
            skipped.append((match, f"no rating for {miss}"))
            print(f"[elo2] skip '{match}': no rating for {miss}")
            continue

        r = match_probs(rh, ra)
        note = f"rating {rh:.0f} vs {ra:.0f} (d={r['d']:+.0f}, neutral)"
        _q(all_rows, match, "Moneyline", f"Will {home} win the match?",
           r["pH"], commence, match_id, note)
        _q(all_rows, match, "Moneyline", f"Will {away} win the match?",
           r["pA"], commence, match_id, note)
        _q(all_rows, match, "Moneyline", "Will the match end in a draw?",
           r["pD"], commence, match_id, note)
        _q(all_rows, match, "Total Goals", "Will total goals be over 1.5?",
           r["p_over_15"], commence, match_id, note)
        _q(all_rows, match, "Total Goals", "Will total goals be over 2.5?",
           r["p_over_25"], commence, match_id, note)
        _q(all_rows, match, "Total Goals", "Will total goals be over 3.5?",
           r["p_over_35"], commence, match_id, note)
        _q(all_rows, match, "BTTS", "Will both teams score?",
           r["p_btts"], commence, match_id, note)

        table.append(dict(match=match, rh=rh, ra=ra, d=r["d"],
                          pH=r["pH"], pD=r["pD"], pA=r["pA"],
                          market_pH=float(m["pH"])))

    qdf = pd.DataFrame(all_rows).sort_values(
        ["commence", "match", "market"]).reset_index(drop=True)
    qdf.to_csv(QUESTIONS_OUT, index=False)
    print(f"[elo2] wrote {len(qdf)} questions across {len(table)} matches "
          f"-> {QUESTIONS_OUT}")
    return rdf, qdf, pd.DataFrame(table), skipped


def main():
    rdf, qdf, table, skipped = build()
    if skipped:
        print(f"\nSkipped {len(skipped)} matches:")
        for match, why in skipped:
            print(f"  - {match}: {why}")

    print("\n=== Top 15 teams by rating ===")
    print(f"{'#':>3}  {'team':<26}{'rating':>9}{'n':>7}  last_played")
    for i, t in enumerate(rdf.head(15).itertuples(index=False), 1):
        print(f"{i:>3}  {t.team:<26}{t.rating:>9.1f}{t.n_matches:>7}  {t.last_played}")

    table = table.copy()
    table["diff"] = table["pH"] - table["market_pH"]
    table["absdiff"] = table["diff"].abs()
    top = table.sort_values("absdiff", ascending=False).head(5)
    print("\n=== 5 biggest model-vs-market home-win disagreements ===")
    print(f"{'match':<40}{'model_pH':>9}{'mkt_pH':>9}{'diff':>9}")
    for t in top.itertuples(index=False):
        print(f"{t.match[:39]:<40}{t.pH*100:>8.1f}%{t.market_pH*100:>8.1f}%"
              f"{t.diff*100:>+8.1f}%")


if __name__ == "__main__":
    main()
