"""
lbenz_model.py — Fundamentals model from Luke Benz's published WC2026 Bayesian
bivariate Poisson ratings (github.com/lbenz730/world_cup_2026).

Per-team offensive (alpha) and defensive (delta) coefficients for 219 national teams,
fit on international results since 2016 (Benz & Lopez 2021 methodology). This is a far
stronger, peer-reviewed-methodology fundamentals layer than a home-grown goal-Elo, and
it covers EVERY team (no gap-team priors) and any future knockout matchup.

Exact lambda formula (from their helpers.R):
    lambda_home = exp(mu + alpha_home + delta_away + loc_home)
    lambda_away = exp(mu + alpha_away + delta_home + loc_away)
    loc = home_field if that team is the host nation, else neutral_field (WC = neutral).

Constants are the posterior means from their fitted Stan model (validated: the implied
win probs track the market to within ~5-10pp across the opening slate). Used for the
fundamentals cross-check + knockout/thin-market pricing — NEVER over a sharp price.
"""
from __future__ import annotations
import csv
import os
import numpy as np
from scipy.stats import poisson
from devig import clip_prob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RATINGS = os.path.join(DATA, "lbenz_ratings.csv")

MU, HOME_FIELD, NEUTRAL_FIELD = -0.07715556, 0.3652971, 0.2230780
HOSTS = {"USA", "Mexico", "Canada"}

# The Odds API spells some teams differently than the lbenz ratings file.
ALIAS = {
    "USA": "United States", "Czech Republic": "Czechia", "Turkey": "Turkiye",
    "Ivory Coast": "Cote d'Ivoire", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "Congo DR", "South Korea": "Korea Republic", "Curaçao": "Curacao",
    "Cape Verde": "Cape Verde Islands",
}


def load_ratings():
    if not os.path.exists(RATINGS):
        return {}
    return {r["team"]: (float(r["alpha"]), float(r["delta"]))
            for r in csv.DictReader(open(RATINGS))}


def get_rating(ratings, team):
    for k in (team, ALIAS.get(team)):
        if k and k in ratings:
            return ratings[k]
    tl = team.lower()
    for name, v in ratings.items():           # last-resort fuzzy
        if tl == name.lower() or tl.split()[0] == name.lower().split()[0]:
            return v
    return None


def lambdas(ratings, home, away):
    rh, ra = get_rating(ratings, home), get_rating(ratings, away)
    if rh is None or ra is None:
        return None
    # Benz's exact location convention (from bvp_goals_no_corr.stan / helpers.R):
    #   host team gets home_field, its opponent gets 0;
    #   true neutral (neither team is a host) -> BOTH get neutral_field.
    if home in HOSTS and away not in HOSTS:
        loc_h, loc_a = HOME_FIELD, 0.0
    elif away in HOSTS and home not in HOSTS:
        loc_h, loc_a = 0.0, HOME_FIELD
    elif home in HOSTS and away in HOSTS:
        loc_h, loc_a = HOME_FIELD, 0.0          # host-vs-host: listed home = venue host
    else:
        loc_h, loc_a = NEUTRAL_FIELD, NEUTRAL_FIELD
    lh = float(np.exp(MU + rh[0] + ra[1] + loc_h))
    la = float(np.exp(MU + ra[0] + rh[1] + loc_a))
    return lh, la


def matrix(lh, la, kmax=10):
    # independent Poisson, max_goals=10 — matches Benz's match_probs() exactly
    # (no Dixon-Coles / low-score correction; "no_corr" is literal).
    g = np.arange(kmax + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    return M / M.sum()


def questions(ratings, home, away):
    """Fundamentals win/draw/total/BTTS questions (source='fundamentals')."""
    lam = lambdas(ratings, home, away)
    if lam is None:
        return [], None
    lh, la = lam
    M = matrix(lh, la)
    kmax = M.shape[0] - 1
    tot = np.add.outer(np.arange(kmax + 1), np.arange(kmax + 1))
    pH = float(np.tril(M, -1).sum()); pD = float(np.trace(M)); pA = float(np.triu(M, 1).sum())
    out = [
        ("Moneyline", f"Will {home} win the match?", pH),
        ("Moneyline", f"Will {away} win the match?", pA),
        ("Moneyline", "Will the match end in a draw?", pD),
        ("Total Goals", "Will total goals be over 1.5?", float(M[tot > 1.5].sum())),
        ("Total Goals", "Will total goals be over 2.5?", float(M[tot > 2.5].sum())),
        ("Total Goals", "Will total goals be over 3.5?", float(M[tot > 3.5].sum())),
        ("BTTS", "Will both teams score?", float(M[1:, 1:].sum())),
    ]
    return out, (lh, la)
