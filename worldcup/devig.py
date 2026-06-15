"""
devig.py — Core probability utilities for the World Cup model.

The dominant strategy for relative-Brier scoring is to anchor every prediction to
the sharpest available market (Pinnacle), strip the bookmaker margin ("vig"), and
report the resulting calibrated probability. This module provides:

  1. de-vig of 2-way and n-way markets (multiplicative / proportional method)
  2. inversion of de-vigged h2h + totals odds into a double-Poisson goal model,
     from which EVERY goal-derived question (correct score, BTTS, odd/even,
     team totals, half markets, arbitrary handicaps) can be answered consistently.

Everything is plain numpy/scipy so it slots into the existing repo with no new deps.
"""
from __future__ import annotations

import numpy as np
from math import exp, factorial
from scipy.optimize import minimize

# Tail clip: never report 0/1. A confident-wrong call costs (1-0)^2 = 1.0 in Brier,
# which is catastrophic under relative scoring. Clip protects the downside while
# costing almost nothing on the upside (see README strategy notes).
PROB_FLOOR = 0.03
PROB_CEIL = 0.97

MAX_GOALS = 12  # truncation of the Poisson scoreline matrix (0..MAX_GOALS each side)


def clip_prob(p: float) -> float:
    return float(min(PROB_CEIL, max(PROB_FLOOR, p)))


def implied(odds: float) -> float:
    """Decimal odds -> raw implied probability (still contains vig)."""
    if odds is None or not np.isfinite(odds) or odds <= 1.0:
        return np.nan
    return 1.0 / odds


def devig(odds_list):
    """Multiplicative de-vig of an n-way market.

    Takes a list of decimal odds for mutually-exclusive, exhaustive outcomes and
    returns normalised probabilities summing to 1. This is the standard sharp-book
    de-vig; for Pinnacle the multiplicative method is well-behaved.
    """
    raw = np.array([implied(o) for o in odds_list], dtype=float)
    if np.any(~np.isfinite(raw)):
        return None
    s = raw.sum()
    if s <= 0:
        return None
    return raw / s


def devig_two(odds_yes: float, odds_no: float) -> float:
    """De-vig a 2-way (YES/NO) market, returning P(YES)."""
    p = devig([odds_yes, odds_no])
    return float(p[0]) if p is not None else np.nan


# ---------------------------------------------------------------------------
# Poisson scoreline model
# ---------------------------------------------------------------------------

def _pois_pmf(lam: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    # exp(-lam) * lam^k / k!
    logpmf = -lam + k * np.log(max(lam, 1e-9)) - np.array([np.log(float(factorial(i))) for i in k])
    return np.exp(logpmf)


def score_matrix(lam_h: float, lam_a: float, rho: float = -0.04, kmax: int = MAX_GOALS) -> np.ndarray:
    """Double-Poisson scoreline matrix M[i,j] = P(home=i, away=j).

    Dixon-Coles low-score correlation tweak (rho) inflates 0-0/1-1 and deflates
    1-0/0-1 slightly, matching real football's draw structure better than a plain
    independent Poisson. rho defaults to a calibrated -0.04.
    """
    ph = _pois_pmf(lam_h, kmax)
    pa = _pois_pmf(lam_a, kmax)
    M = np.outer(ph, pa)
    # Dixon-Coles adjustment on the four lowest scorelines
    tau = np.ones((kmax + 1, kmax + 1))
    tau[0, 0] = 1 - lam_h * lam_a * rho
    tau[0, 1] = 1 + lam_h * rho
    tau[1, 0] = 1 + lam_a * rho
    tau[1, 1] = 1 - rho
    M = M * tau
    M = M / M.sum()
    return M


def _hda_from_matrix(M: np.ndarray):
    pH = np.tril(M, -1).sum()   # home > away
    pD = np.trace(M)
    pA = np.triu(M, 1).sum()
    return pH, pD, pA


def model_over_prob(M: np.ndarray, line: float) -> float:
    """P(total goals over `line`) handling .0/.25/.5/.75 Asian lines.

    Quarter lines (x.25, x.75) are the average of the two adjacent half/whole lines.
    Whole lines push on exact totals, so probability is renormalised over the
    non-push mass (matches how over/under odds are de-vigged).
    """
    kmax = M.shape[0] - 1
    totals = np.add.outer(np.arange(kmax + 1), np.arange(kmax + 1))
    frac = round(line * 4) % 4

    def over_half(half_line):  # half_line is x.5
        return float(M[totals > half_line].sum())

    def over_whole(whole_line):  # integer line, renormalise over push
        over = float(M[totals > whole_line].sum())
        push = float(M[totals == whole_line].sum())
        denom = 1.0 - push
        return over / denom if denom > 1e-9 else over

    if frac == 2:                      # x.5
        return over_half(line)
    if frac == 0:                      # x.0
        return over_whole(line)
    if frac == 1:                      # x.25  -> avg of x.0 and x.5
        return 0.5 * over_whole(line - 0.25) + 0.5 * over_half(line + 0.25)
    # frac == 3                        # x.75 -> avg of x.5 and (x+1).0
    return 0.5 * over_half(line - 0.25) + 0.5 * over_whole(line + 0.25)


def fit_poisson(pH, pD, pA, total_line=None, p_over=None, rho=-0.04):
    """Invert de-vigged market into (lam_home, lam_away).

    Fits the two Poisson means so the model reproduces the de-vigged home/away win
    probabilities and (if supplied) the de-vigged over probability at the market
    total line. The draw falls out of the fit. Returns (lam_h, lam_a).
    """
    # Sensible starting point from a rough supremacy/total guess
    tot0 = total_line if total_line else 2.6
    # supremacy proxy from win-prob gap
    sup0 = np.clip((pH - pA) * 1.6, -2.5, 2.5)
    x0 = np.array([max(0.2, (tot0 + sup0) / 2), max(0.2, (tot0 - sup0) / 2)])

    def loss(x):
        lh, la = max(0.05, x[0]), max(0.05, x[1])
        M = score_matrix(lh, la, rho)
        mH, mD, mA = _hda_from_matrix(M)
        e = 4.0 * ((mH - pH) ** 2 + (mA - pA) ** 2) + 2.0 * (mD - pD) ** 2
        if total_line is not None and p_over is not None and np.isfinite(p_over):
            e += 3.0 * (model_over_prob(M, total_line) - p_over) ** 2
        return e

    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
    lh, la = float(max(0.05, res.x[0])), float(max(0.05, res.x[1]))
    return lh, la
