"""
referee_model.py — Referee card-strictness factor for the cards/booking markets.

Referee strictness is the biggest under-priced signal in card markets (refs vary ~2x
in cards/game, stable within a season). Given a match's assigned referee, we scale the
expected total cards by (referee_cards_per_game / field_average).

Data:
  data/referees.csv               per-referee cards/game (worldfootball.net, 52 WC refs)
  data/wc_referee_assignments.csv match -> assigned referee (confirmed early rounds)

Note: for matches that already have sharp Pinnacle card odds, the market already prices
the referee — this factor mainly sharpens the model's card numbers on un-priced matches
and serves as a cross-check elsewhere.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

_RATE, _AVG, _ASSIGN = None, 4.2, None


def _load():
    global _RATE, _AVG, _ASSIGN
    if _RATE is not None:
        return
    _RATE = {}
    p = os.path.join(DATA, "referees.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            try:
                _RATE[r["referee"]] = float(r["cards_per_game"])
            except (ValueError, KeyError):
                pass
    _AVG = sum(_RATE.values()) / len(_RATE) if _RATE else 4.2
    _ASSIGN = {}
    p = os.path.join(DATA, "wc_referee_assignments.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r.get("referee"):
                _ASSIGN[(r["home_team"], r["away_team"])] = r["referee"]


def card_factor(home, away):
    """Return (factor, referee_name). factor=1.0 if no assigned ref or no rate."""
    _load()
    ref = _ASSIGN.get((home, away))
    if not ref:
        return 1.0, None
    rate = _RATE.get(ref)
    if not rate:
        return 1.0, ref
    return rate / _AVG, ref
