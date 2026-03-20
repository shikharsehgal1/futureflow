"""
predict_cl.py — Price CL/Europa fixtures using cross-league ratings

Uses cross-league ratings from build_cross_ratings.py to generate
H/D/A probabilities, xG, fair odds, AH spread and EV for any fixture.

Fixtures are entered manually in the FIXTURES list below, or passed
via CSV file with columns: Date, HomeTeam, AwayTeam, Competition,
optionally OddsH, OddsD, OddsA.

Usage:
    python3.10 crossleague/predict_cl.py
    python3.10 crossleague/predict_cl.py --fixtures crossleague/data/upcoming_cl.csv
"""

import os, pickle, argparse
import pandas as pd
import numpy as np
from scipy.special import factorial

# ── FIXTURES TO PREDICT (edit this each week) ─────────────────────────────────
# Format: (Date, HomeTeam, AwayTeam, Competition, OddsH, OddsD, OddsA)
# OddsH/D/A are optional — set to None if not available
FIXTURES = [
    # CL Quarter-Finals Leg 1
    ("2026-04-07", "Sporting CP",    "Arsenal",        "Champions League", 4.45, 3.63, 1.81),
    ("2026-04-07", "Real Madrid",    "Bayern Munich",  "Champions League", 3.14, 3.91, 2.10),
    ("2026-04-08", "Barcelona",      "Ath Madrid",     "Champions League", 1.57, 4.98, 4.61),
    ("2026-04-08", "Paris SG",       "Liverpool",      "Champions League", 1.97, 4.00, 3.41),
]

# ── CONFIG ────────────────────────────────────────────────────────────────────
RATINGS_PATH = "crossleague/data/cross_ratings.pkl"
ALERT_THRESHOLD = 0.06

MEAN_HOME = 1.50   # approximate European competition goal averages
MEAN_AWAY = 1.20


def load_ratings():
    with open(RATINGS_PATH, "rb") as f:
        return pickle.load(f)


def poisson_probs(mu, mean_home=MEAN_HOME, mean_away=MEAN_AWAY, max_g=10):
    total = mean_home + mean_away
    lH    = np.clip((total + mu) / 2, 0.05, None)
    lA    = np.clip((total - mu) / 2, 0.05, None)
    g     = np.arange(max_g + 1)
    pH    = np.exp(-lH) * lH**g / factorial(g)
    pA    = np.exp(-lA) * lA**g / factorial(g)
    M     = np.outer(pH, pA)
    pHwin = np.tril(M, -1).sum()
    pDraw = M.trace()
    pAwin = np.triu(M, 1).sum()
    pO25  = 1 - sum(M[i,j] for i in range(max_g+1)
                    for j in range(max_g+1) if i+j<=2 and i<max_g+1 and j<max_g+1)
    return (np.clip(pHwin, 1e-6, 1-1e-6),
            np.clip(pDraw, 1e-6, 1-1e-6),
            np.clip(pAwin, 1e-6, 1-1e-6),
            np.clip(pO25,  1e-6, 1-1e-6),
            lH, lA)


def predict_match(home, away, ratings, home_adv, odds_h=None, odds_d=None, odds_a=None):
    r_h = ratings.get(home, 0.0)
    r_a = ratings.get(away, 0.0)
    mu  = r_h - r_a + home_adv

    pH, pD, pA, pO25, xGH, xGA = poisson_probs(mu)

    # Fair odds
    fair_h = round(1/pH, 2)
    fair_d = round(1/pD, 2)
    fair_a = round(1/pA, 2)

    # EV
    ev_h = ev_d = ev_a = None
    if odds_h and odds_d and odds_a:
        ev_h = round(pH * odds_h - 1, 4)
        ev_d = round(pD * odds_d - 1, 4)
        ev_a = round(pA * odds_a - 1, 4)

    # AH spread (nearest 0.25)
    ah = round(mu * 4) / 4
    # Flip sign for display: positive = home gives goals
    ah_str = f"{-ah:+.2f}" if ah != 0 else "0"

    # Delta vs market
    dh = dd = da = None
    if odds_h and odds_d and odds_a:
        total   = 1/odds_h + 1/odds_d + 1/odds_a
        mkt_h   = (1/odds_h) / total
        mkt_d   = (1/odds_d) / total
        mkt_a   = (1/odds_a) / total
        dh = pH - mkt_h
        dd = pD - mkt_d
        da = pA - mkt_a

    return {
        "mu":     mu,
        "pH": pH, "pD": pD, "pA": pA, "pO25": pO25,
        "xGH": xGH, "xGA": xGA,
        "fair_h": fair_h, "fair_d": fair_d, "fair_a": fair_a,
        "ev_h": ev_h, "ev_d": ev_d, "ev_a": ev_a,
        "ah": ah_str,
        "dh": dh, "dd": dd, "da": da,
    }


def print_predictions(fixtures, result):
    ratings  = result["ratings"]
    home_adv = result["home_adv"]

    print(f"\n{'='*90}")
    print(f"  CL/EUROPA PREDICTIONS  |  Cross-League Ratings  |  {result['ref_date'].date()}")
    print(f"{'='*90}")
    print(f"\n  {'Date':<12} {'Home':<22} {'Away':<22}  {'xGH':>5} {'xGA':>5}  "
          f"{'H%':>6} {'D%':>6} {'A%':>6}  {'O25%':>6}  "
          f"{'Mkt_H':>6} {'Mkt_D':>6} {'Mkt_A':>6}  "
          f"{'ΔH':>6} {'ΔD':>6} {'ΔA':>6}")
    print(f"  {'-'*88}")

    alerts = []

    for row in fixtures:
        date, home, away, comp = row[:4]
        odds_h = row[4] if len(row) > 4 else None
        odds_d = row[5] if len(row) > 5 else None
        odds_a = row[6] if len(row) > 6 else None

        if home not in ratings:
            print(f"  WARNING: {home} not in cross-league ratings")
        if away not in ratings:
            print(f"  WARNING: {away} not in cross-league ratings")

        p = predict_match(home, away, ratings, home_adv,
                          odds_h, odds_d, odds_a)

        mkt_str = ""
        dlt_str = ""
        alert   = ""

        if odds_h:
            total = 1/odds_h + 1/odds_d + 1/odds_a
            mh = (1/odds_h)/total; md = (1/odds_d)/total; ma = (1/odds_a)/total
            mkt_str = f"  {mh:>5.1%} {md:>6.1%} {ma:>6.1%}"
            dh, dd, da = p["dh"], p["dd"], p["da"]
            dlt_str = f"  {dh:>+5.1%} {dd:>+5.1%} {da:>+5.1%}"
            if max(abs(dh), abs(dd), abs(da)) >= ALERT_THRESHOLD:
                alert = "  ◀ ALERT"
                alerts.append((date, home, away, comp, p, odds_h, odds_d, odds_a))
        else:
            mkt_str = f"  {'—':>5} {'—':>6} {'—':>6}"
            dlt_str = f"  {'—':>5} {'—':>5} {'—':>5}"

        print(f"  {date:<12} {home:<22} {away:<22}  "
              f"{p['xGH']:>5.2f} {p['xGA']:>5.2f}  "
              f"{p['pH']:>5.1%} {p['pD']:>6.1%} {p['pA']:>6.1%}  "
              f"{p['pO25']:>5.1%}"
              f"{mkt_str}{dlt_str}{alert}")

    # Market pricing
    print(f"\n\n{'='*90}")
    print(f"  MARKET PRICING")
    print(f"{'='*90}")
    print(f"\n  {'Home':<22} {'Away':<22}  {'AH':>6}  "
          f"{'FairH':>7} {'FairD':>7} {'FairA':>7}  "
          f"{'EvH%':>7} {'EvD%':>7} {'EvA%':>7}  {'Best':>10}")
    print(f"  {'-'*85}")

    for row in fixtures:
        date, home, away, comp = row[:4]
        odds_h = row[4] if len(row) > 4 else None
        odds_d = row[5] if len(row) > 5 else None
        odds_a = row[6] if len(row) > 6 else None
        if not odds_h:
            continue

        p = predict_match(home, away, ratings, home_adv,
                          odds_h, odds_d, odds_a)

        ev_h, ev_d, ev_a = p["ev_h"], p["ev_d"], p["ev_a"]
        best_ev  = max(ev_h, ev_d, ev_a)
        best_lbl = ["H","D","A"][[ev_h, ev_d, ev_a].index(best_ev)]
        star     = f"{best_ev:>+.1%} ★({best_lbl})" if best_ev > 0 else \
                   f"{best_ev:>+.1%}  ({best_lbl})"

        print(f"  {home:<22} {away:<22}  {p['ah']:>6}  "
              f"{p['fair_h']:>7.2f} {p['fair_d']:>7.2f} {p['fair_a']:>7.2f}  "
              f"{ev_h:>+6.1%} {ev_d:>+6.1%} {ev_a:>+6.1%}  {star}")

    # Alerts
    if alerts:
        print(f"\n\n{'='*90}")
        print(f"  ALERTS — {len(alerts)} game(s) where model disagrees by >{ALERT_THRESHOLD*100:.0f}pp")
        print(f"{'='*90}")
        for date, home, away, comp, p, oh, od, oa in alerts:
            total = 1/oh + 1/od + 1/oa
            mh = (1/oh)/total; md = (1/od)/total; ma = (1/oa)/total
            max_d = max(abs(p["dh"]), abs(p["dd"]), abs(p["da"]))
            side  = ["Home","Draw","Away"][
                [abs(p["dh"]),abs(p["dd"]),abs(p["da"])].index(max_d)]
            sign  = "+" if [p["dh"],p["dd"],p["da"]][
                [abs(p["dh"]),abs(p["dd"]),abs(p["da"])].index(max_d)] > 0 else "-"
            print(f"\n  {home} vs {away} ({comp})")
            print(f"    Model sees {side} probability as {'higher' if sign=='+' else 'lower'} "
                  f"than market ({sign}{max_d:.1%})")
            print(f"    Model: H={p['pH']:.1%}  D={p['pD']:.1%}  A={p['pA']:.1%}")
            print(f"    Mkt:   H={mh:.1%}  D={md:.1%}  A={ma:.1%}")


def main(fixtures_path=None):
    result = load_ratings()

    if fixtures_path:
        fix_df   = pd.read_csv(fixtures_path)
        fixtures = []
        for _, row in fix_df.iterrows():
            f = [row["Date"], row["HomeTeam"], row["AwayTeam"],
                 row.get("Competition","CL")]
            for col in ["OddsH","OddsD","OddsA"]:
                f.append(row[col] if col in row and pd.notna(row[col]) else None)
            fixtures.append(tuple(f))
    else:
        fixtures = FIXTURES

    print_predictions(fixtures, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default=None,
                        help="CSV with Date,HomeTeam,AwayTeam,Competition,OddsH,OddsD,OddsA")
    args = parser.parse_args()
    main(args.fixtures)
