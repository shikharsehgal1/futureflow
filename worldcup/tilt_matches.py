"""
tilt_matches.py — directional lambda tilts for selected matches.

Skews the per-match goal rates in data/wc_match_summary.csv toward a favored team
(boost its lambda, cut the opponent's). sp_solver.py derives EVERY question from
these lambdas, so one tilt propagates coherently to moneyline, halftime, team
scoring, SoT/fouls/corners comparisons, etc.

Backs up the original to wc_match_summary.csv.bak so it is fully reversible
(python3 tilt_matches.py --restore).

STRENGTH s in [0,1]: favored λ *= (1 + 0.50 s),  opponent λ *= (1 - 0.55 s).
  moderate ("upweight")        ~0.4
  heavy    ("heavily upweight")~0.7
  v.heavy  ("very heavy")       1.0
"""
from __future__ import annotations
import csv, os, shutil, sys
from scipy.stats import poisson
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUMMARY = os.path.join(DATA, "wc_match_summary.csv")
BAK = SUMMARY + ".bak"

# (match, favored team, strength)
TILTS = [
    ("Iraq vs Norway",        "Norway",    0.4),
    ("Argentina vs Algeria",  "Argentina", 1.0),
    ("Austria vs Jordan",     "Austria",   0.7),
    ("Portugal vs DR Congo",  "Portugal",  1.0),
    ("Argentina vs Austria",  "Argentina", 1.0),   # both targets -> Argentina (very heavy) wins
    ("Norway vs Senegal",     "Norway",    0.4),
    ("Portugal vs Uzbekistan","Portugal",  1.0),
    ("Norway vs France",      "Norway",    0.4),    # France (competitor) downweighted here
    ("Colombia vs Portugal",  "Portugal",  1.0),
    ("Algeria vs Austria",    "Austria",   0.7),
    ("Jordan vs Argentina",   "Argentina", 1.0),
]


def _moneyline(lh, la, k=11):
    g = np.arange(k); M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la)); M /= M.sum()
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def restore():
    if os.path.exists(BAK):
        shutil.copy(BAK, SUMMARY); print(f"Restored {SUMMARY} from backup.")
    else:
        print("No backup found.")


def main():
    if "--restore" in sys.argv:
        return restore()
    # --rebaseline: treat the CURRENT summary as the new clean baseline (use after
    # predict_wc.py regenerates honest numbers), then tilt on top of it.
    if "--rebaseline" in sys.argv:
        shutil.copy(SUMMARY, BAK); print(f"Re-baselined {BAK} from fresh summary.")
    if not os.path.exists(BAK):
        shutil.copy(SUMMARY, BAK); print(f"Backed up -> {BAK}")

    rows = list(csv.DictReader(open(BAK)))   # always tilt from the clean baseline
    cfg = {m: (fav, s) for m, fav, s in TILTS}
    for r in rows:
        if r["match"] not in cfg or not r.get("lam_h"):
            continue
        fav, s = cfg[r["match"]]
        home, away = [t.strip() for t in r["match"].split(" vs ")]
        lh, la = float(r["lam_h"]), float(r["lam_a"])
        boost, cut = 1 + 0.50 * s, 1 - 0.55 * s
        if fav == home:
            lh2, la2 = lh * boost, la * cut
        else:
            lh2, la2 = lh * cut, la * boost
        pH0, _, _ = _moneyline(lh, la); pH1, pD1, pA1 = _moneyline(lh2, la2)
        favp0 = pH0 if fav == home else (1 - pH0 - _moneyline(lh, la)[1])
        favp1 = pH1 if fav == home else pA1
        r["lam_h"], r["lam_a"] = round(lh2, 3), round(la2, 3)
        print(f"  {r['match']:26s} fav {fav:10s} s={s:.1f}  "
              f"λ ({lh:.2f},{la:.2f})->({lh2:.2f},{la2:.2f})  "
              f"{fav} win {100*favp0:.0f}%->{100*favp1:.0f}%  draw->{100*pD1:.0f}%")

    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\nWrote tilted lambdas -> {SUMMARY} ({len(cfg)} matches). Now run sp_solver.py.")


if __name__ == "__main__":
    main()
