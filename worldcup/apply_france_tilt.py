"""
apply_france_tilt.py — manual directional override for France vs Senegal.

Deliberately skews the published numbers toward France: raises France's goal
rate, cuts Senegal's, and re-derives EVERY result-linked question from that one
shifted double-Poisson so the whole set stays mutually consistent (moneyline,
double chance, handicap, team totals, first-to-score, correct score, halves all
move together — no contradictions). A wider supremacy gap simultaneously
collapses the draw and Senegal, which is exactly the requested tilt.

This is an explicit, NON-calibrated override (the opposite of the honest-sharp
philosophy in predict_wc.py). It writes to its own CSV and never touches
wc_questions.csv / the sharp source model.

Knob: STRENGTH in [0,1].  0 = untouched model, 1 = maximal squeeze.
Run:  python apply_france_tilt.py [STRENGTH]
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

from devig import score_matrix, _hda_from_matrix
from predict_wc import generate_questions

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SUMMARY = os.path.join(DATA, "wc_match_summary.csv")
OUT = os.path.join(DATA, "wc_questions_france_tilt.csv")

MATCH = "France vs Senegal"


def tilt(strength: float):
    s = max(0.0, min(1.0, strength))
    summ = pd.read_csv(SUMMARY)
    row = summ[summ["match"] == MATCH].iloc[0]
    lh0, la0 = float(row["lam_h"]), float(row["lam_a"])

    # Heavy France tilt: push France's rate up, Senegal's down. At STRENGTH=1
    # France's rate is +50% and Senegal's is -55% — a lopsided supremacy that
    # drives France-win toward the cap and starves the draw / Senegal markets.
    lh = lh0 * (1 + 0.50 * s)
    la = la0 * (1 - 0.55 * s)

    M = score_matrix(lh, la)
    pH, pD, pA = _hda_from_matrix(M)   # moneyline derived from the SAME matrix -> coherent

    mp = dict(home="France", away="Senegal", pH=pH, pD=pD, pA=pA,
              src_h2h="france-tilt", lam_h=lh, lam_a=la,
              total_line=np.nan, p_over=np.nan,
              spread_line=np.nan, spread_home_p=np.nan)
    rows = generate_questions(mp, row["commence"])
    df = pd.DataFrame(rows)
    df["match_id"] = row["match_id"]
    df["commence"] = row["commence"]
    df.to_csv(OUT, index=False)

    print(f"STRENGTH={s:.2f}  France λ {lh0:.3f} -> {lh:.3f}   Senegal λ {la0:.3f} -> {la:.3f}")
    print(f"Moneyline:  France {100*pH:.1f}%   Draw {100*pD:.1f}%   Senegal {100*pA:.1f}%")
    print(f"Wrote {len(df)} tilted questions -> {OUT}")
    return df


if __name__ == "__main__":
    strength = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    tilt(strength)
