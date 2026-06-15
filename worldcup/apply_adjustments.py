"""
apply_adjustments.py — Adjusted fundamentals model.

Base = Luke Benz's published Bayesian bivariate-Poisson ratings (lbenz_model.py),
then layered with two signals that model doesn't capture:
  * diaspora / host-crowd edge  (home_advantage.csv)   — goals added to supremacy
  * in-tournament momentum      (momentum_weights.json) — +0.107 g/goal over-perf, cap ±0.3

Output data/wc_fundamentals.csv: a venue- and form-aware fundamentals win/draw/total/BTTS
for every match (and any future knockout pairing).

CRITICAL: this is the MODEL/cross-check layer. NEVER entered over a sharp market price
(the market already prices ratings, crowd, form and injuries — overriding it would
double-count and worsen Brier). Its job: (1) a strong cross-check so divergences from
the market are meaningful, (2) pricing knockout / thin markets as results accumulate.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from devig import model_over_prob, _hda_from_matrix, clip_prob
from lbenz_model import load_ratings, lambdas, matrix as bvp_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "wc_fundamentals.csv")
MOM_CAP = 0.30


def momentum_nudges(ratings, coef):
    """Per-team in-tournament over/under-performance vs the lbenz expectation."""
    path = os.path.join(DATA, "wc_scores.json")
    if not os.path.exists(path):
        return {}
    perf = {}
    for e in json.load(open(path)):
        if not e.get("completed"):
            continue
        sc = {s["name"]: int(s["score"]) for s in (e.get("scores") or []) if s.get("score") is not None}
        h, a = e["home_team"], e["away_team"]
        if h not in sc or a not in sc:
            continue
        lam = lambdas(ratings, h, a)
        if lam is None:
            continue
        exp_gd = lam[0] - lam[1]
        gd = sc[h] - sc[a]
        perf.setdefault(h, []).append(gd - exp_gd)
        perf.setdefault(a, []).append(-gd + exp_gd)
    return {t: float(np.clip(coef * np.mean(v), -MOM_CAP, MOM_CAP)) for t, v in perf.items()}


def main():
    ratings = load_ratings()
    if not ratings:
        print("lbenz_ratings.csv missing — run fetch of github.com/lbenz730/world_cup_2026 first.")
        return
    ha = {}
    p = os.path.join(DATA, "home_advantage.csv")
    if os.path.exists(p):
        ha = {r["match_id"]: float(r.get("net_home_adv_goals", 0.0) or 0.0)
              for r in pd.read_csv(p).to_dict("records")}
    coef = 0.107
    mp = os.path.join(DATA, "momentum_weights.json")
    if os.path.exists(mp):
        coef = float(json.load(open(mp)).get("per_game_rating_update", coef))
    mom = momentum_nudges(ratings, coef)

    summ = pd.read_csv(os.path.join(DATA, "wc_match_summary.csv"))
    rows, divergences, unmatched = [], [], []
    for _, m in summ.iterrows():
        home, away = m["match"].split(" vs ")
        lam = lambdas(ratings, home, away)
        if lam is None:
            unmatched.append(m["match"]); continue
        lh, la = lam
        total = lh + la
        sup = (lh - la) + ha.get(m["match_id"], 0.0) + mom.get(home, 0.0) - mom.get(away, 0.0)
        lh2, la2 = max(0.05, (total + sup) / 2), max(0.05, (total - sup) / 2)
        M = bvp_matrix(lh2, la2)          # independent Poisson — matches Benz (no Dixon-Coles)
        pH, pD, pA = _hda_from_matrix(M)
        kmax = M.shape[0] - 1
        tot = np.add.outer(np.arange(kmax + 1), np.arange(kmax + 1))

        def q(mkt, question, prob):
            rows.append(dict(match=m["match"], market=mkt, question=question,
                             prob=round(clip_prob(prob), 4), pct=round(100 * clip_prob(prob), 1),
                             source="fundamentals", note="lbenz bvp + crowd + momentum",
                             commence=m["commence"], match_id=m["match_id"]))
        q("Moneyline", f"Will {home} win the match?", pH)
        q("Moneyline", f"Will {away} win the match?", pA)
        q("Moneyline", "Will the match end in a draw?", pD)
        q("Total Goals", "Will total goals be over 2.5?", float(M[tot > 2.5].sum()))
        q("BTTS", "Will both teams score?", float(M[1:, 1:].sum()))
        if pd.notna(m.get("pH")):
            divergences.append((abs(pH - float(m["pH"])), m["match"], pH, float(m["pH"]),
                                ha.get(m["match_id"], 0.0)))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} fundamentals questions ({len(summ) - len(unmatched)} matches) -> {OUT}")
    if unmatched:
        print(f"UNMATCHED teams (no lbenz rating): {unmatched}")
    if mom:
        print("Momentum applied: " + ", ".join(f"{t} {v:+.2f}"
              for t, v in sorted(mom.items(), key=lambda x: -abs(x[1]))[:6]))
    divergences.sort(reverse=True)
    print("\nBiggest fundamentals-vs-market home-win gaps (cross-check flags):")
    for d, mt, pf, pm, h in divergences[:8]:
        print(f"  {mt:34s} fundamentals {100*pf:4.0f}%  market {100*pm:4.0f}%  (gap {100*d:+.0f}pp, crowd {h:+.2f})")


if __name__ == "__main__":
    main()
