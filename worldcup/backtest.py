"""
backtest.py — Walk-forward backtest of the fundamentals engine on 49k international
results (martj42). Honest measurement of the part of the model WE control: the
goal-based Elo + Poisson derivation. (The market-anchored layer can't be Brier-beaten,
so it isn't the thing under test here.)

Protocol (no leakage):
  * Sort all matches chronologically.
  * BURN-IN  (< CALIB_START): update Elo only, no scoring.
  * CALIB    [CALIB_START, TEST_START): fit the two free derivation params — Elo points
    per goal of supremacy (C) and the average total goals (T) — by minimising RPS.
  * TEST     [TEST_START, today): for each played match, predict 1X2 / over2.5 / BTTS
    from the PRE-match Elo, score it, THEN update Elo (online walk-forward).

Metrics: RPS (ranked probability score — the standard for ordinal 1X2), multiclass
log-loss & Brier, binary Brier for over-2.5 and BTTS, plus calibration buckets.
Baselines: uniform (1/3,1/3,1/3) and historical base-rates. Sliced by competition,
neutrality, and favourite tier. Bootstrap 95% CI on the headline RPS.

Usage: python3 backtest.py [--test-start 2018-01-01]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from math import factorial

RESULTS = "data/intl_results.csv"
INIT, HFA, KBASE = 1500.0, 65.0, 40.0
TW = {"FIFA World Cup": 1.0, "FIFA World Cup qualification": 0.85,
      "UEFA Euro": 1.0, "Copa América": 1.0, "African Cup of Nations": 0.9,
      "Friendly": 0.5}
_POIS = {}


def pois(lam, kmax=10):
    key = round(lam, 3)
    if key not in _POIS:
        k = np.arange(kmax + 1)
        _POIS[key] = np.exp(-lam) * lam ** k / np.array([factorial(i) for i in k], float)
    return _POIS[key]


def hda_probs(sup, T, kmax=10):
    lh, la = max(0.05, (T + sup) / 2), max(0.05, (T - sup) / 2)
    M = np.outer(pois(lh, kmax), pois(la, kmax))
    M /= M.sum()
    pH = np.tril(M, -1).sum(); pD = np.trace(M); pA = np.triu(M, 1).sum()
    tot = np.add.outer(np.arange(kmax + 1), np.arange(kmax + 1))
    p_o25 = M[tot > 2.5].sum()
    p_btts = M[1:, 1:].sum()
    return np.array([pH, pD, pA]), p_o25, p_btts


def margin_mult(gd):
    gd = abs(gd)
    return 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)


def tw(t):
    for k, v in TW.items():
        if isinstance(t, str) and k in t:
            return v
    return 0.6


def rps(p, outcome):                       # outcome: 0=home,1=draw,2=away
    o = np.zeros(3); o[outcome] = 1
    cp, co = np.cumsum(p), np.cumsum(o)
    return 0.5 * ((cp[0] - co[0]) ** 2 + (cp[1] - co[1]) ** 2)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2018-01-01")
    ap.add_argument("--calib-start", default="2014-01-01")
    args = ap.parse_args()

    df = pd.read_csv(RESULTS).dropna(subset=["home_score", "away_score"])
    df = df[df["date"] >= "1990-01-01"].sort_values("date").reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int); df["away_score"] = df["away_score"].astype(int)

    elo = {}
    def R(t): return elo.get(t, INIT)

    def outcome(hs, as_): return 0 if hs > as_ else (1 if hs == as_ else 2)

    # ---- pass 1: collect calibration (sup_raw, outcome, total) on the calib window ----
    calib = []
    base_counts = np.zeros(3)
    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = m["home_score"], m["away_score"]
        hfa = 0 if m["neutral"] else HFA
        sup_raw = R(h) + hfa - R(a)
        oc = outcome(hs, as_)
        if args.calib_start <= m["date"] < args.test_start:
            calib.append((sup_raw, oc, hs + as_))
            base_counts[oc] += 1
        # update
        Eh = 1 / (1 + 10 ** (-sup_raw / 400)); Sh = [1, 0.5, 0][oc]
        d = KBASE * tw(m["tournament"]) * margin_mult(hs - as_) * (Sh - Eh)
        elo[h] = R(h) + d; elo[a] = R(a) - d

    base_rate = base_counts / base_counts.sum()
    T = float(np.mean([t for _, _, t in calib]))
    # fit C (Elo points per goal) by minimising calib RPS
    bestC, bestR = 150, 9.9
    for C in range(80, 281, 10):
        r = np.mean([rps(hda_probs(sr / C, T)[0], oc) for sr, oc, _ in calib])
        if r < bestR:
            bestR, bestC = r, C
    C = bestC
    print(f"Calibrated on {len(calib)} matches [{args.calib_start},{args.test_start}): "
          f"C={C} Elo/goal, T={T:.2f} goals, base-rate H/D/A={base_rate.round(3)}")

    # ---- pass 2: walk-forward test (re-init Elo, replay from scratch) ----
    elo = {}
    rows = []
    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = m["home_score"], m["away_score"]
        hfa = 0 if m["neutral"] else HFA
        sup_raw = R(h) + hfa - R(a)
        oc = outcome(hs, as_)
        if m["date"] >= args.test_start:
            p, p_o25, p_btts = hda_probs(sup_raw / C, T)
            rows.append(dict(date=m["date"], tournament=m["tournament"], neutral=bool(m["neutral"]),
                             oc=oc, total=hs + as_, btts=int(hs >= 1 and as_ >= 1),
                             pH=p[0], pD=p[1], pA=p[2], p_o25=p_o25, p_btts=p_btts,
                             rps=rps(p, oc), fav=p.max()))
        Eh = 1 / (1 + 10 ** (-sup_raw / 400)); Sh = [1, 0.5, 0][oc]
        d = KBASE * tw(m["tournament"]) * margin_mult(hs - as_) * (Sh - Eh)
        elo[h] = R(h) + d; elo[a] = R(a) - d

    bt = pd.DataFrame(rows)
    n = len(bt)
    ll = -np.mean([np.log(max(1e-9, [r.pH, r.pD, r.pA][r.oc])) for r in bt.itertuples()])
    brier3 = np.mean([(np.array([r.pH, r.pD, r.pA]) - np.eye(3)[r.oc]) ** 2 @ np.ones(3) for r in bt.itertuples()])
    b_o25 = np.mean((bt.p_o25 - (bt.total > 2.5)) ** 2)
    b_btts = np.mean((bt.p_btts - bt.btts) ** 2)
    base_rps = np.mean([rps(base_rate, o) for o in bt.oc])
    unif_rps = np.mean([rps(np.array([1/3, 1/3, 1/3]), o) for o in bt.oc])

    # bootstrap CI on RPS
    rng = np.random.default_rng(0)
    boots = [bt.rps.values[rng.integers(0, n, n)].mean() for _ in range(500)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\n=== WALK-FORWARD TEST: {n} matches since {args.test_start} ===")
    print(f"  Elo model RPS   : {bt.rps.mean():.4f}   (95% CI {lo:.4f}-{hi:.4f})")
    print(f"  base-rate  RPS  : {base_rps:.4f}   |  uniform RPS: {unif_rps:.4f}")
    skill = (base_rps - bt.rps.mean()) / base_rps * 100
    print(f"  -> RPS skill vs base-rate: {skill:+.1f}%   (good intl 1X2 RPS ~0.18-0.20)")
    print(f"  multiclass log-loss: {ll:.4f}  |  1X2 Brier: {brier3:.4f}")
    print(f"  Over-2.5 Brier: {b_o25:.4f}  |  BTTS Brier: {b_btts:.4f}  (vs 0.25 coin-flip)")

    print("\nBy competition:")
    bt["comp"] = bt.tournament.apply(lambda t: "World Cup" if t == "FIFA World Cup"
                                     else ("WC qual" if "World Cup qual" in str(t)
                                           else ("Friendly" if t == "Friendly" else "Other")))
    for c, g in bt.groupby("comp"):
        print(f"  {c:10s} n={len(g):>5}  RPS={g.rps.mean():.4f}")
    print("\nBy venue:  " + "  ".join(
        f"{'neutral' if k else 'home/away'} RPS={g.rps.mean():.4f} (n={len(g)})"
        for k, g in bt.groupby("neutral")))

    print("\nCalibration (favourite prob vs actual win-rate of the favourite):")
    bt["fav_oc"] = bt.apply(lambda r: int(np.argmax([r.pH, r.pD, r.pA]) == r.oc), axis=1)
    bt["fb"] = (bt.fav * 10).astype(int).clip(3, 9)
    for b, g in bt.groupby("fb"):
        print(f"  pred {b*10:>2}-{b*10+10}%  n={len(g):>4}  fav actually wins {100*g.fav_oc.mean():.0f}%")

    bt.to_csv("data/backtest_results.csv", index=False)
    print("\nLog -> data/backtest_results.csv")


if __name__ == "__main__":
    run()
