"""
backtest_ablation.py — Does each new feature actually IMPROVE the model?

Walk-forward (no leakage) over 49k international results, adding features to the Elo
1X2 base one at a time and measuring the change in mean RPS, with a paired bootstrap
CI on the difference. A feature is "kept" only if it lowers RPS with a CI excluding 0.

Features tested (both derivable from intl_results.csv — no new data needed):
  * rest_diff : (home_rest_days - away_rest_days), capped — fixture-congestion edge.
  * h2h_gd    : mean goal-diff in prior meetings, oriented to the current home team.

Coefficients are fit ONLY on the calibration window [2014,2018); the test window
[2018,now) is scored once with frozen coefficients. Mirrors backtest.py's protocol.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from backtest import pois, hda_probs, margin_mult, tw, rps, INIT, HFA, KBASE

RESULTS = "data/intl_results.csv"
CALIB_START, TEST_START = "2014-01-01", "2018-01-01"
REST_CAP = 14


def collect():
    df = pd.read_csv(RESULTS).dropna(subset=["home_score", "away_score"])
    df = df[df["date"] >= "1990-01-01"].sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int); df["away_score"] = df["away_score"].astype(int)

    elo = {}; last_date = {}; h2h = defaultdict(lambda: deque(maxlen=10))
    def R(t): return elo.get(t, INIT)

    calib, test = [], []
    for _, m in df.iterrows():
        h, a, d = m["home_team"], m["away_team"], m["date"]
        hs, as_ = m["home_score"], m["away_score"]
        oc = 0 if hs > as_ else (1 if hs == as_ else 2)
        hfa = 0 if m["neutral"] else HFA
        sup_raw = R(h) + hfa - R(a)
        # rest-days differential
        def rest(t):
            if t not in last_date:
                return REST_CAP
            return min(REST_CAP, (pd.Timestamp(d) - pd.Timestamp(last_date[t])).days)
        rest_diff = rest(h) - rest(a)
        # head-to-head goal diff oriented to current home team
        pair = frozenset((h, a))
        prior = h2h[pair]
        h2h_gd = np.mean([gd if ht == h else -gd for ht, gd in prior]) if prior else 0.0
        rec = (sup_raw, oc, hs + as_, rest_diff, float(h2h_gd))
        if CALIB_START <= d < TEST_START:
            calib.append(rec)
        elif d >= TEST_START:
            test.append(rec)
        # updates
        Eh = 1 / (1 + 10 ** (-sup_raw / 400)); Sh = [1, 0.5, 0][oc]
        delta = KBASE * tw(m["tournament"]) * margin_mult(hs - as_) * (Sh - Eh)
        elo[h] = R(h) + delta; elo[a] = R(a) - delta
        last_date[h] = d; last_date[a] = d
        h2h[pair].append((h, hs - as_))
    return calib, test


def rps_for(records, C, b_rest, b_h2h, T):
    tot = 0.0
    for sup_raw, oc, _, rest_diff, h2h_gd in records:
        sup = sup_raw / C + b_rest * rest_diff + b_h2h * h2h_gd
        tot += rps(hda_probs(sup, T)[0], oc)
    return tot / len(records)


def main():
    calib, test = collect()
    T = float(np.mean([t for _, _, t, _, _ in calib]))
    # fit C (baseline), then each feature coef, on calibration RPS
    C = min(range(80, 281, 10), key=lambda c: rps_for(calib, c, 0, 0, T))
    b_rest = min(np.arange(-0.03, 0.0301, 0.005), key=lambda b: rps_for(calib, C, b, 0, T))
    b_h2h = min(np.arange(-0.05, 0.3001, 0.025), key=lambda b: rps_for(calib, C, 0, b, T))
    print(f"Calibrated: C={C} Elo/goal, T={T:.2f}, b_rest={b_rest:+.3f} g/day, b_h2h={b_h2h:+.3f}")
    print(f"(calib n={len(calib)}, test n={len(test)})\n")

    variants = {
        "V0 Elo only":        (0.0, 0.0),
        "V1 +rest-days":      (b_rest, 0.0),
        "V2 +head-to-head":   (0.0, b_h2h),
        "V3 +both":           (b_rest, b_h2h),
    }
    # per-match RPS arrays for paired bootstrap
    arr = {}
    for name, (br, bh) in variants.items():
        arr[name] = np.array([rps(hda_probs(sr / C + br * rd + bh * hg, T)[0], oc)
                              for sr, oc, _, rd, hg in test])
    base = arr["V0 Elo only"]
    rng = np.random.default_rng(0); n = len(base)
    print(f"{'variant':20s} {'RPS':>8} {'Δ vs V0':>10} {'95% CI of Δ':>22} {'verdict':>9}")
    for name, a in arr.items():
        diff = a - base
        if name == "V0 Elo only":
            print(f"{name:20s} {a.mean():8.4f} {'—':>10} {'(baseline)':>22}")
            continue
        boots = [diff[rng.integers(0, n, n)].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        keep = "KEEP" if hi < 0 else ("hurts" if lo > 0 else "noise")
        print(f"{name:20s} {a.mean():8.4f} {diff.mean():+10.5f} {f'[{lo:+.5f}, {hi:+.5f}]':>22} {keep:>9}")
    print("\n(Δ negative = lower RPS = better. KEEP only if the whole CI is < 0.)")


if __name__ == "__main__":
    main()
