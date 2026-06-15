"""
size_positions.py — Distribution-aware position sizing against the P&L thresholds.

simulate_tournament.py emits the FULL settlement distribution for every team in
every contract (wc_contract_distributions.json). This module turns that
distribution + a quoted price into a recommended position, balancing:

  * EDGE      — (fair - price) for a long, (price - fair) for a short.
  * VARIANCE  — fractional-Kelly sizing: n* ≈ kelly · edge / Var(S) · bankroll,
                with bankroll = the market's |P&L threshold| (the drawdown budget).
  * TAIL RISK — a per-symbol worst-case cap using the ACTUAL adverse quantile of
                the settlement (not a Gaussian σ), since these payoffs are very
                skewed (Set 1 is Bernoulli, Set 2 is a step ladder). We never let
                one name's bad-case loss exceed |threshold| / RISK_SLOTS.
  * HARD CAP  — |position| ≤ MAX_POS (100).

Settlement distributions per contract:
  Set 1 (Winner)      : 100 w.p. P(champion), else 0.
  Set 2 (Advancement) : step ladder {0,2,4,8,16,24,32,64} w/ the simulated stage probs.
  Set 3 (Total goals) : the simulated per-team goal histogram.

CLI: `python3 size_positions.py` prints, for each contract, the recommended
position if the market were quoting at the model fair value ± a test mispricing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FAIRS = os.path.join(HERE, "data", "wc_contract_fair_values.csv")
DIST = os.path.join(HERE, "data", "wc_contract_distributions.json")

MAX_POS = 100
SET_PNL_THRESHOLD = {1: 500_000.0, 2: 300_000.0, 3: 100_000.0}   # |threshold|
SET_PRICE_RANGE = {1: (0.0, 100.0), 2: (0.0, 64.0), 3: (0.0, 100.0)}
SET_NAME = {1: "Winner", 2: "Advancement", 3: "Total Goals"}

# Stage payout ladder for Set 2.
STAGE_PAYOUT = {"group_exit": 0, "lose_R32": 2, "lose_R16": 4, "lose_QF": 8,
                "fourth": 16, "third": 24, "runner_up": 32, "champion": 64}

KELLY_FRAC = 0.30        # fraction of full Kelly (conservative)
RISK_SLOTS = 25.0        # assume up to ~this many concurrent meaningful positions
RISK_FRAC = 0.5          # use at most this fraction of the threshold as tail budget
ADVERSE_Q = 0.05         # 5th/95th percentile defines the "bad case"
MIN_EDGE_FRAC = 0.03     # ignore edges smaller than this fraction of price range


@dataclass
class Dist:
    values: np.ndarray   # settlement outcomes
    probs: np.ndarray    # probabilities (sum to 1)

    @property
    def mean(self) -> float:
        return float((self.values * self.probs).sum())

    @property
    def var(self) -> float:
        m = self.mean
        return float((self.probs * (self.values - m) ** 2).sum())

    def quantile(self, alpha: float) -> float:
        """Lowest value whose cumulative prob >= alpha."""
        order = np.argsort(self.values)
        v, p = self.values[order], self.probs[order]
        cum = np.cumsum(p)
        i = int(np.searchsorted(cum, alpha))
        return float(v[min(i, len(v) - 1)])


class Sizer:
    def __init__(self):
        self.fairs = pd.read_csv(FAIRS)
        with open(DIST) as f:
            self.dist = json.load(f)

    def settlement_dist(self, cset: int, team: str) -> Dist:
        d = self.dist[team]
        if cset == 1:
            p = d["stage_probs"]["champion"]
            return Dist(np.array([0.0, 100.0]), np.array([1 - p, p]))
        if cset == 2:
            names = list(STAGE_PAYOUT.keys())
            vals = np.array([STAGE_PAYOUT[n] for n in names], dtype=float)
            probs = np.array([d["stage_probs"][n] for n in names], dtype=float)
            return Dist(vals, probs / probs.sum())
        if cset == 3:
            hist = d["goals_hist"]
            vals = np.array([float(k) for k in hist.keys()])
            cnts = np.array([float(v) for v in hist.values()])
            return Dist(vals, cnts / cnts.sum())
        raise ValueError(cset)

    def target_position(self, cset: int, team: str, price: float) -> Tuple[int, dict]:
        """
        Recommended SIGNED target position at `price` (>0 long, <0 short, 0 flat).
        Returns (position, diagnostics).
        """
        S = self.settlement_dist(cset, team)
        fair, var = S.mean, S.var
        lo, hi = SET_PRICE_RANGE[cset]
        rng = hi - lo
        bankroll = SET_PNL_THRESHOLD[cset]
        tail_budget = RISK_FRAC * bankroll / RISK_SLOTS

        long_edge = fair - price
        short_edge = price - fair
        min_edge = MIN_EDGE_FRAC * rng

        if max(long_edge, short_edge) < min_edge or var <= 1e-9:
            return 0, dict(fair=fair, reason="edge<min or zero variance")

        if long_edge >= short_edge:
            side, edge = +1, long_edge
            adverse = S.quantile(ADVERSE_Q)            # bad case for a long: low settle
            loss_per = max(price - adverse, 1e-6)
        else:
            side, edge = -1, short_edge
            adverse = S.quantile(1 - ADVERSE_Q)        # bad case for a short: high settle
            loss_per = max(adverse - price, 1e-6)

        n_kelly = KELLY_FRAC * (edge / var) * bankroll
        n_risk = tail_budget / loss_per
        n = int(min(MAX_POS, n_kelly, n_risk))
        diag = dict(fair=round(fair, 2), edge=round(edge, 2), sd=round(var ** 0.5, 2),
                    adverse=round(adverse, 2), loss_per=round(loss_per, 2),
                    n_kelly=round(n_kelly, 1), n_risk=round(n_risk, 1),
                    binding=("kelly" if n_kelly <= min(n_risk, MAX_POS)
                             else "risk" if n_risk <= MAX_POS else "cap"))
        return side * n, diag


    def portfolio_tail_risk(self, cset: int,
                            positions: Dict[str, int],
                            prices: Dict[str, float]) -> dict:
        """
        Sum the per-name adverse-case (5%/95% quantile) loss across all open
        positions in one market. If this approaches |threshold| you are over-
        exposed even if every single name is within its own cap. Note: this is a
        conservative SUM of marginal tails (ignores diversification across teams),
        so it's an upper bound on simultaneous bad-case loss.
        """
        total = 0.0
        rows = []
        for team, pos in positions.items():
            if pos == 0:
                continue
            S = self.settlement_dist(cset, team)
            px = prices[team]
            if pos > 0:
                adverse = S.quantile(ADVERSE_Q)        # long bad case: low settle
                loss = pos * max(px - adverse, 0.0)
            else:
                adverse = S.quantile(1 - ADVERSE_Q)    # short bad case: high settle
                loss = (-pos) * max(adverse - px, 0.0)
            total += loss
            rows.append((team, pos, round(loss, 0)))
        thr = SET_PNL_THRESHOLD[cset]
        return dict(total_tail_loss=round(total, 0), threshold=thr,
                    utilization=round(total / thr, 3), positions=rows)


def _demo():
    sz = Sizer()
    cols = {1: 'winner', 2: 'advance', 3: 'goals'}
    for cset in (1, 2, 3):
        print(f"\n=== Set {cset} ({SET_NAME[cset]}) — targets at a 35% mispricing ===")
        print(f"{'team':<20}{'fair':>7}{'price':>7}{'pos':>6}{'sd':>6}"
              f"{'n_kelly':>9}{'n_risk':>8}  binding  scenario")
        top = sz.fairs.sort_values(f"set{cset}_{cols[cset]}_fair", ascending=False).head(6)
        targets, prices = {}, {}
        for k, (_, r) in enumerate(top.iterrows()):
            team = r["team"]
            fair = sz.settlement_dist(cset, team).mean
            # alternate: half the names cheap (go long), half rich (go short)
            cheap = k % 2 == 0
            price = max(0.0, fair * (0.65 if cheap else 1.35))
            pos, d = sz.target_position(cset, team, price)
            targets[team] = pos; prices[team] = price
            print(f"{team:<20}{d['fair']:>7.2f}{price:>7.2f}{pos:>6}{d.get('sd',0):>6.1f}"
                  f"{d.get('n_kelly',0):>9}{d.get('n_risk',0):>8}  {d.get('binding','-'):<7}"
                  f"  {'cheap→long' if cheap else 'rich→short'}")
        risk = sz.portfolio_tail_risk(cset, targets, prices)
        print(f"  portfolio adverse-case loss ≈ {risk['total_tail_loss']:,.0f} "
              f"/ {risk['threshold']:,.0f}  ({risk['utilization']:.0%} of threshold)")


if __name__ == "__main__":
    _demo()
