"""
trade_wc.py — Trading bot for the World Cup futures competition.

Sits on top of trading_client.Client (the DRW games SDK) and trades each team
contract toward the model fair value produced by simulate_tournament.py.

THREE markets, each (almost certainly) a separate game_id with one symbol per
team. You run ONE bot per market and tell it which contract set it is:

    set 1 — Winner      : symbol settles 100 (champion) / 0          col=set1_winner_fair
    set 2 — Advancement : symbol settles 0..64 by finishing stage    col=set2_advance_fair
    set 3 — Total goals : symbol settles accumulated goals (group+KO) col=set3_goals_fair

Strategy
--------
For each team symbol every tick:
  * fair value (fv) comes from the Monte Carlo (wc_contract_fair_values.csv).
  * model uncertainty (edge buffer) scales with that contract's spread (winner
    market is the most skewed/uncertain, goals market the noisiest in absolute
    points). We only TAKE when the book is beyond fv by more than the buffer.
  * we also QUOTE passively a buffer-width around fv, sized to remaining capacity,
    to earn the spread when the book is near fair.
Risk
----
  * |position| capped at MAX_POS (100) per symbol.
  * Below the market's P&L threshold the bot flips to REDUCE-ONLY: it will only
    send orders that shrink an existing position (never grows risk). Thresholds:
    set1 -500k, set2 -300k, set3 -100k.
  * As results land the model is re-run and fair values reloaded live (call
    reload_fairs()); each result collapses uncertainty and should move quotes —
    which is exactly what the competition rewards.

Two platform specifics to confirm against the live API the first time you run
(they don't change the logic, only the wire format):
  1. Symbol naming — assumed display_symbol == team name within a market. Print
     `client.order_books.keys()` once connected and adjust SYMBOL_OVERRIDES if not.
  2. order_type / side encoding — send_order(display_symbol, px, qty, order_type).
     We pass positive qty for buys, negative for sells, order_type=ORDER_TYPE.
     If the platform wants side in order_type instead, set SIDE_IN_ORDER_TYPE=True.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

from size_positions import (Sizer, MAX_POS, SET_PNL_THRESHOLD as ABS_THRESHOLD,
                            SET_PRICE_RANGE, SET_NAME, RISK_FRAC)

HERE = os.path.dirname(os.path.abspath(__file__))
FAIRS = os.path.join(HERE, "data", "wc_contract_fair_values.csv")

# The DRW games SDK lives in the sibling trading-simulator-client/ dir. Add it to
# the path; fall back to a stub base class so --paper works even where the SDK
# (or aiohttp) is unavailable — live trading then fails with a clear message.
sys.path.insert(0, os.path.join(HERE, "..", "trading-simulator-client"))
try:
    from trading_client import Client
except Exception as _e:           # noqa: BLE001 — any import problem -> paper-only
    _CLIENT_IMPORT_ERROR = _e

    class Client:                 # minimal stub; live trading unavailable
        def __init__(self, *a, **k):
            raise RuntimeError(
                f"trading_client unavailable ({_CLIENT_IMPORT_ERROR}); "
                "live trading needs the DRW SDK + network. Use --paper here.")
else:
    _CLIENT_IMPORT_ERROR = None

SET_COLUMN = {1: "set1_winner_fair", 2: "set2_advance_fair", 3: "set3_goals_fair"}
# Reduce-only kicks in at the (negative) P&L threshold.
SET_PNL_THRESHOLD = {k: -v for k, v in ABS_THRESHOLD.items()}

# Passive quote half-width, a fraction of that contract's price range (post bid/ask
# this far off fair to earn the spread when the book is near fair). The TAKE side is
# governed by the Sizer's own edge gate (MIN_EDGE_FRAC), so no separate take buffer.
SET_QUOTE_FRAC = {1: 0.04, 2: 0.035, 3: 0.04}
QUOTE_SIZE = 5          # passive order size per quote
TAKE_SIZE = 15          # max size to cross per tick when chasing a target

# Live reprice: re-run the simulation (and optionally an upstream data refresh) on
# an interval AND whenever a result notification lands, then hot-reload fair values.
# This is the "react to new information" edge — each result collapses uncertainty.
SIM = os.path.join(HERE, "simulate_tournament.py")
REPRICE_SIMS = 100_000          # sims per live reprice
RESULT_KEYWORDS = ("full time", "final", "result", "advances", "eliminated",
                   "wins", "knocked out", "qualifies")

SYMBOL_OVERRIDES: Dict[str, str] = {}   # team -> platform symbol, if they differ
SIDE_IN_ORDER_TYPE = False
ORDER_TYPE = "LIMIT"


class WorldCupBot(Client):
    def __init__(self, session, game_id, token, contract_set: int,
                 reprice_mins: int = 0, refresh_cmd: Optional[List[str]] = None):
        super().__init__(session, game_id, token)
        assert contract_set in (1, 2, 3), "contract_set must be 1, 2 or 3"
        self.cset = contract_set
        self.lo, self.hi = SET_PRICE_RANGE[contract_set]
        self.threshold = SET_PNL_THRESHOLD[contract_set]
        self.qwidth = SET_QUOTE_FRAC[contract_set] * (self.hi - self.lo)
        self.reprice_mins = reprice_mins       # 0 = no periodic reprice
        self.refresh_cmd = refresh_cmd         # optional upstream data refresh before re-sim
        self._repricing = False
        self.sizer = Sizer()                   # distribution-aware position sizing
        self.fairs: Dict[str, float] = {}
        self.team_of: Dict[str, str] = {}      # symbol -> team
        self.realized = 0.0                    # realized P&L from fills
        self.avg_cost: Dict[str, float] = {}   # symbol -> avg entry price
        self.reload_fairs()

    # ---- fair values ----------------------------------------------------
    def reload_fairs(self) -> None:
        """(Re)load model fair values + distributions for this set. Call after a re-sim."""
        self.sizer = Sizer()
        df = pd.read_csv(FAIRS)
        col = SET_COLUMN[self.cset]
        self.fairs, self.team_of = {}, {}
        for _, r in df.iterrows():
            team = r["team"]
            sym = SYMBOL_OVERRIDES.get(team, team)
            self.fairs[sym] = float(r[col])
            self.team_of[sym] = team
        print(f"[{SET_NAME[self.cset]}] loaded {len(self.fairs)} fair values "
              f"(quote width {self.qwidth:.2f})")

    def fair(self, symbol: str) -> Optional[float]:
        return self.fairs.get(symbol)

    async def reprice(self) -> None:
        """Re-run the simulation (after an optional data refresh) and hot-reload
        fair values. Safe to call concurrently — coalesces into one run."""
        if self._repricing:
            return
        self._repricing = True
        try:
            if self.refresh_cmd:
                print(f"[{SET_NAME[self.cset]}] refreshing data: {' '.join(self.refresh_cmd)}")
                p = await asyncio.create_subprocess_exec(*self.refresh_cmd)
                await p.wait()
            print(f"[{SET_NAME[self.cset]}] repricing ({REPRICE_SIMS:,} sims) ...")
            p = await asyncio.create_subprocess_exec(
                sys.executable, SIM, str(REPRICE_SIMS),
                stdout=asyncio.subprocess.DEVNULL)
            await p.wait()
            self.reload_fairs()
            print(f"[{SET_NAME[self.cset]}] reprice complete — fair values reloaded")
        except Exception as e:
            print(f"[{SET_NAME[self.cset]}] reprice failed: {e}")
        finally:
            self._repricing = False

    async def _reprice_loop(self) -> None:
        while self.reprice_mins > 0:
            await asyncio.sleep(self.reprice_mins * 60)
            await self.reprice()

    # ---- P&L / risk -----------------------------------------------------
    def estimated_pnl(self) -> float:
        """Realized + mark-to-model unrealized P&L, vs the market threshold."""
        unreal = 0.0
        for sym, pos in self.positions.items():
            fv = self.fair(sym)
            if fv is None or pos == 0:
                continue
            unreal += pos * (fv - self.avg_cost.get(sym, fv))
        return self.realized + unreal

    def reduce_only(self) -> bool:
        return self.estimated_pnl() <= self.threshold

    def portfolio_ok(self) -> bool:
        """Stop adding risk once summed adverse-case tail loss nears the budget."""
        prices = {self.team_of[s]: self.avg_cost.get(s, self.fair(s) or 0.0)
                  for s in self.positions}
        teams = {self.team_of[s]: p for s, p in self.positions.items()}
        risk = self.sizer.portfolio_tail_risk(self.cset, teams, prices)
        return risk["utilization"] < RISK_FRAC      # headroom under the threshold

    def _clamp(self, px: float) -> float:
        return max(self.lo, min(self.hi, round(px, 2)))

    async def _order(self, symbol: str, px: float, qty: int) -> None:
        """qty>0 buy, qty<0 sell. Honors reduce-only, portfolio risk, position caps."""
        if qty == 0:
            return
        pos = self.positions.get(symbol, 0)
        new_pos = pos + qty
        if abs(new_pos) > MAX_POS:                       # respect position cap
            qty = (MAX_POS if qty > 0 else -MAX_POS) - pos
            if qty == 0:
                return
        adds_risk = abs(pos + qty) > abs(pos)
        if adds_risk and (self.reduce_only() or not self.portfolio_ok()):
            return                                       # at a risk limit: reduce only
        ot = ("BID" if qty > 0 else "ASK") if SIDE_IN_ORDER_TYPE else ORDER_TYPE
        try:
            await self.send_order(symbol, self._clamp(px), abs(qty)
                                  if SIDE_IN_ORDER_TYPE else qty, ot)
        except Exception as e:           # never let one bad order kill the loop
            print(f"[{SET_NAME[self.cset]}] order {symbol} {qty}@{px:.2f} failed: {e}")

    # ---- core policy ----------------------------------------------------
    async def quote_symbol(self, symbol: str) -> None:
        fv = self.fair(symbol)
        team = self.team_of.get(symbol)
        if fv is None or team is None:
            return
        book = self.order_books.get(symbol)
        pos = self.positions.get(symbol, 0)
        best_bid = book.best_bid_px if book else None
        best_ask = book.best_ask_px if book else None

        # 1) TAKE toward the Sizer's distribution-aware target. The Sizer evaluates
        #    edge vs price (with its own MIN_EDGE gate) and returns a signed target
        #    position; we move toward it, capped at TAKE_SIZE per tick.
        if best_ask is not None:
            tgt, _ = self.sizer.target_position(self.cset, team, best_ask)
            if tgt > pos:                                # buying is justified at the ask
                await self._order(symbol, best_ask, min(TAKE_SIZE, tgt - pos))
        if best_bid is not None:
            tgt, _ = self.sizer.target_position(self.cset, team, best_bid)
            if tgt < pos:                                # selling is justified at the bid
                await self._order(symbol, best_bid, max(-TAKE_SIZE, tgt - pos))

        # 2) QUOTE passively around fair, skewed against inventory, to earn spread
        #    when the book sits near fair. Capped so passive fills can't exceed the
        #    Sizer's target magnitude at the quote price.
        skew = self.qwidth * (pos / MAX_POS) * 0.5
        bid_px, ask_px = fv - self.qwidth - skew, fv + self.qwidth - skew
        tgt_bid, _ = self.sizer.target_position(self.cset, team, bid_px)
        tgt_ask, _ = self.sizer.target_position(self.cset, team, ask_px)
        if tgt_bid > pos:
            await self._order(symbol, bid_px, min(QUOTE_SIZE, tgt_bid - pos))
        if tgt_ask < pos:
            await self._order(symbol, ask_px, max(-QUOTE_SIZE, tgt_ask - pos))

    async def on_fills(self, new_fills) -> None:
        for f in new_fills:
            sym = f.display_symbol
            qty = f.traded_qty                           # signed: + buy, - sell
            pos = self.positions.get(sym, 0)
            prev_cost = self.avg_cost.get(sym, f.px)
            if pos == 0 or (pos > 0) == (qty > 0):       # opening / adding
                tot = abs(pos) + abs(qty)
                self.avg_cost[sym] = (abs(pos) * prev_cost + abs(qty) * f.px) / tot if tot else f.px
            else:                                        # closing -> realize P&L
                closed = min(abs(qty), abs(pos))
                self.realized += closed * (f.px - prev_cost) * (1 if pos > 0 else -1)

    async def on_notification(self, message: str) -> None:
        """A result posting collapses uncertainty — reprice immediately."""
        await super().on_notification(message)
        if any(k in message.lower() for k in RESULT_KEYWORDS):
            print(f"[{SET_NAME[self.cset]}] result notification -> repricing")
            await self.reprice()

    async def on_start(self) -> None:
        print(f"[{SET_NAME[self.cset]}] trading game {self.web_url}")
        if self.reprice_mins > 0:
            asyncio.create_task(self._reprice_loop())
        while True:
            for symbol in list(self.fairs.keys()):
                await self.quote_symbol(symbol)
            if self.reduce_only():
                print(f"[{SET_NAME[self.cset]}] REDUCE-ONLY "
                      f"(pnl≈{self.estimated_pnl():,.0f} ≤ {self.threshold:,.0f})")
            await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# paper mode — exercise the FULL decision path locally (no DRW network needed)
# against a synthetic order book, using the exact Sizer logic the live bot uses.
# ---------------------------------------------------------------------------
def paper_run(contract_set: int, mispricing: float = 0.30, seed: int = 7) -> None:
    import numpy as np
    sz = Sizer()
    col = SET_COLUMN[contract_set]
    df = pd.read_csv(FAIRS).sort_values(col, ascending=False)
    lo, hi = SET_PRICE_RANGE[contract_set]
    half = max(0.5, 0.01 * (hi - lo))      # realistic TIGHT market half-spread
    rng = np.random.default_rng(seed)

    print(f"\n=== PAPER {SET_NAME[contract_set]} (set {contract_set}) — synthetic book, "
          f"±{mispricing:.0%} random mispricing, {2*half:.1f}-wide book ===")
    print(f"{'team':<20}{'fair':>7}{'mid':>7}{'bid':>7}{'ask':>7}{'action':>9}{'tgt':>6}  binding")
    book_pos: Dict[str, int] = {}
    prices: Dict[str, float] = {}
    n_trades = 0
    for _, r in df.head(20).iterrows():
        team = r["team"]
        fair = float(r[col])
        # synthetic mid mispriced vs fair; tight book around it
        mid = float(np.clip(fair * (1 + rng.uniform(-mispricing, mispricing)), lo, hi))
        bid, ask = round(max(lo, mid - half), 2), round(min(hi, mid + half), 2)
        tgt_buy, d_buy = sz.target_position(contract_set, team, ask)
        tgt_sell, d_sell = sz.target_position(contract_set, team, bid)
        action, tgt, dgn = "—", 0, {}
        if tgt_buy > 0:
            action, tgt, dgn = f"BUY@{ask:.2f}", tgt_buy, d_buy
        elif tgt_sell < 0:
            action, tgt, dgn = f"SELL@{bid:.2f}", tgt_sell, d_sell
        if tgt != 0:
            book_pos[team] = tgt; prices[team] = ask if tgt > 0 else bid; n_trades += 1
        print(f"{team:<20}{fair:>7.2f}{mid:>7.2f}{bid:>7.2f}{ask:>7.2f}"
              f"{action:>9}{tgt:>6}  {dgn.get('binding','-')}")
    risk = sz.portfolio_tail_risk(contract_set, book_pos, prices)
    print(f"  -> {n_trades} positions, adverse-case loss ≈ {risk['total_tail_loss']:,.0f}"
          f" / {risk['threshold']:,.0f} ({risk['utilization']:.0%} of threshold)")


# ---------------------------------------------------------------------------
# entrypoint — live on the DRW network, or --paper locally
# ---------------------------------------------------------------------------
async def run_market(token: str, game_id: int, contract_set: int, reprice_mins: int = 0):
    from trading_client import create_session
    async with create_session() as session:
        bot = WorldCupBot(session, game_id, token, contract_set, reprice_mins=reprice_mins)
        await bot.start()


if __name__ == "__main__":
    if "--paper" in sys.argv:
        # usage: python3 trade_wc.py --paper [set 1|2|3 | all]
        rest = [a for a in sys.argv[1:] if a != "--paper"]
        which = rest[0] if rest else "all"
        for cs in ([1, 2, 3] if which == "all" else [int(which)]):
            paper_run(cs)
        sys.exit(0)

    TOKEN = os.environ.get("WC_TOKEN", "PASTE_TOKEN_HERE")
    # usage: python3 trade_wc.py <game_id> <contract_set 1|2|3> [reprice_mins]
    gid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cset = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rmin = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    asyncio.run(run_market(TOKEN, gid, cset, rmin))
