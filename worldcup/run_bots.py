"""
run_bots.py — Launch all three WorldCup trading bots concurrently + leaderboard tracker.

Usage:
    python3 run_bots.py
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trading-simulator-client"))

from trade_wc import WorldCupBot, BASE_URL
from trading_client import create_session
import aiohttp

GAMES = {
    170: {
        "contract_set": 1,
        "name": "Binary",
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJTSElLSEFSU0VIR0FMQFVDSElDQUdPLkVEVSIsImV4cCI6MTc4NDc2NTk0M30.NjzF9kRMfNJu9Ios1WL7RfajCwhvLso4_jLgQ7VIcOQ",
    },
    171: {
        "contract_set": 2,
        "name": "Points",
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJTSElLSEFSU0VIR0FMQFVDSElDQUdPLkVEVSIsImV4cCI6MTc4NDc2NTk2M30.AOzLK9R0xEKJFw5_XjcUIk2g535JIKteD4L3OPlEROg",
    },
    172: {
        "contract_set": 3,
        "name": "Goals",
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJTSElLSEFSU0VIR0FMQFVDSElDQUdPLkVEVSIsImV4cCI6MTc4NDc2NTk4Mn0.CbLpdSn94sSwGvcC5qF6oKNiEnaIAWwVDu7IvM0err4",
    },
}

MY_USER_ID = 1145
LEADERBOARD_INTERVAL = 60   # seconds between leaderboard prints
ARB_SCAN_INTERVAL = 15      # seconds between arb scans

# Goals locked in from completed group stage games (update as tournament progresses)
# These are floor values for Set3: any team with ask < locked_goals = guaranteed profit
REAL_GOALS: dict = {
    'Spain': 9, 'Germany': 9, 'Netherlands': 8, 'Switzerland': 6,
    'Japan': 5, 'Uruguay': 5, 'Brazil': 5, 'Argentina': 5,
    'Egypt': 4, 'Canada': 4, 'Austria': 4, 'Norway': 3,
    'England': 3, 'Portugal': 3, 'Sweden': 3, 'United States': 3,
    'Ecuador': 2, 'Turkey': 2, 'Iran': 2, 'Morocco': 2,
    'France': 2, 'Belgium': 2, 'Mexico': 2, 'South Korea': 2,
    'Ivory Coast': 2, 'Cabo Verde': 2, 'Australia': 2,
    'Scotland': 2, 'Paraguay': 2,
}


async def fetch_leaderboard(game_id: int, token: str, name: str) -> dict:
    """Fetch all accounts for a game and return ranked P&L."""
    h = {"Authorization": f"Bearer {token}"}
    try:
        async with aiohttp.ClientSession(headers=h) as s:
            # /fills gives our fills; use /account for our own P&L
            async with s.get(
                f"{BASE_URL}/api/games/trading-simulator/{game_id}/account"
            ) as r:
                if r.status == 200:
                    return await r.json()
    except Exception:
        pass
    return {}


async def leaderboard_loop():
    """Every LEADERBOARD_INTERVAL seconds, fetch & print P&L across all games."""
    await asyncio.sleep(10)  # let bots settle first
    while True:
        ts = time.strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"  LEADERBOARD SNAPSHOT  {ts}")
        print(f"{'='*60}")
        total_pnl = 0.0
        for gid, cfg in GAMES.items():
            acc = await fetch_leaderboard(gid, cfg["token"], cfg["name"])
            pnl = acc.get("pnl", 0.0)
            cash = acc.get("cash", 0.0)
            ro = acc.get("reduce_only", False)
            pos = {k: v for k, v in acc.get("positions", {}).items() if v != 0}
            total_pnl += pnl
            flag = " [REDUCE-ONLY]" if ro else ""
            print(f"  {cfg['name']:<8} game={gid}  pnl={pnl:+10.2f}  cash={cash:8.2f}{flag}")
            # print top positions
            top = sorted(pos.items(), key=lambda x: -abs(x[1]))[:8]
            if top:
                pos_str = "  ".join(f"{s}:{v:+d}" for s, v in top)
                print(f"           pos: {pos_str}")
        print(f"  {'TOTAL':8}            pnl={total_pnl:+10.2f}")
        print(f"{'='*60}\n")
        await asyncio.sleep(LEADERBOARD_INTERVAL)


async def arb_scanner_loop():
    """
    Every ARB_SCAN_INTERVAL seconds, scan all three orderbooks for mechanical arbs:
      1. Set3 floor arb: any team where ask < locked goals scored → guaranteed profit
      2. Set1 basket sum: if sum(best_asks) < 95 → buy basket; if sum(best_bids) > 105 → sell basket
      3. Cross-game Set2 vs Set1: if Set1_bid > 0 but Set2_ask < 2 → Set2 is free money
    Prints alerts and size; execution happens via the individual bots' quote_symbol loops.
    """
    await asyncio.sleep(20)  # let bots connect first
    tok172 = GAMES[172]["token"]
    tok170 = GAMES[170]["token"]
    tok171 = GAMES[171]["token"]

    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{BASE_URL}/api/games/trading-simulator/172/orderbooks",
                                 headers={"Authorization": f"Bearer {tok172}"}) as r:
                    books172 = await r.json() if r.status == 200 and 'json' in r.content_type else {}
                async with s.get(f"{BASE_URL}/api/games/trading-simulator/170/orderbooks",
                                 headers={"Authorization": f"Bearer {tok170}"}) as r:
                    books170 = await r.json() if r.status == 200 and 'json' in r.content_type else {}
                async with s.get(f"{BASE_URL}/api/games/trading-simulator/171/orderbooks",
                                 headers={"Authorization": f"Bearer {tok171}"}) as r:
                    books171 = await r.json() if r.status == 200 and 'json' in r.content_type else {}

            found_any = False

            # 1. Set3 floor arb
            for sym, book in books172.items():
                locked = REAL_GOALS.get(sym, 0)
                if not locked:
                    continue
                asks = {float(k): v for k, v in book.get('asks', {}).items() if v > 0}
                ba = min(asks) if asks else None
                if ba is not None and ba < locked:
                    profit = locked - ba
                    qty = min(100, asks[ba])  # max available at that price
                    print(f"[ARB] *** Set3 FLOOR: BUY {sym} @ {ba:.2f} (locked={locked} goals) "
                          f"→ +{profit:.2f}/ct guaranteed  avail={qty}")
                    found_any = True

            # 2. Set1 basket sum check
            bids_sum = asks_sum = 0.0
            n_bids = n_asks = 0
            for sym, book in books170.items():
                bids = {float(k): v for k, v in book.get('bids', {}).items() if v > 0}
                asks = {float(k): v for k, v in book.get('asks', {}).items() if v > 0}
                if bids: bids_sum += max(bids); n_bids += 1
                if asks: asks_sum += min(asks); n_asks += 1
            if n_bids >= 40:
                if bids_sum > 105:
                    print(f"[ARB] *** Set1 BASKET SELL: sum_bids={bids_sum:.1f} > 105 → sell all 48 teams")
                    found_any = True
                if asks_sum < 95:
                    print(f"[ARB] *** Set1 BASKET BUY: sum_asks={asks_sum:.1f} < 95 → buy all 48 teams")
                    found_any = True

            # 3. Set2 vs Set1 cross-game: Set1_bid>0 but Set2_ask<2 (Set2 must be >= 2 if team can win)
            for sym in books170:
                b1 = books170.get(sym, {}); b2 = books171.get(sym, {})
                bids1 = {float(k): v for k, v in b1.get('bids', {}).items() if v > 0}
                asks2 = {float(k): v for k, v in b2.get('asks', {}).items() if v > 0}
                bid1 = max(bids1) if bids1 else 0
                ask2 = min(asks2) if asks2 else None
                if bid1 > 1.0 and ask2 is not None and ask2 < 2.0:
                    print(f"[ARB] *** Cross-game S1vS2: {sym} Set1_bid={bid1:.2f}>0 "
                          f"but Set2_ask={ask2:.2f}<2 → BUY Set2")
                    found_any = True

            if not found_any:
                pass  # silent when no arbs — keeps log clean

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[ARB] scanner error: {e}")

        await asyncio.sleep(ARB_SCAN_INTERVAL)


async def run_one(game_id: int, cfg: dict):
    """Run a single bot, auto-restarting on any error (transient WS drops etc)."""
    import sys
    # Before each reprice, pull fresh Odds API scores (--no-niche = 3 credits only)
    here = os.path.dirname(os.path.abspath(__file__))
    refresh_cmd = [sys.executable, os.path.join(here, "refresh.py"), "--no-niche"]

    backoff = 2.0
    while True:
        try:
            async with create_session() as session:
                bot = WorldCupBot(
                    session=session,
                    game_id=game_id,
                    token=cfg["token"],
                    contract_set=cfg["contract_set"],
                    reprice_mins=30,
                    refresh_cmd=refresh_cmd,
                    base_url=BASE_URL,
                )
                await bot.start()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[run_bots] game {game_id} crashed ({e}), restarting in {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)  # cap at 30s
        else:
            backoff = 2.0  # reset on clean exit


async def main():
    tasks = [
        asyncio.create_task(run_one(gid, cfg))
        for gid, cfg in GAMES.items()
    ]
    tasks.append(asyncio.create_task(leaderboard_loop()))
    tasks.append(asyncio.create_task(arb_scanner_loop()))

    for gid, cfg in GAMES.items():
        print(f"[run_bots] Launched game {gid} ({cfg['name']}, Set {cfg['contract_set']})")
    print("[run_bots] Leaderboard tracker active (every 60s). Ctrl-C to stop.\n")

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[run_bots] Shutting down...")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
