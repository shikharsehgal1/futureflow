"""
simulate_tournament.py — Monte Carlo of the full 2026 World Cup, pricing all three
trading-competition contract sets off a SINGLE simulation.

Every contract is a deterministic function of each team's trajectory through the
48-team bracket, so one simulation prices everything:

  Set 1 — Winner       (settles 100 champion / 0 else)  -> fair = P(champion) * 100
  Set 2 — Advancement  (settles 0..64 by finishing stage) -> fair = E[stage payout]
  Set 3 — Total goals  (settles accumulated goals, group+KO) -> fair = E[goals]

Goal model (per the build decision):
  * GROUP games: de-vigged Pinnacle lambdas from wc_match_summary.csv where present
    (sharp); Elo-derived lambdas (build_elo_v2 supremacy formula) as fallback.
  * KNOCKOUT games (bracket not yet drawn): Elo-derived lambdas, neutral venue.
  * Extra-time goals COUNT toward Set 3; penalty-shootout goals do NOT; own goals
    are not credited (immaterial here — we never simulate OGs).

Bracket + groups are the official 2026 draw (FIFA / Wikipedia, Dec 5 2025 draw):
  groups A-L, R32 slot map (matches 73-88), R16/QF/SF connections, and the
  best-third slot-allocation lists (each third-slot accepts thirds from 5 groups;
  the perfect matching reproduces FIFA Annex C per qualifying-group combination).

Outputs:
  data/wc_contract_fair_values.csv   — one row per team: fair value + risk stats
                                        for all three contracts.
  data/wc_contract_distributions.json — full stage / goal distributions per team
                                        (for position sizing against the P&L thresholds).

Pure numpy/pandas. Vectorized across all sims.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SUMMARY = os.path.join(DATA, "wc_match_summary.csv")
RATINGS = os.path.join(DATA, "intl_ratings.csv")
OUT_CSV = os.path.join(DATA, "wc_contract_fair_values.csv")
OUT_JSON = os.path.join(DATA, "wc_contract_distributions.json")

# Elo -> goals constants. GOALS_PER_RATING / BASE_TOTAL are PRIORS — calibrate()
# overwrites the live values (_BETA, _BASE_TOTAL) by fitting them to the sharp
# market on the 71 group games, which tames the favorite bias of the raw 1/120.
GOALS_PER_RATING = 1.0 / 120.0
BASE_TOTAL = 2.6
SUP_CAP = 3.0
ET_FRAC = 1.0 / 3.0          # extra time is 30 min = 1/3 of 90
LAM_FLOOR = 0.1

# Knockouts compound per-match edge over 5 rounds; KO_SHRINK<1 regresses the
# (calibrated) rating gap toward a coin flip in KO games only, to further temper
# top-seed overconfidence. 1.0 = rely purely on the market calibration.
KO_SHRINK = 1.0

# Live (possibly calibrated) goal-model params; set by calibrate().
_BETA = GOALS_PER_RATING
_BASE_TOTAL = BASE_TOTAL

# ---------------------------------------------------------------------------
# Official 2026 World Cup groups (odds-data spellings = canonical team names)
# ---------------------------------------------------------------------------
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia & Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
GROUP_LETTERS = list(GROUPS.keys())

# Odds-data spelling -> intl_ratings.csv (martj42 results) spelling, when different.
NAME_ALIASES = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}

# ---------------------------------------------------------------------------
# R32 slot map (official). Each entry: (sideA, sideB) where a side is one of
#   ("W", "<letter>")  group winner
#   ("RU", "<letter>") group runner-up
#   ("3", <match_no>)  best-third assigned to that match's third-slot
# ---------------------------------------------------------------------------
R32 = {
    73: (("RU", "A"), ("RU", "B")),
    74: (("W", "E"),  ("3", 74)),
    75: (("W", "F"),  ("RU", "C")),
    76: (("W", "C"),  ("RU", "F")),
    77: (("W", "I"),  ("3", 77)),
    78: (("RU", "E"), ("RU", "I")),
    79: (("W", "A"),  ("3", 79)),
    80: (("W", "L"),  ("3", 80)),
    81: (("W", "D"),  ("3", 81)),
    82: (("W", "G"),  ("3", 82)),
    83: (("RU", "K"), ("RU", "L")),
    84: (("W", "H"),  ("RU", "J")),
    85: (("W", "B"),  ("3", 85)),
    86: (("W", "J"),  ("RU", "H")),
    87: (("W", "K"),  ("3", 87)),
    88: (("RU", "D"), ("RU", "G")),
}

# Each third-slot accepts a best-third from one of these 5 groups (official lists).
THIRD_SLOTS = {
    74: ["A", "B", "C", "D", "F"],
    77: ["C", "D", "F", "G", "H"],
    79: ["C", "E", "F", "H", "I"],
    80: ["E", "H", "I", "J", "K"],
    81: ["B", "E", "F", "I", "J"],
    82: ["A", "E", "H", "I", "J"],
    85: ["E", "F", "G", "I", "J"],
    87: ["D", "E", "I", "J", "L"],
}

# R16 / QF / SF connections (winner of match X). Final = W101 vs W102; bronze = L101 vs L102.
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}

# Stage payouts for Set 2 (advancement).
PAY_LOSE_R32 = 2
PAY_LOSE_R16 = 4
PAY_LOSE_QF = 8
PAY_BRONZE_LOSER = 16   # 4th place (lost bronze final)
PAY_BRONZE_WINNER = 24  # 3rd place (won bronze final)
PAY_RUNNER_UP = 32
PAY_CHAMPION = 64


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------

def build_team_index():
    """Canonical team list (48) and team -> global index."""
    teams = [t for g in GROUPS.values() for t in g]
    assert len(teams) == 48 and len(set(teams)) == 48, "expected 48 unique teams"
    return teams, {t: i for i, t in enumerate(teams)}


def load_elo(teams):
    """Return Elo rating array aligned to global team index; assert all resolve."""
    rdf = pd.read_csv(RATINGS)
    rmap = dict(zip(rdf["team"], rdf["rating"]))
    lower = {k.lower(): v for k, v in rmap.items()}
    out = np.empty(len(teams))
    missing = []
    for i, t in enumerate(teams):
        if t in rmap:
            out[i] = rmap[t]
        elif t in NAME_ALIASES and NAME_ALIASES[t] in rmap:
            out[i] = rmap[NAME_ALIASES[t]]
        elif t.lower() in lower:
            out[i] = lower[t.lower()]
        else:
            missing.append(t)
    if missing:
        raise SystemExit(f"No Elo rating for: {missing} — add to NAME_ALIASES.")
    return out


def elo_lambdas(rh, ra, shrink=1.0):
    """Elo rating pair (scalars or arrays) -> (lam_home, lam_away), neutral venue.

    Uses the calibrated slope/total (_BETA, _BASE_TOTAL). `shrink` (<1) further
    regresses the rating gap toward even — used for knockout games.
    """
    sup = np.clip((rh - ra) * _BETA * shrink, -SUP_CAP, SUP_CAP)
    lh = np.maximum(LAM_FLOOR, (_BASE_TOTAL + sup) / 2.0)
    la = np.maximum(LAM_FLOOR, (_BASE_TOTAL - sup) / 2.0)
    return lh, la


def calibrate(idx, elo):
    """Fit goals-per-Elo slope + base total to the sharp market on group games.

    Regresses market supremacy (lam_h - lam_a) on neutral Elo diff with a home
    intercept (sup_mkt = beta*elo_diff + c); we keep `beta` for neutral KO play
    and drop the home tilt `c`. _BASE_TOTAL = mean market total. Falls back to the
    1/120 prior if the market file is unavailable.
    """
    global _BETA, _BASE_TOTAL
    if not os.path.exists(SUMMARY):
        print(f"[sim] no market file — using prior beta={_BETA:.5f}, total={_BASE_TOTAL}")
        return
    sm = pd.read_csv(SUMMARY)
    diffs, sups, totals = [], [], []
    for _, r in sm.iterrows():
        m = str(r["match"])
        if " vs " not in m:
            continue
        h, a = m.split(" vs ", 1)
        if h not in idx or a not in idx or pd.isna(r.get("lam_h")) or pd.isna(r.get("lam_a")):
            continue
        diffs.append(elo[idx[h]] - elo[idx[a]])
        sups.append(float(r["lam_h"]) - float(r["lam_a"]))
        totals.append(float(r["lam_h"]) + float(r["lam_a"]))
    if len(diffs) < 10:
        print(f"[sim] too few market games to calibrate — keeping prior")
        return
    X = np.column_stack([np.array(diffs), np.ones(len(diffs))])
    beta, c = np.linalg.lstsq(X, np.array(sups), rcond=None)[0]
    _BETA = float(beta)
    _BASE_TOTAL = float(np.mean(totals))
    print(f"[sim] calibrated to market: beta={_BETA:.5f} goals/Elo "
          f"(prior {GOALS_PER_RATING:.5f}), home tilt c={c:+.3f}, "
          f"base_total={_BASE_TOTAL:.3f} (prior {BASE_TOTAL})")


def team_to_group():
    return {t: g for g, ts in GROUPS.items() for t in ts}


def load_group_fixtures(idx, elo):
    """
    Build the 6 group matches per group as (home_idx, away_idx, lam_h, lam_a).
    Uses sharp market lambdas from wc_match_summary.csv where present; Elo fallback
    for any missing fixture (so all C(4,2)=6 pairings per group are covered).
    """
    t2g = team_to_group()
    market = {}  # frozenset({home,away}) -> (home, away, lam_h, lam_a)
    if os.path.exists(SUMMARY):
        sm = pd.read_csv(SUMMARY)
        for _, r in sm.iterrows():
            m = str(r["match"])
            if " vs " not in m:
                continue
            h, a = m.split(" vs ", 1)
            if h not in idx or a not in idx:
                continue
            lh, la = r.get("lam_h"), r.get("lam_a")
            if pd.notna(lh) and pd.notna(la):
                market[frozenset((h, a))] = (h, a, float(lh), float(la))

    fixtures = []   # (group, home_idx, away_idx, lam_h, lam_a)
    n_market = n_elo = 0
    for g, ts in GROUPS.items():
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = ts[i], ts[j]
                key = frozenset((a, b))
                if key in market:
                    h, aw, lh, la = market[key]
                    fixtures.append((g, idx[h], idx[aw], lh, la))
                    n_market += 1
                else:
                    lh, la = elo_lambdas(elo[idx[a]], elo[idx[b]])
                    fixtures.append((g, idx[a], idx[b], float(lh), float(la)))
                    n_elo += 1
    assert all(t2g[ts[0]] == g for g, ts in GROUPS.items())
    print(f"[sim] group fixtures: {n_market} sharp (market lambda) + {n_elo} Elo-fallback")
    return fixtures


# ---------------------------------------------------------------------------
# best-third slot assignment (bipartite perfect matching, cached per combo)
# ---------------------------------------------------------------------------

_slot_ids = list(THIRD_SLOTS.keys())
_assign_cache: dict = {}


def assign_thirds(qual_groups):
    """
    qual_groups: tuple of 8 group letters whose third-placed team qualified.
    Returns dict slot_id -> group_letter (a perfect matching respecting the
    official allowed-group lists = FIFA Annex C). Cached per combination.
    """
    key = frozenset(qual_groups)
    if key in _assign_cache:
        return _assign_cache[key]

    # Kuhn's algorithm: match slots -> groups, edge if group in slot's allowed list.
    qual = list(qual_groups)
    adj = {s: [g for g in THIRD_SLOTS[s] if g in qual] for s in _slot_ids}
    match_g = {}  # group -> slot

    def try_kuhn(s, seen):
        for g in adj[s]:
            if g in seen:
                continue
            seen.add(g)
            if g not in match_g or try_kuhn(match_g[g], seen):
                match_g[g] = s
                return True
        return False

    for s in _slot_ids:
        try_kuhn(s, set())

    result = {s: None for s in _slot_ids}
    for g, s in match_g.items():
        result[s] = g
    # Fallback for the rare combo with no perfect matching: fill leftover slots
    # with leftover groups arbitrarily (keeps the sim running; logged once).
    unfilled = [s for s in _slot_ids if result[s] is None]
    leftover = [g for g in qual if g not in match_g]
    for s, g in zip(unfilled, leftover):
        result[s] = g
    if unfilled:
        _assign_cache.setdefault("_imperfect", 0)
        _assign_cache["_imperfect"] += 1
    _assign_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# knockout match simulation (vectorized over sims)
# ---------------------------------------------------------------------------

def play_ko(home_idx, away_idx, elo, rng, goals):
    """
    Simulate one knockout match for all sims. Returns (winner_idx, loser_idx).
    Adds regulation + extra-time goals to `goals` (shootout goals excluded).
    """
    lh, la = elo_lambdas(elo[home_idx], elo[away_idx], shrink=KO_SHRINK)
    gh = rng.poisson(lh)
    ga = rng.poisson(la)

    tie = gh == ga
    # Extra time only where tied after 90.
    gh_et = np.where(tie, rng.poisson(lh * ET_FRAC), 0)
    ga_et = np.where(tie, rng.poisson(la * ET_FRAC), 0)
    gh_tot = gh + gh_et
    ga_tot = ga + ga_et

    still_tie = gh_tot == ga_tot
    # Shootout: home advances with prob lh/(lh+la). Goals NOT counted.
    p_home_so = lh / (lh + la)
    so_home = rng.random(len(home_idx)) < p_home_so
    home_wins = (gh_tot > ga_tot) | (still_tie & so_home)

    winner = np.where(home_wins, home_idx, away_idx)
    loser = np.where(home_wins, away_idx, home_idx)

    n = len(home_idx)
    sims = np.arange(n)
    np.add.at(goals, (sims, home_idx), gh_tot)
    np.add.at(goals, (sims, away_idx), ga_tot)
    return winner, loser


# ---------------------------------------------------------------------------
# main simulation
# ---------------------------------------------------------------------------

def simulate(n_sims=40000, seed=20260611):
    teams, idx = build_team_index()
    n_teams = len(teams)
    elo = load_elo(teams)
    calibrate(idx, elo)                       # fit goal model to the sharp market
    fixtures = load_group_fixtures(idx, elo)  # uses calibrated Elo for the fallback game
    rng = np.random.default_rng(seed)

    sims = np.arange(n_sims)
    goals = np.zeros((n_sims, n_teams))          # accumulated goals per team
    payout = np.zeros((n_sims, n_teams), dtype=np.int16)  # Set-2 stage payout
    is_champ = np.zeros((n_sims, n_teams), dtype=bool)

    # ---- group stage ----
    # accumulators per global team idx
    pts = np.zeros((n_sims, n_teams))
    gf = np.zeros((n_sims, n_teams))
    ga_ = np.zeros((n_sims, n_teams))
    for g, h, a, lh, la in fixtures:
        gh = rng.poisson(lh, n_sims)
        gaw = rng.poisson(la, n_sims)
        np.add.at(goals, (sims, np.full(n_sims, h)), gh)
        np.add.at(goals, (sims, np.full(n_sims, a)), gaw)
        h_win = gh > gaw
        a_win = gaw > gh
        draw = gh == gaw
        pts[:, h] += 3 * h_win + draw
        pts[:, a] += 3 * a_win + draw
        gf[:, h] += gh; ga_[:, h] += gaw
        gf[:, a] += gaw; ga_[:, a] += gh

    gd = gf - ga_

    # rank within each group: composite key pts >> gd >> gf >> random tiebreak
    rand_tb = rng.random((n_sims, n_teams))
    rank_key = pts * 1e9 + (gd + 200) * 1e4 + gf * 1e1 + rand_tb

    winners = {}   # letter -> (n_sims,) global team idx
    runners = {}
    thirds = {}    # letter -> (n_sims,) global team idx
    third_key = {}  # letter -> (n_sims,) ranking key of that group's third
    for g, ts in GROUPS.items():
        gidx = np.array([idx[t] for t in ts])           # 4 global indices
        sub = rank_key[:, gidx]                          # (n_sims, 4)
        order = np.argsort(-sub, axis=1)                 # best first
        winners[g] = gidx[order[:, 0]]
        runners[g] = gidx[order[:, 1]]
        thirds[g] = gidx[order[:, 2]]
        # third-place ranking key: pts, gd, gf (NO random — FIFA uses fair play/ranking,
        # we approximate with a tiny deterministic jitter folded into rank_key already)
        third_team = thirds[g]
        third_key[g] = (pts[sims, third_team] * 1e9
                        + (gd[sims, third_team] + 200) * 1e4
                        + gf[sims, third_team] * 1e1
                        + rand_tb[sims, third_team])

    # ---- pick 8 best thirds, assign to slots ----
    tk = np.stack([third_key[g] for g in GROUP_LETTERS], axis=1)  # (n_sims, 12)
    # top 8 group-columns per sim
    top8_cols = np.argsort(-tk, axis=1)[:, :8]                    # (n_sims, 8)

    # third-slot team per sim: start as -1, fill via per-combo matching
    slot_team = {s: np.full(n_sims, -1, dtype=np.int64) for s in _slot_ids}
    # group qualifying combos repeat heavily -> group sims by combo for speed
    combo_keys = [tuple(sorted(GROUP_LETTERS[c] for c in row)) for row in top8_cols]
    combo_arr = np.array(["".join(k) for k in combo_keys])
    for combo in np.unique(combo_arr):
        mask = combo_arr == combo
        qual = tuple(combo)  # letters in order
        assign = assign_thirds(qual)
        for s, g in assign.items():
            if g is None:
                continue
            slot_team[s][mask] = thirds[g][mask]

    # ---- knockouts ----
    win_of = {}   # match_no -> (n_sims,) winner team idx
    lose_of = {}

    def side_team(side):
        kind, ref = side
        if kind == "W":
            return winners[ref]
        if kind == "RU":
            return runners[ref]
        if kind == "3":
            return slot_team[ref]
        raise ValueError(side)

    # R32
    for m, (sa, sb) in R32.items():
        h = side_team(sa); a = side_team(sb)
        w, l = play_ko(h, a, elo, rng, goals)
        win_of[m] = w; lose_of[m] = l
        payout[sims, l] = PAY_LOSE_R32   # losers settle 2

    # R16
    for m, (ma, mb) in R16.items():
        w, l = play_ko(win_of[ma], win_of[mb], elo, rng, goals)
        win_of[m] = w; lose_of[m] = l
        payout[sims, l] = PAY_LOSE_R16

    # QF
    for m, (ma, mb) in QF.items():
        w, l = play_ko(win_of[ma], win_of[mb], elo, rng, goals)
        win_of[m] = w; lose_of[m] = l
        payout[sims, l] = PAY_LOSE_QF

    # SF
    for m, (ma, mb) in SF.items():
        w, l = play_ko(win_of[ma], win_of[mb], elo, rng, goals)
        win_of[m] = w; lose_of[m] = l

    # Bronze final: losers of SF 101 vs 102. Winner=3rd(24), loser=4th(16).
    bw, bl = play_ko(lose_of[101], lose_of[102], elo, rng, goals)
    payout[sims, bw] = PAY_BRONZE_WINNER
    payout[sims, bl] = PAY_BRONZE_LOSER

    # Final: winners of SF 101 vs 102. Winner=champion(64), loser=runner-up(32).
    cw, cl = play_ko(win_of[101], win_of[102], elo, rng, goals)
    payout[sims, cw] = PAY_CHAMPION
    payout[sims, cl] = PAY_RUNNER_UP
    is_champ[sims, cw] = True

    return dict(teams=teams, idx=idx, elo=elo, n_sims=n_sims,
                goals=goals, payout=payout, is_champ=is_champ)


# ---------------------------------------------------------------------------
# aggregation -> fair values + risk stats
# ---------------------------------------------------------------------------

STAGE_BUCKETS = [
    ("group_exit", 0), ("lose_R32", 2), ("lose_R16", 4), ("lose_QF", 8),
    ("fourth", 16), ("third", 24), ("runner_up", 32), ("champion", 64),
]


def aggregate(res):
    teams = res["teams"]
    goals = res["goals"]
    payout = res["payout"]
    is_champ = res["is_champ"]
    n = res["n_sims"]

    rows = []
    dist = {}
    for i, t in enumerate(teams):
        gi = goals[:, i]
        pi = payout[:, i]
        # Set 1 — winner
        p_champ = float(is_champ[:, i].mean())
        set1_fair = p_champ * 100.0
        # Set 2 — advancement
        set2_fair = float(pi.mean())
        # Set 3 — total goals
        set3_fair = float(gi.mean())

        stage_probs = {name: float((pi == val).mean()) for name, val in STAGE_BUCKETS}
        # P(reach each round) for intuition / trading the advancement curve
        p_reach_ko = float((pi >= 2).mean())     # made R32 (advanced from group)
        p_reach_r16 = float((pi >= 4).mean())
        p_reach_qf = float((pi >= 8).mean())
        p_reach_sf = float((pi >= 16).mean())
        p_reach_final = float(((pi == 32) | (pi == 64)).mean())

        rows.append(dict(
            team=t, elo=round(float(res["elo"][i]), 1),
            set1_winner_fair=round(set1_fair, 3),
            set2_advance_fair=round(set2_fair, 3),
            set3_goals_fair=round(set3_fair, 3),
            p_champion=round(p_champ, 4),
            p_advance_group=round(p_reach_ko, 4),
            p_reach_r16=round(p_reach_r16, 4),
            p_reach_qf=round(p_reach_qf, 4),
            p_reach_sf=round(p_reach_sf, 4),
            p_reach_final=round(p_reach_final, 4),
            set2_sd=round(float(pi.std()), 3),
            goals_sd=round(float(gi.std()), 3),
            goals_p10=int(np.percentile(gi, 10)),
            goals_p50=int(np.percentile(gi, 50)),
            goals_p90=int(np.percentile(gi, 90)),
        ))
        dist[t] = dict(
            stage_probs=stage_probs,
            goals_hist={int(k): int(v) for k, v in
                        zip(*np.unique(gi.astype(int), return_counts=True))},
        )

    df = pd.DataFrame(rows).sort_values("set1_winner_fair", ascending=False).reset_index(drop=True)
    return df, dist


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    print(f"[sim] running {n_sims:,} tournament simulations ...")
    res = simulate(n_sims)
    df, dist = aggregate(res)

    df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(dist, f)

    if _assign_cache.get("_imperfect"):
        print(f"[sim] WARN: {_assign_cache['_imperfect']} third-combos used fallback assignment")

    # sanity: champion probs sum to 1, payout EV averages, goal totals plausible
    print(f"[sim] sum P(champion) = {df['p_champion'].sum():.3f} (should be ~1.000)")
    print(f"[sim] mean set2 over all teams = {df['set2_advance_fair'].mean():.3f}")
    print(f"\n=== Top 15 by Set 1 (winner) fair value ===")
    cols = ["team", "elo", "set1_winner_fair", "set2_advance_fair",
            "set3_goals_fair", "p_advance_group", "p_reach_final"]
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(df[cols].head(15).to_string(index=False))
    print(f"\n[sim] wrote {OUT_CSV}")
    print(f"[sim] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
