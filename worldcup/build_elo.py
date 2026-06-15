"""
build_elo.py — International-football Elo fundamentals model (SECOND signal).

The primary World Cup model (predict_wc.py) anchors to de-vigged Pinnacle odds.
This module is an independent, model-based cross-check: it converts the two teams'
international Elo ratings into a full probability picture for every match, with NO
home advantage (World Cup matches are at neutral venues).

Pipeline:
  1. Fetch & cache eloratings.net ratings to data/elo_ratings.json (re-fetch only if
     the cache is missing or older than 1 day).
  2. For each match in data/wc_match_summary.csv, map both team names to Elo, then:
       - expected home score  E = 1 / (1 + 10^(-dElo/400))     (standard Elo formula,
         neutral venue -> no home bonus)
       - supremacy in goals    sup ~ dElo / 150
       - total goals           tot = 2.6 (league-neutral baseline)
       - lam_h, lam_a          fit the EXISTING double-Poisson (fit_poisson) to the
         Elo-implied win/draw/loss + total, so the scoreline matrix is consistent
         with the rest of the repo and every goal-derived question reuses devig.py.
  3. Emit data/wc_elo_questions.csv with the SAME columns as wc_questions.csv,
     source="elo:model": moneyline (each team + draw), total goals over 2.5, BTTS.

Plain numpy/scipy/pandas + requests; no new heavy deps. Reuses devig.py wholesale.
"""
from __future__ import annotations

import io
import json
import os
import time
import numpy as np
import pandas as pd
import requests

from devig import (fit_poisson, score_matrix, model_over_prob, clip_prob,
                   _hda_from_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SUMMARY = os.path.join(DATA, "wc_match_summary.csv")
ELO_CACHE = os.path.join(DATA, "elo_ratings.json")
OUT = os.path.join(DATA, "wc_elo_questions.csv")

WORLD_URL = "https://eloratings.net/World.tsv"
TEAMS_URL = "https://eloratings.net/en.teams.tsv"

CACHE_MAX_AGE = 24 * 3600  # re-fetch only if cache is missing or older than 1 day

# Elo -> goals model constants
GOALS_PER_ELO = 1.0 / 150.0   # supremacy (goals) ~ dElo / 150
BASE_TOTAL = 2.6              # league-neutral expected total goals
SUP_CAP = 3.0                 # cap supremacy so blowout Elo gaps stay sane

# Manual aliases: Odds-API spelling -> eloratings.net team name (en.teams.tsv).
# Most names match an en.teams.tsv column directly (incl. its "&" aliases); these
# cover the handful that differ. Resolution still falls back to the teams file.
NAME_ALIASES = {
    "South Korea": "South Korea",
    "Czech Republic": "Czechia",
    "USA": "United States",
    "Turkey": "Turkey",
    "Ivory Coast": "Ivory Coast",
    "Curaçao": "Curaçao",
    "Curacao": "Curaçao",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "DR Congo",
    "Cape Verde": "Cape Verde",
    "South Africa": "South Africa",
    "Scotland": "Scotland",
}


# ---------------------------------------------------------------------------
# fetch + cache
# ---------------------------------------------------------------------------

def _fetch_tsv(url: str) -> list[list[str]]:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    # eloratings serves UTF-8 but omits a charset header, so requests guesses
    # latin-1 and mangles accents (Curaçao -> CuraÃ§ao). Decode the raw bytes.
    text = r.content.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append(line.split("\t"))
    return rows


def _build_ratings() -> dict:
    """Fetch World.tsv + en.teams.tsv and build {team_name: elo}.

    World.tsv layout (verified): col0 rank, col1 rank, col2 ISO-ish code, col3 Elo, ...
    en.teams.tsv layout: col0 code, col1 canonical name, col2.. = aliases.
    We key the ratings dict by EVERY known name/alias so the match file resolves
    directly, then add our explicit NAME_ALIASES on top.
    """
    world = _fetch_tsv(WORLD_URL)
    teams = _fetch_tsv(TEAMS_URL)

    code_to_elo = {}
    for row in world:
        if len(row) < 4:
            continue
        code = row[2].strip()
        try:
            elo = float(row[3])
        except (ValueError, IndexError):
            continue
        code_to_elo[code] = elo

    code_to_names = {}
    name_to_code = {}
    for row in teams:
        if len(row) < 2:
            continue
        code = row[0].strip()
        names = [n.strip() for n in row[1:] if n.strip()]
        if not names:
            continue
        code_to_names[code] = names
        for n in names:
            name_to_code.setdefault(n, code)

    ratings = {}
    for name, code in name_to_code.items():
        if code in code_to_elo:
            ratings[name] = code_to_elo[code]

    return {"fetched_at": time.time(), "ratings": ratings,
            "n_world": len(code_to_elo), "n_named": len(ratings)}


def load_ratings(force: bool = False) -> dict:
    """Return cached ratings, re-fetching only if missing/stale or force=True."""
    if not force and os.path.exists(ELO_CACHE):
        age = time.time() - os.path.getmtime(ELO_CACHE)
        if age < CACHE_MAX_AGE:
            with open(ELO_CACHE) as f:
                data = json.load(f)
            print(f"[elo] using cache ({age/3600:.1f}h old, "
                  f"{len(data.get('ratings', {}))} teams)")
            return data
    print("[elo] fetching fresh ratings from eloratings.net ...")
    data = _build_ratings()
    os.makedirs(DATA, exist_ok=True)
    with open(ELO_CACHE, "w") as f:
        json.dump(data, f, indent=0)
    print(f"[elo] cached {data['n_named']} named teams "
          f"({data['n_world']} ranked codes) -> {ELO_CACHE}")
    return data


# ---------------------------------------------------------------------------
# Elo -> probabilities
# ---------------------------------------------------------------------------

def resolve_elo(team: str, ratings: dict):
    """Map an Odds-API team name to an Elo, via alias then direct lookup. None if unmatched."""
    if team in ratings:
        return ratings[team]
    alias = NAME_ALIASES.get(team)
    if alias and alias in ratings:
        return ratings[alias]
    # last resort: case-insensitive direct match
    low = team.lower()
    for name, elo in ratings.items():
        if name.lower() == low:
            return elo
    return None


def elo_match_probs(elo_h: float, elo_a: float):
    """Convert an Elo pair into a consistent scoreline model (neutral venue).

    Returns dict with pH/pD/pA, lam_h/lam_a, p_over_25, p_btts, and the intermediate
    expected score / supremacy. Reuses fit_poisson + score_matrix from devig.py so the
    derived markets are identical in spirit to the market-anchored Poisson elsewhere.
    """
    d_elo = elo_h - elo_a                              # no home advantage added
    exp_home = 1.0 / (1.0 + 10.0 ** (-d_elo / 400.0))  # expected score in [0,1]

    # supremacy (goal margin) from Elo diff, capped; total fixed at a neutral baseline
    sup = float(np.clip(d_elo * GOALS_PER_ELO, -SUP_CAP, SUP_CAP))
    tot = BASE_TOTAL
    lam_h0 = max(0.1, (tot + sup) / 2.0)
    lam_a0 = max(0.1, (tot - sup) / 2.0)

    # Seed pH/pD/pA from the supremacy guess so fit_poisson has a target. We let the
    # Poisson itself define the draw, then renormalise to expected_home as a tie-break.
    M0 = score_matrix(lam_h0, lam_a0)
    pH0, pD0, pA0 = _hda_from_matrix(M0)

    # Nudge the win/loss split toward the Elo expected score while keeping the draw.
    # expected_home = pH + 0.5*pD  =>  target pH given pD0 and exp_home.
    win_mass = 1.0 - pD0
    target_pH = float(np.clip(exp_home - 0.5 * pD0, 0.01, win_mass - 0.01))
    target_pA = win_mass - target_pH
    pD = pD0

    lam_h, lam_a = fit_poisson(target_pH, pD, target_pA, total_line=tot, p_over=None)
    M = score_matrix(lam_h, lam_a)
    pH, pD, pA = _hda_from_matrix(M)

    p_over_25 = model_over_prob(M, 2.5)
    p_btts = float(M[1:, 1:].sum())

    return dict(elo_h=elo_h, elo_a=elo_a, d_elo=d_elo, exp_home=exp_home,
                sup=sup, lam_h=lam_h, lam_a=lam_a,
                pH=float(pH), pD=float(pD), pA=float(pA),
                p_over_25=float(p_over_25), p_btts=p_btts, M=M)


# ---------------------------------------------------------------------------
# question rows (mirror predict_wc.py's _q contract)
# ---------------------------------------------------------------------------

def _q(rows, match, mkt, question, prob, commence, match_id, note=""):
    if prob is None or not np.isfinite(prob):
        return
    rows.append(dict(match=match, market=mkt, question=question,
                     prob=round(clip_prob(prob), 4),
                     pct=round(100 * clip_prob(prob), 1),
                     source="elo:model", note=note,
                     commence=commence, match_id=match_id))


def build():
    data = load_ratings()
    ratings = data["ratings"]

    summ = pd.read_csv(SUMMARY)

    all_rows = []
    table = []
    skipped = []
    for _, m in summ.iterrows():
        match = m["match"]
        commence = m["commence"]
        match_id = m["match_id"]
        if " vs " not in match:
            skipped.append((match, "unparseable match name"))
            continue
        home, away = match.split(" vs ", 1)

        elo_h = resolve_elo(home, ratings)
        elo_a = resolve_elo(away, ratings)
        if elo_h is None or elo_a is None:
            miss = home if elo_h is None else away
            if elo_h is None and elo_a is None:
                miss = f"{home} & {away}"
            skipped.append((match, f"no Elo for {miss}"))
            print(f"[elo] skip '{match}': no Elo for {miss}")
            continue

        r = elo_match_probs(elo_h, elo_a)

        note = f"Elo {elo_h:.0f} vs {elo_a:.0f} (d={r['d_elo']:+.0f}, neutral)"
        _q(all_rows, match, "Moneyline", f"Will {home} win the match?",
           r["pH"], commence, match_id, note)
        _q(all_rows, match, "Moneyline", f"Will {away} win the match?",
           r["pA"], commence, match_id, note)
        _q(all_rows, match, "Moneyline", "Will the match end in a draw?",
           r["pD"], commence, match_id, note)
        _q(all_rows, match, "Total Goals", "Will total goals be over 2.5?",
           r["p_over_25"], commence, match_id, note)
        _q(all_rows, match, "BTTS", "Will both teams score?",
           r["p_btts"], commence, match_id, note)

        table.append(dict(match=match, elo_h=elo_h, elo_a=elo_a,
                          d_elo=r["d_elo"], pH=r["pH"], pD=r["pD"], pA=r["pA"],
                          p_over_25=r["p_over_25"], p_btts=r["p_btts"],
                          market_pH=float(m["pH"])))

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["commence", "match", "market"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    return df, pd.DataFrame(table), skipped


def main():
    df, table, skipped = build()
    print(f"\nWrote {len(df)} Elo questions across {len(table)} matches -> {OUT}")
    if skipped:
        print(f"\nSkipped {len(skipped)} matches:")
        for match, why in skipped:
            print(f"  - {match}: {why}")

    # Summary table: Elo model vs market, with divergence flag
    print("\n=== Elo model vs market (home win prob) ===")
    print(f"{'match':<40}{'elo_pH':>8}{'mkt_pH':>8}{'diff':>8}")
    for _, t in table.iterrows():
        diff = t["pH"] - t["market_pH"]
        flag = "  <-- >10pp" if abs(diff) > 0.10 else ""
        print(f"{t['match'][:39]:<40}{t['pH']*100:>7.1f}%{t['market_pH']*100:>7.1f}%"
              f"{diff*100:>+7.1f}%{flag}")


if __name__ == "__main__":
    main()
