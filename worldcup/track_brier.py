"""
track_brier.py — Settle the model's predictions against real results and score them.

Pulls finished-match scores from The Odds API /scores endpoint and computes the
Brier score of every goal-derivable question in wc_questions.csv (moneyline, double
chance, totals, odd/even, BTTS, team totals, handicaps, correct score). Markets that
need in-match stats (corners, cards, offsides, fouls, shots, 1st-half, players) can't
be settled from the score alone and are reported as "needs stats feed".

We can't see the platform's field-average Brier, so we benchmark against the naive
0.5 baseline (Brier 0.25) and report a skill score + calibration buckets — enough to
prove the model has edge over time and see which markets it's strongest on.

Usage:
  python3 track_brier.py            # fetch scores, settle, print report
  python3 track_brier.py --no-fetch # re-score from cached scores only (0 credits)
"""
from __future__ import annotations
import argparse, json, os, re, sys, datetime as dt
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# Prefer the frozen prediction snapshot (survives matches leaving the odds feed);
# fall back to the live questions table if no snapshot exists yet.
_LOG = os.path.join(DATA, "predictions_log.csv")
QUESTIONS = _LOG if os.path.exists(_LOG) else os.path.join(DATA, "wc_questions.csv")
SCORES_CACHE = os.path.join(DATA, "wc_scores.json")
BRIER_LOG = os.path.join(DATA, "brier_log.csv")

API_KEY = os.environ.get("ODDS_API_KEY", "0ce377e63ef8333b1794c5f26e08b384")
SPORT = "soccer_fifa_world_cup"


def fetch_scores(days_from=3):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/scores"
    r = requests.get(url, params={"apiKey": API_KEY, "daysFrom": days_from}, timeout=30)
    rem = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        print(f"  scores fetch failed {r.status_code}: {r.text[:140]}", file=sys.stderr)
        return None
    data = r.json()
    json.dump(data, open(SCORES_CACHE, "w"))
    print(f"  fetched scores · {rem} credits left", file=sys.stderr)
    return data


def finished_scores(data):
    """match_id -> (home, away, hg, ag)."""
    out = {}
    for e in data or []:
        if not e.get("completed"):
            continue
        sc = {s["name"]: int(s["score"]) for s in (e.get("scores") or []) if s.get("score") is not None}
        h, a = e["home_team"], e["away_team"]
        if h in sc and a in sc:
            out[e["id"]] = (h, a, sc[h], sc[a])
    return out


def _num(q):
    m = re.search(r"(\d+\.?\d*)", q)
    return float(m.group(1)) if m else None


def resolve(question, market, home, away, hg, ag):
    """Return 1/0 for YES/NO, or None if not settleable from the final score."""
    q = question
    tot = hg + ag
    if market == "Moneyline":
        if "end in a draw" in q: return int(hg == ag)
        if home in q: return int(hg > ag)
        if away in q: return int(ag > hg)
    if market == "Double Chance":
        if "1X" in q: return int(hg >= ag)
        if "X2" in q: return int(ag >= hg)
        if "12" in q: return int(hg != ag)
    if market == "Total Goals":
        n = _num(q)
        if n is not None and "over" in q.lower(): return int(tot > n)
    if market == "Total Goals O/E":
        if "even" in q.lower(): return int(tot % 2 == 0)
        if "odd" in q.lower(): return int(tot % 2 == 1)
    if market == "BTTS":
        yes = int(hg >= 1 and ag >= 1)
        return (1 - yes) if "NOT" in q else yes
    if market == "Team Total":
        n = _num(q)
        if n is not None:
            tg = hg if home in q else (ag if away in q else None)
            if tg is not None: return int(tg > n)
    if market == "First To Score" and "no goals" in q.lower():
        return int(tot == 0)
    if market == "Handicap":
        d = hg - ag
        if "win by 3+" in q: return int((d >= 3) if home in q.split(" win")[0] else (-d >= 3))
        if "win by 2+" in q: return int((d >= 2) if home in q.split(" win")[0] else (-d >= 2))
        if "avoid losing by 2+" in q:
            return int(d >= -1) if home in q else int(-d >= -1)
    if market == "Correct Score":
        m = re.search(r"(\d+)-(\d+)", q)
        if m: return int(hg == int(m.group(1)) and ag == int(m.group(2)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="use cached scores (0 credits)")
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()

    if args.no_fetch and os.path.exists(SCORES_CACHE):
        data = json.load(open(SCORES_CACHE))
    else:
        data = fetch_scores(args.days)
        if data is None and os.path.exists(SCORES_CACHE):
            data = json.load(open(SCORES_CACHE))

    fin = finished_scores(data)
    if not fin:
        print("No finished matches yet — nothing to settle.")
        return
    df = pd.read_csv(QUESTIONS)

    rows = []
    for mid, (h, a, hg, ag) in fin.items():
        sub = df[df.match_id == mid]
        if sub.empty:                       # fall back to name match (id namespaces can differ)
            sub = df[df.match == f"{h} vs {a}"]
        for _, r in sub.iterrows():
            o = resolve(r["question"], r["market"], h, a, hg, ag)
            if o is None:
                continue
            p = float(r["prob"])
            rows.append(dict(match=r["match"], market=r["market"], question=r["question"],
                             prob=p, outcome=o, brier=round((p - o) ** 2, 4),
                             source=str(r["source"]).split(":")[0], tier=r["tier"],
                             score=f"{hg}-{ag}"))
    if not rows:
        print(f"{len(fin)} finished match(es) but no settleable goal-market questions found.")
        return
    out = pd.DataFrame(rows)
    out.to_csv(BRIER_LOG, index=False)

    print(f"\n=== SETTLED {len(fin)} match(es), {len(out)} goal-market questions ===")
    for mid, (h, a, hg, ag) in fin.items():
        print(f"  {h} {hg}-{ag} {a}")
    base = 0.25  # Brier of always predicting 0.5
    print(f"\nMODEL Brier (lower=better):  {out.brier.mean():.4f}   "
          f"vs 0.5-baseline {base:.4f}   skill={(base-out.brier.mean())/base*100:+.0f}%")
    print("\nBy market:")
    g = out.groupby("market").agg(n=("brier", "size"), brier=("brier", "mean")).sort_values("brier")
    for mkt, row in g.iterrows():
        print(f"  {mkt:16s} n={int(row['n']):>3}  Brier={row['brier']:.4f}")
    print("\nBy source:")
    for src, row in out.groupby("source").agg(n=("brier", "size"), brier=("brier", "mean")).iterrows():
        print(f"  {src:12s} n={int(row['n']):>3}  Brier={row['brier']:.4f}")
    # calibration buckets
    print("\nCalibration (predicted% vs actual hit-rate):")
    out["bucket"] = (out["prob"] * 10).astype(int).clip(0, 9)
    for b, row in out.groupby("bucket").agg(n=("outcome", "size"), hit=("outcome", "mean")).iterrows():
        print(f"  {b*10:>3}-{b*10+10:<3}%  n={int(row['n']):>3}  actual={100*row['hit']:.0f}%")
    print(f"\nLog -> {BRIER_LOG}")
    print("(Corners/cards/offsides/fouls/shots/1st-half/players need a stats feed to settle.)")


if __name__ == "__main__":
    main()
