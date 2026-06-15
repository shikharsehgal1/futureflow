"""
sp_brier.py — Live scoreboard: read our actual Brier per prediction from the platform.

The SportsPredict /v1/predictions feed carries a `brier_score` per prediction that
populates as each market settles. This reads it directly — our REAL score, straight
from the source (no local settlement needed; survives matches leaving any feed).

Usage:  SP_API_KEY=... python3 sp_brier.py
"""
from __future__ import annotations
import json, os, sys, urllib.request
from collections import defaultdict

B = "https://api.sportspredict.com/api"
EV = "aa5572ec-5930-4d99-b06b-f8966333d172"


def main():
    k = os.environ.get("SP_API_KEY") or sys.exit("set SP_API_KEY")
    req = urllib.request.Request(f"{B}/v1/predictions?event_id={EV}",
        headers={"Authorization": f"Bearer {k}", "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    preds = json.loads(urllib.request.urlopen(req, timeout=30).read())
    settled = [p for p in preds if p.get("brier_score") is not None]
    print(f"predictions on record: {len(preds)} | settled (scored): {len(settled)}")
    if not settled:
        print("No markets have settled yet — Brier scores populate as matches finish.")
        return
    briers = [float(p["brier_score"]) for p in settled]
    mean = sum(briers) / len(briers)
    base = 0.25  # always-0.5 baseline
    print(f"\nMEAN Brier (settled): {mean:.4f}   vs 0.5-baseline {base:.4f}   "
          f"skill {100*(base-mean)/base:+.0f}%   (lower=better)")
    # by market keyword bucket
    def bucket(q):
        ql = q.lower()
        for kw in ["win the match", "halftime", "corner", "offside", "foul", "card",
                   "shot on target", "both teams", "second half", "total goals", "score"]:
            if kw in ql: return kw
        return "other"
    g = defaultdict(list)
    for p in settled:
        g[bucket(p.get("question", ""))].append(float(p["brier_score"]))
    print("\nBy market type (mean Brier, n):")
    for kw, v in sorted(g.items(), key=lambda x: sum(x[1]) / len(x[1])):
        print(f"  {kw:16s} {sum(v)/len(v):.4f}  (n={len(v)})")
    # best & worst calls
    sp = sorted(settled, key=lambda p: float(p["brier_score"]))
    print("\nBest calls:")
    for p in sp[:4]:
        print(f"  {p['brier_score']:.3f}  {p.get('probability')}%  {p.get('question','')[:55]}")
    print("Worst calls:")
    for p in sp[-4:]:
        print(f"  {p['brier_score']:.3f}  {p.get('probability')}%  {p.get('question','')[:55]}")


if __name__ == "__main__":
    main()
