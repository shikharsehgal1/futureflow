"""
sp_submit.py — Submit our model probabilities to the SportsPredict Probability Cup.

Reads data/sp_entries.csv (market_id, prob_pct, confidence) + data/sp_questions.csv
(market_id -> lobby_id), and POSTs {market_id, lobby_id, probability} for every
high/med-confidence entry that has a probability. Low-confidence rows are skipped.

SECURITY: key from env SP_API_KEY only. Submissions can be updated until each match
locks, so re-running overwrites with the latest model number.

Usage:
  python3 sp_submit.py --dry-run        # show what would be submitted, send nothing
  python3 sp_submit.py --max 1          # submit ONE (test the contract)
  python3 sp_submit.py                  # submit all high/med
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, urllib.request

B = "https://api.sportspredict.com/api"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def key():
    k = os.environ.get("SP_API_KEY")
    if not k: sys.exit("set SP_API_KEY")
    return k


def _send(url, method, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method=method,
        headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                 "Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    try:
        r = urllib.request.urlopen(req, timeout=20); return r.status, r.read(200).decode("utf-8", "replace")
    except urllib.error.HTTPError as e: return e.code, e.read(200).decode("utf-8", "replace")
    except Exception as e: return None, str(e)[:120]


def post(market_id, lobby_id, prob):
    return _send(B + "/v1/predictions", "POST",
                 {"market_id": market_id, "lobby_id": lobby_id, "probability": int(prob)})


def patch(pred_id, prob):
    return _send(f"{B}/v1/predictions/{pred_id}", "PATCH", {"probability": int(prob)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max", type=int, default=0, help="submit at most N (0=all)")
    ap.add_argument("--delay", type=float, default=1.3, help="seconds between submissions")
    args = ap.parse_args()

    lobby = {r["market_id"]: r["lobby_id"] for r in csv.DictReader(open(f"{DATA}/sp_questions.csv"))}
    # live market status -> only submit to OPEN markets (skip locked/in-play matches)
    req = urllib.request.Request(f"{B}/v1/markets?event_id=aa5572ec-5930-4d99-b06b-f8966333d172",
        headers={"Authorization": f"Bearer {key()}", "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    status = {m["id"]: m.get("status") for m in json.loads(urllib.request.urlopen(req, timeout=30).read())}
    entries = [r for r in csv.DictReader(open(f"{DATA}/sp_entries.csv"))
               if r["confidence"] in ("high", "med") and r["prob_pct"] not in ("", None)
               and status.get(r["market_id"]) == "open"]
    n_locked = sum(1 for r in csv.DictReader(open(f"{DATA}/sp_entries.csv"))
                   if status.get(r["market_id"]) and status.get(r["market_id"]) != "open")
    # existing predictions: market_id -> (prediction_id, current_probability)  [for PATCH-update]
    preq = urllib.request.Request(f"{B}/v1/predictions?event_id=aa5572ec-5930-4d99-b06b-f8966333d172",
        headers={"Authorization": f"Bearer {key()}", "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    existing = {p["market_id"]: (p["id"], p.get("probability"))
                for p in json.loads(urllib.request.urlopen(preq, timeout=30).read())}
    print(f"{len(entries)} entries on OPEN markets ({n_locked} locked-skip, {len(existing)} already predicted).")
    created = updated = unchanged = ok = 0
    for r in entries:
        if args.max and (created + updated) >= args.max: break
        mid, lob, p = r["market_id"], lobby.get(r["market_id"], ""), int(float(r["prob_pct"]))
        if not lob: continue
        if args.dry_run:
            print(f"  DRY {p:>3}%  {r['match']}: {r['question'][:55]}"); continue
        if mid in existing:
            pid, cur = existing[mid]
            if cur == p:
                unchanged += 1; continue          # no change -> skip (saves calls)
            send = lambda: patch(pid, p); kind = "upd"
        else:
            send = lambda: post(mid, lob, p); kind = "new"
        for attempt in range(5):
            st, body = send()
            if st != 429: break
            time.sleep(8 * (attempt + 1))
        good = st in (200, 201)
        ok += good
        if kind == "new" and good: created += 1
        elif kind == "upd" and good: updated += 1
        elif not good:
            print(f"  ERR {st} {p:>3}%  {r['match']}: {r['question'][:46]} | {body[:80]}")
        time.sleep(args.delay)
    if not args.dry_run:
        print(f"created {created}, updated {updated}, unchanged-skipped {unchanged}")
        req = urllib.request.Request(f"{B}/v1/predictions?event_id=aa5572ec-5930-4d99-b06b-f8966333d172",
            headers={"Authorization": f"Bearer {key()}", "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        try:
            print(f"predictions on record: {len(json.loads(urllib.request.urlopen(req, timeout=20).read()))}")
        except Exception as e:
            print("read-back failed:", e)


if __name__ == "__main__":
    main()
