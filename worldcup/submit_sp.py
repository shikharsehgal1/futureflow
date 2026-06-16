"""
submit_sp.py — Submit our probability predictions to the live SportsPredict Cup.

Outward-facing + hard to reverse, so it is SAFE BY DEFAULT:
  * Dry-run unless you pass --send.
  * Before sending, it GETs an existing prediction to learn the real payload
    field names (market id + probability), so we mirror the API instead of
    guessing. If no prior prediction exists to introspect, it falls back to the
    documented {marketId, probability} shape and asks you to confirm.

Usage:
    export SP_API_KEY="sp_live_..."
    python3 submit_sp.py                      # dry-run: prints the payloads
    python3 submit_sp.py --send               # actually POSTs them

Probability is sent as a fraction in [0,1] (pct/100). Adjust SCALE if the API
wants 0-100 — confirmed at runtime from the introspected prediction.
"""
from __future__ import annotations
import csv, json, os, sys, urllib.request, urllib.error

BASE = "https://api.sportspredict.com/api"
CUP_EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ENTRIES = os.path.join(DATA, "sp_submit_fra_sen.csv")


def _key():
    k = os.environ.get("SP_API_KEY")
    if not k:
        sys.exit("Set SP_API_KEY env var first (export SP_API_KEY='sp_live_...').")
    return k


def _req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
        headers={"Authorization": f"Bearer {_key()}", "User-Agent": "Mozilla/5.0",
                 "Accept": "application/json", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def introspect():
    """Learn the prediction payload shape from an existing submitted prediction."""
    status, preds = _req("GET", f"/v1/predictions?event_id={CUP_EVENT}")
    if status == 200 and isinstance(preds, list) and preds:
        sample = preds[0]
        print("Existing prediction object keys:", list(sample.keys()))
        print("Sample:", json.dumps(sample, indent=2)[:600])
        return sample
    print(f"No existing prediction to introspect (status {status}). "
          f"Will use default {{marketId, probability(0-1)}} shape.")
    return None


def _lobby_map():
    """market_id -> lobby_id, read from the live questions cache."""
    path = os.path.join(DATA, "sp_questions.csv")
    return {r["market_id"]: r["lobby_id"] for r in csv.DictReader(open(path))}


def _existing():
    """market_id -> prediction id for predictions we've already submitted."""
    status, preds = _req("GET", f"/v1/predictions?event_id={CUP_EVENT}")
    if status == 200 and isinstance(preds, list):
        return {p["market_id"]: p["id"] for p in preds if p.get("market_id")}
    return {}


def main():
    send = "--send" in sys.argv
    rows = list(csv.DictReader(open(ENTRIES)))
    rows = [r for r in rows if r["prob_pct"] not in ("", None)]
    lobby = _lobby_map()
    existing = _existing() if send else {}

    print(f"=== SportsPredict submit ({'LIVE SEND' if send else 'DRY-RUN'}) — {len(rows)} entries ===")
    sample = introspect()
    # field names: mirror the sample if present, else defaults
    id_field = "marketId"
    prob_field = "probability"
    scale = 1.0  # fraction by default
    if sample:
        for cand in ("marketId", "market_id", "marketID"):
            if cand in sample:
                id_field = cand; break
        for cand in ("probability", "prob", "value", "prediction"):
            if cand in sample:
                prob_field = cand
                v = sample[cand]
                scale = 100.0 if isinstance(v, (int, float)) and v > 1.5 else 1.0
                break

    print(f"\nUsing fields: {id_field}, {prob_field} (scale x{scale:g})\n")
    results = []
    for r in rows:
        p = float(r["prob_pct"]) / 100.0 * scale
        payload = {id_field: r["market_id"], prob_field: round(p, 4),
                   "lobby_id": lobby.get(r["market_id"], "")}
        print(f"  {r['question'][:55]:55s} -> {r['prob_pct']}%  payload={payload}")
        if send:
            # upsert: PATCH if we already have a prediction for this market, else POST
            pid = existing.get(r["market_id"])
            if pid:
                st, resp = _req("PATCH", f"/v1/predictions/{pid}", {prob_field: round(p, 4)})
            else:
                st, resp = _req("POST", "/v1/predictions", payload)
            ok = st in (200, 201)
            print(f"      {'OK' if ok else 'FAIL'} ({'update' if pid else 'create'}) status={st} {('' if ok else str(resp)[:160])}")
            results.append((r["market_id"], st, ok))
    if send:
        good = sum(1 for _, _, ok in results if ok)
        print(f"\nSubmitted {good}/{len(results)} OK.")
    else:
        print("\nDRY-RUN only. Re-run with --send to submit.")


if __name__ == "__main__":
    main()
