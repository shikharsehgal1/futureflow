"""
snapshot.py — Freeze predictions so they can be scored after the match finishes.

Problem: a match drops out of the odds feed the moment it kicks off, taking its
predictions out of wc_questions.csv — so the Brier tracker can never settle it.

Fix: maintain data/predictions_log.csv keyed by (match_id, question). Each refresh
upserts the CURRENT prediction for matches that have NOT kicked off yet — so the row
keeps tracking the latest line right up to kickoff, then FREEZES (we stop updating
once commence is in the past). The frozen "closing" prediction persists forever and is
what track_brier.py scores against the result.

Run as part of refresh.py (before the match starts), or standalone:
  python3 snapshot.py
"""
from __future__ import annotations
import csv
import os
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
QUESTIONS = os.path.join(DATA, "wc_questions.csv")
LOG = os.path.join(DATA, "predictions_log.csv")

FIELDS = ["match_id", "match", "market", "question", "prob", "pct", "source",
          "commence", "snapshot_utc", "referee", "ref_card_factor", "weather_adj", "home_adv"]


def _now():
    return dt.datetime.now(dt.timezone.utc)


def main():
    if not os.path.exists(QUESTIONS):
        print("no wc_questions.csv"); return
    cur = list(csv.DictReader(open(QUESTIONS)))
    log = {}
    if os.path.exists(LOG):
        for r in csv.DictReader(open(LOG)):
            log[(r["match_id"], r["question"])] = r

    now = _now()
    frozen = updated = added = 0
    for r in cur:
        key = (r["match_id"], r["question"])
        try:
            ko = dt.datetime.fromisoformat(r["commence"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            ko = now  # unknown kickoff -> treat as now (freeze immediately)
        started = now >= ko
        if key in log and started:
            frozen += 1
            continue  # match underway/finished -> keep the frozen closing prediction
        row = {k: r.get(k, "") for k in FIELDS}
        row["snapshot_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        if key in log:
            updated += 1
        else:
            added += 1
        log[key] = row

    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in log.values():
            w.writerow({k: row.get(k, "") for k in FIELDS})
    print(f"snapshot: {len(log)} logged questions "
          f"({added} new, {updated} refreshed pre-kickoff, {frozen} frozen post-kickoff) -> {LOG}")


if __name__ == "__main__":
    main()
