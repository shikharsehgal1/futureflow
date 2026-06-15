"""
build_extra_stats.py — per-team match-stat rates for 18 teams NOT covered by the
StatsBomb WC18/22 pipeline.

FBref (Method 1) is fully blocked behind a Cloudflare "Just a moment..." challenge
that cloudscraper cannot pass (403 on every URL, incl. robots.txt). So we use
Method 2: the martj42 international-results dataset for REAL goals_for/goals_against
from each team's last ~20 matches; every other market is prior-filled (we have no
shots/corners/etc. source). Goals are still shrunk toward the prior with weight 3,
exactly like fetch_statsbomb.py.
"""
from __future__ import annotations
import os, csv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "team_stats_extra.csv")
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

PRIOR = dict(goals=1.35, shots=11.5, sot=4.0, corners=4.8, offsides=1.8,
             fouls=11.5, cards=2.0, pens=0.10)
PRIOR_W = 3.0
N_RECENT = 20

# output name -> dataset name (only differs where the dataset spells it otherwise)
TARGETS = {
    "Algeria": "Algeria", "Austria": "Austria",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde", "Curaçao": "Curaçao",
    "Czech Republic": "Czech Republic", "DR Congo": "DR Congo",
    "Haiti": "Haiti", "Iraq": "Iraq", "Ivory Coast": "Ivory Coast",
    "Jordan": "Jordan", "New Zealand": "New Zealand", "Norway": "Norway",
    "Paraguay": "Paraguay", "Scotland": "Scotland",
    "South Africa": "South Africa", "Turkey": "Turkey",
    "Uzbekistan": "Uzbekistan",
}

STATS = ["goals", "shots", "sot", "corners", "offsides", "fouls", "cards", "pens"]


def shrink(raw_sum, n, prior):
    return (raw_sum + prior * PRIOR_W) / (n + PRIOR_W) if (n + PRIOR_W) > 0 else prior


def main():
    import pandas as pd
    try:
        df = pd.read_csv(RESULTS_URL)
    except Exception:
        df = pd.read_csv("/tmp/intl_results.csv")
    df = df.sort_values("date")

    rows = []
    coverage = []
    for out_name, ds_name in TARGETS.items():
        n = 0
        gf_sum = ga_sum = 0.0
        try:
            # only matches actually played (scores present) — excludes the future
            # WC2026 fixtures the dataset already lists with NaN scores
            played = df[((df.home_team == ds_name) | (df.away_team == ds_name))
                        & df.home_score.notna() & df.away_score.notna()]
            m = played.tail(N_RECENT)
            for _, r in m.iterrows():
                if r.home_team == ds_name:
                    gf, ga = r.home_score, r.away_score
                else:
                    gf, ga = r.away_score, r.home_score
                if pd.isna(gf) or pd.isna(ga):
                    continue
                gf_sum += float(gf); ga_sum += float(ga); n += 1
        except Exception as e:
            print(f"  {out_name}: scrape error {e}")

        row = {"team": out_name, "n_matches": n}
        # goals are REAL (shrunk); everything else is the prior for both for/against
        row["goals_for"] = round(shrink(gf_sum, n, PRIOR["goals"]), 3)
        row["goals_against"] = round(shrink(ga_sum, n, PRIOR["goals"]), 3)
        for s in STATS:
            if s == "goals":
                continue
            row[s + "_for"] = round(PRIOR[s], 3)
            row[s + "_against"] = round(PRIOR[s], 3)
        rows.append(row)
        coverage.append((out_name, n))

    fields = ["team", "n_matches"] + [f"{s}_{d}" for s in STATS for d in ("for", "against")]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} teams -> {OUT}")
    for t, n in coverage:
        print(f"  {t}: {n} real matches (goals real, rest prior)")


if __name__ == "__main__":
    main()
