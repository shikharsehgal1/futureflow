"""
fetch_lbenz.py — Re-pull Luke Benz's published bivariate-Poisson ratings.

His ratings.csv updates when he refits (after each group matchweek + R32/R16:
2026-06-17, 06-23, 06-27, 07-04, 07-08). Run this in the refresh cycle so our
fundamentals layer tracks his latest alpha/delta. Free, no auth (raw GitHub).

NOTE: his global constants (mu, home_field, neutral_field) live in an uncommitted
posterior.rds and are NOT in the repo, so we keep our validated hardcoded triple in
lbenz_model.py and accept it may drift slightly after his refit dates — the per-team
alpha/delta (which dominate match-to-match differences) are refreshed here.
"""
import os
import urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = "https://raw.githubusercontent.com/lbenz730/world_cup_2026/main/predictions/ratings.csv"


def main():
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        if b"alpha" in data[:200] and b"," in data[:200]:
            open(os.path.join(DATA, "lbenz_ratings.csv"), "wb").write(data)
            n = data.count(b"\n") - 1
            print(f"lbenz ratings refreshed: {n} teams")
        else:
            print("lbenz ratings: unexpected content, kept existing")
    except Exception as e:
        print(f"lbenz ratings fetch failed ({e}); kept existing")


if __name__ == "__main__":
    main()
