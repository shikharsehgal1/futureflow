"""
Exports the internal Part 6 fitting dataframe for inspection.
For each league-season, produces:
  - One row per game with y_target = GoalDiff (real result), weighted by (1 - market_weight)
  - One row per game with y_target = mu_mkt (implied goal differential), weighted by market_weight
  - One dummy row per team (Bayesian prior, target=0)
  - One dummy row for home advantage (prior, target=0.3)

Fixes applied:
  1. real_actual rows with weight=0 are dropped (at mw=1.0 they're dead weight)
  2. Prior team rows correctly show HomeTeam=team, AwayTeam=anchor (matches design matrix)
  3. Prior weight ratio column added so you can see prior influence vs game influence

Blending logic:
  market_weight=1.0 → only implied rows carry weight (actual rows dropped)
  market_weight=0.5 → both rows contribute equally
  market_weight=0.0 → only actual rows carry weight (implied rows have weight=0)

Usage:
    python export_part6_df.py                    # all league-seasons
    python export_part6_df.py --div E0 --season 2023-24  # one league-season
"""

import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/big5_with_probs.csv", encoding="utf-8", low_memory=False)
df["Date"]     = pd.to_datetime(df["Date"])
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]

HALF_LIFE     = 90
MARKET_WEIGHT = 0.95
HA_PRIOR      = 0.3
COVID_SEASONS = ["2019-20", "2020-21"]


# ── BUILD FITTING DATAFRAME ───────────────────────────────────────────────────
def build_fitting_df(subset, half_life=HALF_LIFE, market_weight=MARKET_WEIGHT,
                     ha_prior=HA_PRIOR):
    """
    Returns the exact dataframe that fit_ratings() uses internally.

    Row types:
      real_actual  — actual goal differential (y_target = GoalDiff)
                     dropped if weight=0 (i.e. market_weight=1.0)
      real_implied — market-implied goal differential (y_target = mu_mkt)
                     dropped if weight=0 (i.e. market_weight=0.0)
      prior_team   — ghost-team prior per team (target=ha_prior=0.3)
                     fake home game vs __LEAGUE_AVG__ (ghost, rating=0)
                     [team=+1, __LEAGUE_AVG__=0, HFA=+1] = 0.3
                     → pulls team toward league avg AND bundles HFA prior
      prior_ha     — pure HFA prior (__LEAGUE_AVG__ vs __LEAGUE_AVG__)
                     [__LEAGUE_AVG__=+1, __LEAGUE_AVG__=-1, HFA=+1] = 0.3
                     → team indicators cancel → pure HFA signal

    Blending via weights, not y_target:
      actual weight  = decay_weight * (1 - market_weight)
      implied weight = decay_weight * market_weight
    """
    subset = subset.copy().reset_index(drop=True)
    subset["GoalDiff"] = subset["HomeGoals"] - subset["AwayGoals"]

    ref_date = subset["Date"].max()
    if half_life is None:
        subset["decay_weight"] = 1.0
    else:
        subset["decay_weight"] = 0.5 ** (
            (ref_date - subset["Date"]).dt.days / half_life)

    # Adaptive prior weight
    n_games        = len(subset)
    n_teams        = len(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    games_per_team = max(n_games * 2 / n_teams, 0.1)
    adaptive_prior = 3.0 / games_per_team

    teams  = sorted(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    anchor = "__LEAGUE_AVG__"   # ghost anchor — never plays, rating fixed at 0
    rows   = []

    for _, r in subset.iterrows():
        base = {
            "Date":      r["Date"],
            "Div":       r["Div"],
            "season":    r["season"],
            "HomeTeam":  r["HomeTeam"],
            "AwayTeam":  r["AwayTeam"],
            "HomeGoals": r["HomeGoals"],
            "AwayGoals": r["AwayGoals"],
            "GoalDiff":  r["GoalDiff"],
            "mu_mkt":    r["mu_mkt"],
        }

        # ── Row type 1: real actual (GoalDiff) ───────────────────────────────
        # FIX 1: drop if weight=0 (market_weight=1.0) — dead rows
        actual_weight = r["decay_weight"] * (1 - market_weight)
        if actual_weight > 0:
            rows.append({
                **base,
                "row_type":      "real_actual",
                "y_target":      r["GoalDiff"],
                "weight":        actual_weight,
                "prior_ratio":   np.nan,
                "note":          f"actual goal diff (mw={market_weight})",
            })

        # ── Row type 2: real implied (mu_mkt) ─────────────────────────────────
        # FIX 1: drop if weight=0 (market_weight=0.0) — dead rows
        implied_weight = r["decay_weight"] * market_weight
        if implied_weight > 0:
            rows.append({
                **base,
                "row_type":      "real_implied",
                "y_target":      r["mu_mkt"],
                "weight":        implied_weight,
                "prior_ratio":   np.nan,
                "note":          "market-implied goal diff (from Part 5)",
            })

    # Total game weight (for prior ratio calculation)
    total_game_weight = sum(
        r["weight"] for r in rows
        if r["row_type"] in ("real_actual", "real_implied")
    )

    # ── Row type 3: team shrinkage priors (decoupled) ────────────────────────
    # One row per team — pulls team rating toward 0 (league average).
    # HFA column = 0, so this ONLY shrinks the team rating, not HFA.
    # Target = 0.0 (league average), weight = team_prior_w.
    team_prior_w = adaptive_prior          # 3.0 / games_per_team
    hfa_prior_w  = adaptive_prior * 5/3   # 5.0 / games_per_team — stronger HFA anchor

    for team in teams:
        prior_ratio = team_prior_w / total_game_weight if total_game_weight > 0 else np.nan
        rows.append({
            "row_type":      "prior_team",
            "Date":           pd.NaT,
            "Div":            subset["Div"].iloc[0],
            "season":         subset["season"].iloc[0],
            "HomeTeam":       team,
            "AwayTeam":       anchor,   # __LEAGUE_AVG__ ghost — rating=0
            "HomeGoals":      np.nan,
            "AwayGoals":      np.nan,
            "GoalDiff":       np.nan,
            "mu_mkt":         np.nan,
            "y_target":       0.0,      # shrink team toward 0 (league avg)
            "weight":         team_prior_w,
            "prior_ratio":    prior_ratio,
            "note":           f"prior: {team}=0 (league avg shrinkage)  "
                              f"[design: {team}=+1, HFA=0, target=0.0]",
        })

    # ── Row type 4: HFA anchor prior (decoupled) ──────────────────────────────
    # Pure HFA signal — no team indicators involved.
    # Target = ha_prior (0.3), weight = hfa_prior_w (stronger than team prior).
    # This anchors HFA independently of how strongly teams are shrunk.
    prior_ratio = hfa_prior_w / total_game_weight if total_game_weight > 0 else np.nan
    rows.append({
        "row_type":      "prior_ha",
        "Date":           pd.NaT,
        "Div":            subset["Div"].iloc[0],
        "season":         subset["season"].iloc[0],
        "HomeTeam":       "__LEAGUE_AVG__",
        "AwayTeam":       "__LEAGUE_AVG__",
        "HomeGoals":      np.nan,
        "AwayGoals":      np.nan,
        "GoalDiff":       np.nan,
        "mu_mkt":         np.nan,
        "y_target":       ha_prior,
        "weight":         hfa_prior_w,
        "prior_ratio":    prior_ratio,
        "note":           f"prior: HFA={ha_prior} goals (decoupled anchor)  "
                          f"[design: HFA=+1 only, target={ha_prior}, w={hfa_prior_w:.4f}]",
    })

    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(divs=None, seasons=None):
    all_frames = []

    groups = df.groupby(["Div", "season"])
    for (div, season), subset in groups:
        if divs    and div    not in divs:    continue
        if seasons and season not in seasons: continue

        fitting_df = build_fitting_df(subset)
        all_frames.append(fitting_df)

    if not all_frames:
        print("No data matched filters.")
        return

    out = pd.concat(all_frames, ignore_index=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Total rows       : {len(out):,}")
    print(f"League-seasons   : {out.groupby(['Div', 'season']).ngroups}")
    print(f"\nRow type breakdown:")
    print(out["row_type"].value_counts().to_string())

    # ── Sanity check ──────────────────────────────────────────────────────────
    actual  = out[out["row_type"] == "real_actual"]["y_target"]
    implied = out[out["row_type"] == "real_implied"]["y_target"]
    if len(actual) > 0 and len(implied) > 0:
        n = min(len(actual), len(implied))
        match = (actual.values[:n] == implied.values[:n]).mean()
        print(f"\nSanity check — actual y_target == implied y_target: "
              f"{match:.1%} (should be ~0%)")

    print(f"\nWeight sanity check:")
    wt = out.groupby("row_type")["weight"].agg(["sum","mean","max"])
    print(wt.to_string())

    print(f"\nPrior influence (prior_ratio = prior_weight / total_game_weight):")
    pr = out[out["row_type"].isin(["prior_team","prior_ha"])].groupby("row_type")["prior_ratio"].mean()
    print(pr.to_string())
    print("  (ratio > 1 = prior dominates, < 1 = data dominates)")

    # ── Sample ────────────────────────────────────────────────────────────────
    sample_div = divs[0]    if divs    else "E0"
    sample_s   = seasons[0] if seasons else "2023-24"
    sample = out[(out["Div"] == sample_div) & (out["season"] == sample_s)]

    print(f"\nSample — {sample_div} {sample_s} ({len(sample)} rows total):")

    if len(sample[sample["row_type"] == "real_actual"]) > 0:
        print(f"\n  Real actual rows (first 3):")
        print(sample[sample["row_type"] == "real_actual"]
              [["HomeTeam","AwayTeam","GoalDiff","mu_mkt","y_target","weight","note"]]
              .head(3).to_string(index=False))

    print(f"\n  Real implied rows (first 3):")
    print(sample[sample["row_type"] == "real_implied"]
          [["HomeTeam","AwayTeam","GoalDiff","mu_mkt","y_target","weight","note"]]
          .head(3).to_string(index=False))

    print(f"\n  Prior team rows (first 3):")
    print(sample[sample["row_type"] == "prior_team"]
          [["HomeTeam","AwayTeam","y_target","weight","prior_ratio","note"]]
          .head(3).to_string(index=False))

    print(f"\n  Prior home advantage row:")
    print(sample[sample["row_type"] == "prior_ha"]
          [["HomeTeam","AwayTeam","y_target","weight","prior_ratio","note"]]
          .to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    out.to_csv("data/part6_fitting_df.csv", index=False)
    print(f"\nSaved → data/part6_fitting_df.csv")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--div",    nargs="*", default=None)
    parser.add_argument("--season", nargs="*", default=None)
    args = parser.parse_args()
    main(divs=args.div, seasons=args.season)
