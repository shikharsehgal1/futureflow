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

HALF_LIFE        = 90
MARKET_HALF_LIFE = 30
MARKET_WEIGHT    = 0.95
HA_PRIOR         = 0.3
COVID_SEASONS    = ["2019-20", "2020-21"]

# ── PRESEASON PRIOR CONFIG ────────────────────────────────────────────────────
TIER1_DIVS         = {"E0","D1","SP1","I1","F1"}
CARRY_WEIGHT_TIER1 = 0.50   # top divisions (regression avg = 0.52)
CARRY_WEIGHT_TIER2 = 0.10   # second divisions (regression avg = 0.10)
PROMOTED_PRIOR     = -0.453 # empirical: newly promoted avg goal diff
RELEGATED_PRIOR    = +0.236 # empirical: newly relegated avg goal diff
PROMOTED_ATK       = -0.18  # empirical attack rating for promoted teams
PROMOTED_DFC       = +0.16  # empirical defence rating for promoted teams
RELEGATED_ATK      = +0.04
RELEGATED_DFC      = -0.14
PRESEASON_WEIGHT   =  5.0   # prior row weight — fades with adaptive_prior


# ── BUILD FITTING DATAFRAME ───────────────────────────────────────────────────
def build_fitting_df(subset, half_life=HALF_LIFE, market_weight=MARKET_WEIGHT,
                     ha_prior=HA_PRIOR, preseason_priors=None):
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

    # ── Row type 5: preseason priors ─────────────────────────────────────────
    # One row per team with an informed starting point from last season.
    # Returning teams: carry_weight * end-of-last-season rating
    # Promoted teams: PROMOTED_PRIOR (-0.57)
    # Relegated teams: RELEGATED_PRIOR (+0.29)
    # Weight = PRESEASON_WEIGHT * adaptive_prior — strong at week 1, fades by week 10
    if preseason_priors:
        ps_weight = PRESEASON_WEIGHT * adaptive_prior
        ps_ratio  = ps_weight / total_game_weight if total_game_weight > 0 else np.nan
        for team, prior_val in preseason_priors.items():
            if team not in teams: continue
            rows.append({
                "row_type":    "prior_preseason",
                "Date":         pd.NaT,
                "Div":          subset["Div"].iloc[0],
                "season":       subset["season"].iloc[0],
                "HomeTeam":     team,
                "AwayTeam":     "__LEAGUE_AVG__",
                "HomeGoals":    np.nan,
                "AwayGoals":    np.nan,
                "GoalDiff":     np.nan,
                "mu_mkt":       np.nan,
                "y_target":     float(prior_val),
                "weight":       ps_weight,
                "prior_ratio":  ps_ratio,
                "note":         f"preseason prior: {team}={prior_val:+.3f}  "
                                f"[carry={CARRY_WEIGHT_TIER1 if subset['Div'].iloc[0] in TIER1_DIVS else CARRY_WEIGHT_TIER2:.2f}, "
                                f"w={ps_weight:.4f}]",
            })

    return pd.DataFrame(rows)


# ── PRESEASON PRIORS ──────────────────────────────────────────────────────────
def build_preseason_priors_export(div, season):
    """Build preseason priors for export inspection."""
    all_seasons = sorted(df[df["Div"]==div]["season"].unique())
    if season not in all_seasons or all_seasons.index(season) == 0:
        return {}
    prev_season = all_seasons[all_seasons.index(season) - 1]

    prev_df = df[(df["Div"]==div) & (df["season"]==prev_season)]
    curr_df = df[(df["Div"]==div) & (df["season"]==season)]
    if len(prev_df) == 0 or len(curr_df) == 0:
        return {}

    teams_prev = set(prev_df["HomeTeam"]) | set(prev_df["AwayTeam"])
    teams_curr = set(curr_df["HomeTeam"]) | set(curr_df["AwayTeam"])
    returning  = teams_curr & teams_prev
    new_teams  = teams_curr - teams_prev

    # End-of-last-season ratings (final 50% of games)
    tail = prev_df.sort_values("Date").tail(max(len(prev_df)//2, 10))
    n      = len(tail)
    n_t    = len(set(tail["HomeTeam"]) | set(tail["AwayTeam"]))
    gpt    = max(n * 2 / n_t, 0.1)
    ap     = 3.0 / gpt
    tpw    = ap; hpw = ap * 5/3
    tms    = sorted(set(tail["HomeTeam"]) | set(tail["AwayTeam"]))
    col    = tms + ["Home_Adv"]
    def mk(h, a):
        r = {c: 0 for c in col}; r["Home_Adv"] = 1
        if h in tms: r[h] =  1
        if a in tms: r[a] = -1
        return [r[c] for c in col]
    X  = np.array([mk(r.HomeTeam, r.AwayTeam) for _, r in tail.iterrows()])
    ya = (tail["HomeGoals"] - tail["AwayGoals"]).values.astype(float)
    yi = tail["mu_mkt"].values.astype(float)
    Xg = np.vstack([X,X]); yg = np.concatenate([ya,yi])
    dw = (0.5 ** ((tail["Date"].max() - tail["Date"]).dt.days / HALF_LIFE)).values
    dm = (0.5 ** ((tail["Date"].max() - tail["Date"]).dt.days / MARKET_HALF_LIFE)).values
    wg = np.concatenate([dw*(1-MARKET_WEIGHT), dm*MARKET_WEIGHT])
    Xp = []; yp2 = []; wp = []
    for t in tms:
        r = {c:0 for c in col}; r[t]=1
        Xp.append([r[c] for c in col]); yp2.append(0.0); wp.append(tpw)
    r = {c:0 for c in col}; r["Home_Adv"]=1
    Xp.append([r[c] for c in col]); yp2.append(HA_PRIOR); wp.append(hpw)
    Xa = np.vstack([Xg, np.array(Xp)])
    ya2 = np.concatenate([yg, np.array(yp2)])
    wa  = np.concatenate([wg, np.array(wp)])
    sq  = np.sqrt(np.clip(wa, 0, None))
    beta, _, _, _ = np.linalg.lstsq(Xa*sq[:,None], ya2*sq, rcond=None)
    if not np.all(np.isfinite(beta)): return {}
    prev_ratings = dict(zip(col[:-1], beta[:-1]))
    mean_r = np.mean(list(prev_ratings.values()))
    prev_ratings = {t: v - mean_r for t, v in prev_ratings.items()}

    carry  = CARRY_WEIGHT_TIER1 if div in TIER1_DIVS else CARRY_WEIGHT_TIER2
    priors = {t: carry * prev_ratings.get(t, 0.0) for t in returning}

    div_to_lower = {"E0":"E1","D1":"D2","SP1":"SP2","I1":"I2","F1":"F2"}
    div_to_upper = {"E1":"E0","D2":"D1","SP2":"SP1","I2":"I1","F2":"F1"}
    for team in new_teams:
        is_promoted = is_relegated = False
        ud = div_to_upper.get(div)
        if ud:
            up = df[(df["Div"]==ud) & (df["season"]==prev_season)]
            if len(up)>0 and team in (set(up["HomeTeam"])|set(up["AwayTeam"])):
                is_relegated = True
        ld = div_to_lower.get(div)
        if ld and not is_relegated:
            lo = df[(df["Div"]==ld) & (df["season"]==prev_season)]
            if len(lo)>0 and team in (set(lo["HomeTeam"])|set(lo["AwayTeam"])):
                is_promoted = True
        priors[team] = PROMOTED_PRIOR if is_promoted else \
                       RELEGATED_PRIOR if is_relegated else 0.0
    return priors


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(divs=None, seasons=None):
    all_frames = []

    groups = df.groupby(["Div", "season"])
    for (div, season), subset in groups:
        if divs    and div    not in divs:    continue
        if seasons and season not in seasons: continue

        fitting_df = build_fitting_df(
            subset,
            preseason_priors=build_preseason_priors_export(div, season)
        )
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
    pr = out[out["row_type"].isin(["prior_team","prior_ha","prior_preseason"])].groupby("row_type")["prior_ratio"].mean()
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

    if len(sample[sample["row_type"] == "prior_preseason"]) > 0:
        print(f"\n  Preseason prior rows (all):")
        ps = sample[sample["row_type"] == "prior_preseason"] \
            [["HomeTeam","y_target","weight","note"]] \
            .sort_values("y_target", ascending=False)
        print(ps.to_string(index=False))

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
