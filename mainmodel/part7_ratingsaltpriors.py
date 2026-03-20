"""
Part 7 — Script 1: Power Ratings
Loads latest data, fits ratings for current season per league,
outputs a clean table in goal differential space and implied win % vs .500.

Usage:
    python part7_ratings.py                    # all leagues
    python part7_ratings.py --div E0           # EPL only
    python part7_ratings.py --div E0 D1 SP1    # multiple leagues
"""

import argparse
import pandas as pd
import numpy as np
from scipy.special import factorial
from scipy.stats import norm
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
CURRENT_SEASON    = "2025-26"
HALF_LIFE         = 90    # days — long-run decay for actual goal diff
MARKET_HALF_LIFE  = 30    # days — steeper decay for mu_mkt (tuned via grid search)
MARKET_WEIGHT     = 0.95  # from Monte Carlo: high mw dominates
FORM_WEIGHT       = 0.15  # blend of short-term form into final rating (0 = off)
FORM_HALF_LIFE    = 21    # days — short window to capture recent momentum
MIN_GAMES         = 5     # minimum games per team before showing a rating
COVID_SEASONS     = ["2019-20","2020-21"]

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

DIV_NAMES = {
    "E0": "Premier League",   "E1": "Championship",
    "D1": "Bundesliga",       "D2": "2. Bundesliga",
    "SP1": "La Liga",         "SP2": "Segunda División",
    "I1": "Serie A",          "I2": "Serie B",
    "F1": "Ligue 1",          "F2": "Ligue 2",
}

# ── 2. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/big5_with_probs.csv", encoding="utf-8", low_memory=False)
df["Date"]     = pd.to_datetime(df["Date"])
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]

with open("data/mean_goals.pkl", "rb") as f:
    mg = pickle.load(f)
MEAN_HOME = mg["home"]
MEAN_AWAY = mg["away"]

def load_mean_goals():
    return MEAN_HOME, MEAN_AWAY


# ── 3b. PRESEASON PRIORS ──────────────────────────────────────────────────────
def build_preseason_priors(div, season, all_df):
    """
    Build {team: prior_rating} for season start using:
      - Returning teams:  carry_weight * end-of-last-season rating
      - Promoted teams:   PROMOTED_PRIOR  (-0.57, empirical Big 5, n=92)
      - Relegated teams:  RELEGATED_PRIOR (+0.29, empirical Big 5, n=92)
      - Unknown new teams: 0.0 (league average)
    """
    all_seasons = sorted(all_df[all_df["Div"]==div]["season"].unique())
    if season not in all_seasons or all_seasons.index(season) == 0:
        return {}
    prev_season = all_seasons[all_seasons.index(season) - 1]

    prev_df = all_df[(all_df["Div"]==div) & (all_df["season"]==prev_season)]
    curr_df = all_df[(all_df["Div"]==div) & (all_df["season"]==season)]
    if len(prev_df) == 0 or len(curr_df) == 0:
        return {}

    teams_prev = set(prev_df["HomeTeam"]) | set(prev_df["AwayTeam"])
    teams_curr = set(curr_df["HomeTeam"]) | set(curr_df["AwayTeam"])
    returning  = teams_curr & teams_prev
    new_teams  = teams_curr - teams_prev

    # End-of-last-season ratings: fit on final 50% of prev season games
    prev_sorted = prev_df.sort_values("Date")
    tail_prev   = prev_sorted.tail(max(len(prev_sorted)//2, 10))
    res = fit_ratings(tail_prev)
    if res is None:
        return {}
    prev_ratings = res["ratings"]
    mean_prev    = np.mean(list(prev_ratings.values()))
    prev_ratings = {t: r - mean_prev for t, r in prev_ratings.items()}

    carry  = CARRY_WEIGHT_TIER1 if div in TIER1_DIVS else CARRY_WEIGHT_TIER2
    priors = {}

    for team in returning:
        priors[team] = carry * prev_ratings.get(team, 0.0)

    # Detect promoted vs relegated for new teams
    div_to_lower = {"E0":"E1","D1":"D2","SP1":"SP2","I1":"I2","F1":"F2"}
    div_to_upper = {"E1":"E0","D2":"D1","SP2":"SP1","I2":"I1","F2":"F1"}
    lower_div = div_to_lower.get(div)
    upper_div = div_to_upper.get(div)

    for team in new_teams:
        is_promoted = is_relegated = False
        if upper_div:
            up = all_df[(all_df["Div"]==upper_div) & (all_df["season"]==prev_season)]
            if len(up) > 0 and team in (set(up["HomeTeam"])|set(up["AwayTeam"])):
                is_relegated = True
        if lower_div and not is_relegated:
            lo = all_df[(all_df["Div"]==lower_div) & (all_df["season"]==prev_season)]
            if len(lo) > 0 and team in (set(lo["HomeTeam"])|set(lo["AwayTeam"])):
                is_promoted = True

        if is_promoted:
            priors[team] = PROMOTED_PRIOR
        elif is_relegated:
            priors[team] = RELEGATED_PRIOR
        else:
            priors[team] = 0.0

    return priors


# ── 3. FIT RATINGS ────────────────────────────────────────────────────────────
def fit_ratings(subset, half_life=HALF_LIFE, market_weight=MARKET_WEIGHT,
                market_half_life=MARKET_HALF_LIFE, form_weight=FORM_WEIGHT,
                form_half_life=FORM_HALF_LIFE, ha_prior=0.3,
                preseason_priors=None):
    """
    Fit goal-differential power ratings via weighted OLS.

    Two upgrades over the base model:

    4. Recency-weighted market signal: mu_mkt rows use a steeper time decay
       (market_half_life=45d) than actual goal rows (half_life=90d). Recent
       market odds already reflect injuries and form — older odds don't.

    3. Form/momentum: a short-term rating (form_half_life=21d) is blended
       with the main rating via form_weight=0.15. Captures recent momentum.
    """
    subset = subset.copy().reset_index(drop=True)
    subset["GoalDiff"] = subset["HomeGoals"] - subset["AwayGoals"]

    n_games        = len(subset)
    n_teams        = len(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    games_per_team = max(n_games * 2 / n_teams, 0.1)
    adaptive_prior = 3.0 / games_per_team

    ref_date = subset["Date"].max()
    days_ago = (ref_date - subset["Date"]).dt.days

    # Long-run decay for actual goal diff rows
    decay_w = (0.5 ** (days_ago / half_life)).astype(float) if half_life \
              else np.ones(n_games)
    # Steeper decay for mu_mkt rows — recent odds more informative
    decay_w_mkt = (0.5 ** (days_ago / market_half_life)).astype(float) \
                  if market_half_life else np.ones(n_games)
    # Short-window decay for form signal
    decay_w_form = (0.5 ** (days_ago / form_half_life)).astype(float) \
                   if form_half_life else np.ones(n_games)

    teams      = sorted(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    anchor     = "__LEAGUE_AVG__"   # ghost anchor — never plays, rating fixed at 0
    free_teams = teams               # ALL real teams are free parameters
    col_order  = free_teams + ["Home_Adv"]

    def make_row(h, a):
        r = {t: 0 for t in col_order}; r["Home_Adv"] = 1
        if h in free_teams: r[h] =  1
        if a in free_teams: r[a] = -1
        return [r[c] for c in col_order]

    X = np.array([make_row(r.HomeTeam, r.AwayTeam) for _, r in subset.iterrows()])

    # ── Two rows per game: actual + implied ───────────────────────────────────
    y_actual  = subset["GoalDiff"].values.astype(float)
    y_implied = subset["mu_mkt"].values.astype(float)

    X_games = np.vstack([X, X])
    y_games = np.concatenate([y_actual, y_implied])
    w_games = np.concatenate([
        decay_w     * (1 - market_weight),   # actual — long decay
        decay_w_mkt * market_weight,          # implied — steeper decay (#4)
    ])

    # ── Prior rows — decoupled team shrinkage and HFA anchor ─────────────────
    # Two independent priors with separate weights and separate target values:
    #
    # Team prior:  [team=+1, HFA=0] = 0
    #   → rating_team = 0  → shrinks team toward league average
    #   → HFA NOT included — team shrinkage does not drag HFA estimate
    #   → weighted by team_prior_w
    #
    # HFA prior:   [HFA=+1] = ha_prior
    #   → HFA = ha_prior  → pure HFA anchor, independent of team ratings
    #   → weighted by hfa_prior_w (stronger — HFA is stable across seasons)
    team_prior_w = adaptive_prior          # 3.0 / games_per_team
    hfa_prior_w  = adaptive_prior * 5/3   # 5.0 / games_per_team — harder HFA anchor

    Xp, yp, wp = [], [], []

    # Team shrinkage rows — one per team, HFA column = 0
    for t in free_teams:
        r = {c: 0 for c in col_order}   # HFA stays 0
        r[t] = 1
        Xp.append([r[c] for c in col_order]); yp.append(0.0); wp.append(team_prior_w)

    # HFA anchor row — pure HFA signal, no team indicators
    r = {c: 0 for c in col_order}; r["Home_Adv"] = 1
    Xp.append([r[c] for c in col_order]); yp.append(ha_prior); wp.append(hfa_prior_w)

    # ── Preseason prior rows ──────────────────────────────────────────────────
    # One row per team with a known preseason prior (returning/promoted/relegated).
    # Target = prior_rating, weight = PRESEASON_WEIGHT * adaptive_prior so it
    # fades naturally as games accumulate — strong at week 1, near-zero by week 10.
    if preseason_priors:
        ps_weight = PRESEASON_WEIGHT * adaptive_prior
        for t, prior_val in preseason_priors.items():
            if t not in free_teams: continue
            r = {c: 0 for c in col_order}   # HFA stays 0
            r[t] = 1
            Xp.append([r[c] for c in col_order])
            yp.append(float(prior_val))
            wp.append(ps_weight)

    Xa = np.vstack([X_games, np.array(Xp)])
    ya = np.concatenate([y_games, np.array(yp)])
    wa = np.concatenate([w_games, np.array(wp)])
    sq = np.sqrt(np.clip(wa, 0, None))

    beta, _, _, _ = np.linalg.lstsq(Xa * sq[:,None], ya * sq, rcond=None)
    if not np.all(np.isfinite(beta)): return None

    coefs = dict(zip(col_order, beta))

    # COVID correction — re-estimate home adv from non-COVID games only
    is_covid = subset["season"].isin(COVID_SEASONS) if "season" in subset.columns \
               else pd.Series(False, index=subset.index)
    if is_covid.any() and not is_covid.all():
        Xnc = X[~is_covid.values]
        ync = y_actual[~is_covid.values]
        wnc = decay_w[~is_covid.values]
        coefs["Home_Adv"] = float(np.average(
            ync - Xnc[:,:-1] @ beta[:-1], weights=wnc))

    base_ratings = {t: coefs[t] for t in free_teams}

    # ── Form/momentum blend (#3) ──────────────────────────────────────────────
    if form_weight > 0 and n_games >= MIN_GAMES * 2:
        X_form = np.vstack([X, X])
        y_form = np.concatenate([y_actual, y_implied])
        w_form = np.concatenate([
            decay_w_form * (1 - market_weight),
            decay_w_form * market_weight,
        ])
        Xa_f = np.vstack([X_form, np.array(Xp)])
        ya_f = np.concatenate([y_form, np.array(yp)])
        wa_f = np.concatenate([w_form, np.array(wp)])
        sq_f = np.sqrt(np.clip(wa_f, 0, None))
        beta_f, _, _, _ = np.linalg.lstsq(Xa_f*sq_f[:,None], ya_f*sq_f, rcond=None)
        if np.all(np.isfinite(beta_f)):
            form_coefs = dict(zip(col_order, beta_f))
            final_ratings = {
                t: (1 - form_weight) * base_ratings[t] + form_weight * form_coefs[t]
                for t in free_teams
            }
        else:
            final_ratings = base_ratings
    else:
        final_ratings = base_ratings

    # ── Langville-Meyer attack/defence split ──────────────────────────────────
    # Uses final (form-blended) overall ratings as the anchor, then splits into
    # attack/defence using opponent-quality-adjusted xG (shots on target proxy).
    overall_arr = np.array([final_ratings.get(t, 0.0) for t in teams])
    n = len(teams)
    mean_home_g, mean_away_g = load_mean_goals()
    mean_goals = (mean_home_g + mean_away_g) / 2

    hg = subset["HomeGoals"].values.astype(int) if "HomeGoals" in subset.columns \
         else np.zeros(n_games, dtype=int)
    ag = subset["AwayGoals"].values.astype(int) if "AwayGoals" in subset.columns \
         else np.zeros(n_games, dtype=int)

    has_sot = ("HST" in subset.columns and "AST" in subset.columns and
               subset["HST"].notna().mean() > 0.5)
    if has_sot:
        valid = subset[subset["HST"].notna() & subset["AST"].notna() &
                       (subset["HST"] > 0) & (subset["AST"] > 0)]
        home_conv = valid["HomeGoals"].sum() / valid["HST"].sum() \
                    if len(valid) >= MIN_GAMES else 0.316
        away_conv = valid["AwayGoals"].sum() / valid["AST"].sum() \
                    if len(valid) >= MIN_GAMES else 0.306
        hst = subset["HST"].fillna(pd.Series(hg / max(home_conv, 0.1), index=subset.index)).values
        ast_ = subset["AST"].fillna(pd.Series(ag / max(away_conv, 0.1), index=subset.index)).values
        xg_h = np.maximum(hst * home_conv, 0.05)
        xg_a = np.maximum(ast_ * away_conv, 0.05)
    else:
        xg_h = np.maximum(hg.astype(float), 0.05)
        xg_a = np.maximum(ag.astype(float), 0.05)

    G = np.zeros((n, n))
    for k, (ht, at, xgh, xga, wk) in enumerate(zip(
            subset["HomeTeam"], subset["AwayTeam"], xg_h, xg_a, decay_w)):
        hi_k = teams.index(ht); ai_k = teams.index(at)
        G[ai_k, hi_k] += wk * xgh
        G[hi_k, ai_k] += wk * xga

    od_off = np.ones(n); od_def = np.ones(n)
    for _ in range(50):
        od_off_new = np.array([sum(G[i,j]/max(od_def[i],1e-6) for i in range(n))
                               for j in range(n)])
        od_def_new = np.array([sum(G[i,j]/max(od_off_new[j],1e-6) for j in range(n))
                               for i in range(n)])
        if (np.max(np.abs(od_off_new-od_off)) < 1e-6 and
                np.max(np.abs(od_def_new-od_def)) < 1e-6): break
        od_off, od_def = od_off_new, od_def_new

    atk_raw = np.log(np.maximum(od_off, 1e-6)); atk_raw -= atk_raw.mean()
    dfc_raw = np.log(np.maximum(od_def, 1e-6)); dfc_raw -= dfc_raw.mean()
    # Dampen split — xGH/xGA should vary meaningfully but not go extreme
    # 0.5 = half the raw LM signal; prevents outlier teams from blowing up lambdas
    SPLIT_DAMP = 0.15
    split   = (atk_raw - dfc_raw) * SPLIT_DAMP
    atk_fin = overall_arr/2 + split
    dfc_fin = -overall_arr/2 + split

    return {
        "ratings":      final_ratings,
        "base_ratings": base_ratings,
        "attack":       {teams[i]: atk_fin[i] for i in range(n)},
        "defence":      {teams[i]: dfc_fin[i] for i in range(n)},
        "home_adv":     coefs["Home_Adv"],
        "n_games":      n_games,
        "ref_date":     ref_date,
        "anchor":       anchor,
        "mean_home":    mean_home_g,
        "mean_away":    mean_away_g,
    }

# ── 4. CONVERT RATINGS TO WIN % VS .500 OPPONENT ─────────────────────────────
def poisson_probs(mu, max_goals=10):
    total = MEAN_HOME + MEAN_AWAY
    lH = np.clip((total + mu) / 2, 0.1, None)
    lA = np.clip((total - mu) / 2, 0.1, None)
    g  = np.arange(max_goals + 1)
    M  = np.outer(np.exp(-lH) * lH**g / factorial(g),
                  np.exp(-lA) * lA**g / factorial(g))
    return np.clip([np.tril(M,-1).sum(), M.trace(), np.triu(M,1).sum()],
                   1e-6, 1-1e-6)

def win_pct_vs_500(rating):
    """Win probability at neutral venue vs a league-average (.500) opponent."""
    pH, pD, pA = poisson_probs(rating)  # opponent rating=0, no home adv
    return pH

# ── 5. BUILD RATINGS TABLE ────────────────────────────────────────────────────
def build_ratings_table(div, season=CURRENT_SEASON):
    subset = df[(df["Div"] == div) & (df["season"] == season)].copy()
    if len(subset) == 0:
        print(f"  No data for {div} {season}")
        return None

    # Build preseason priors from previous season
    preseason_priors = build_preseason_priors(div, season, df)

    res = fit_ratings(subset, preseason_priors=preseason_priors)
    if res is None:
        print(f"  Could not fit ratings for {div} {season}")
        return None

    ratings  = res["ratings"]
    home_adv = res["home_adv"]
    n_games  = res["n_games"]
    ref_date = res["ref_date"]

    # Recentre ratings so league mean = 0 (ghost-team prior causes slight offset)
    mean_r  = np.mean(list(ratings.values()))
    ratings = {t: r - mean_r for t, r in ratings.items()}

    # Count games per team
    games_played = {}
    for _, row in subset.iterrows():
        games_played[row["HomeTeam"]] = games_played.get(row["HomeTeam"], 0) + 1
        games_played[row["AwayTeam"]] = games_played.get(row["AwayTeam"], 0) + 1

    rows = []
    for team, rating in ratings.items():
        gp = games_played.get(team, 0)
        if gp < MIN_GAMES: continue
        win_pct = win_pct_vs_500(rating)
        rows.append({
            "Team":    team,
            "Rating":  round(rating, 3),
            "Win%":    round(win_pct * 100, 1),
            "GP":      gp,
        })

    table = (pd.DataFrame(rows)
             .sort_values("Rating", ascending=False)
             .reset_index(drop=True))
    table.index += 1

    div_name = DIV_NAMES.get(div, div)
    print(f"\n{'='*70}")
    print(f"  {div_name} ({div})  |  {season}  |  "
          f"{n_games} games  |  as of {ref_date.date()}")
    print(f"  Home advantage: {home_adv:+.3f} goals  |  "
          f"Anchor: {res['anchor']} = 0.000  |  attack/defence: LM+xG")
    print(f"{'='*70}")
    print(f"  {'#':>3} {'Team':<22} {'Overall':>8} {'Attack':>8} {'Defence':>9} "
          f"{'Win% vs .500':>13} {'GP':>4}")
    print(f"  {'-'*72}")

    attack  = res.get("attack",  {})
    defence = res.get("defence", {})

    for idx, row in table.iterrows():
        t       = row["Team"]
        bar_len = int((row["Win%"] - 30) / 2)
        bar     = "█" * max(bar_len, 0)
        atk_v   = attack.get(t, 0.0)
        dfc_v   = defence.get(t, 0.0)
        print(f"  {idx:>3} {t:<22} {row['Rating']:>+8.3f} "
              f"{atk_v:>+8.3f} {dfc_v:>+9.3f} "
              f"{row['Win%']:>12.1f}%  {row['GP']:>4}  {bar}")

    return table, home_adv


# ── 6. MAIN ───────────────────────────────────────────────────────────────────
def main(divs=None, season=CURRENT_SEASON):
    if divs is None:
        divs = sorted(df[df["season"]==season]["Div"].unique())

    all_tables = {}
    for div in divs:
        result = build_ratings_table(div, season)
        if result:
            all_tables[div] = result

    print(f"\n\nSummary: fitted {len(all_tables)} leagues")
    print(f"Hyperparams: half_life={HALF_LIFE}d  market_half_life={MARKET_HALF_LIFE}d  "
          f"market_weight={MARKET_WEIGHT}  form_weight={FORM_WEIGHT}")
    return all_tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--div",    nargs="*", default=None,
                        help="Division codes e.g. E0 D1 SP1")
    parser.add_argument("--season", default=CURRENT_SEASON)
    args = parser.parse_args()
    main(args.div, args.season)
