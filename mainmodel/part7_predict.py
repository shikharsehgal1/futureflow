"""
Part 7 — Script 2: Fixture Predictions
Takes upcoming fixtures (from CSV or manual input), applies current ratings,
outputs win/draw/loss probabilities and flags disagreements with market odds.

Fixture CSV format (data/fixtures.csv):
    Date,Div,HomeTeam,AwayTeam,OddsH,OddsD,OddsA
    2025-03-22,E0,Arsenal,Chelsea,2.10,3.40,3.60
    2025-03-22,D1,Bayern Munich,Dortmund,1.55,4.00,6.00

If OddsH/OddsD/OddsA are missing the market comparison section is skipped.

Usage:
    python part7_predict.py                        # read from data/fixtures.csv
    python part7_predict.py --div E0               # filter to EPL only
    python part7_predict.py --alert-threshold 0.05 # flag gaps > 5pp
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from scipy.special import factorial
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
CURRENT_SEASON  = "2025-26"
HALF_LIFE       = 90
MARKET_HALF_LIFE= 30     # tuned via grid search
MARKET_WEIGHT   = 0.95
FORM_WEIGHT     = 0.15
FORM_HALF_LIFE  = 21
MIN_GAMES       = 5
ALERT_THRESHOLD = 0.06
COVID_SEASONS   = ["2019-20","2020-21"]

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

# ── 3. FIT RATINGS ────────────────────────────────────────────────────────────
def fit_ratings(subset, half_life=HALF_LIFE, market_weight=MARKET_WEIGHT,
                market_half_life=MARKET_HALF_LIFE, form_weight=FORM_WEIGHT,
                form_half_life=FORM_HALF_LIFE, ha_prior=0.3,
                preseason_priors=None):
    """
    Fit goal-differential power ratings via weighted OLS.

    Upgrades:
    - Recency-weighted market signal: mu_mkt uses steeper decay (market_half_life)
    - Form/momentum: short-window rating blended via form_weight
    """
    subset = subset.copy().reset_index(drop=True)
    subset["GoalDiff"] = subset["HomeGoals"] - subset["AwayGoals"]

    n_games        = len(subset)
    n_teams        = len(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    games_per_team = max(n_games * 2 / n_teams, 0.1)
    adaptive_prior = 3.0 / games_per_team

    ref_date = subset["Date"].max()
    days_ago = (ref_date - subset["Date"]).dt.days

    decay_w      = (0.5 ** (days_ago / half_life)).astype(float) if half_life \
                   else np.ones(n_games)
    decay_w_mkt  = (0.5 ** (days_ago / market_half_life)).astype(float) \
                   if market_half_life else np.ones(n_games)
    decay_w_form = (0.5 ** (days_ago / form_half_life)).astype(float) \
                   if form_half_life else np.ones(n_games)

    teams      = sorted(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    anchor     = "__LEAGUE_AVG__"
    free_teams = teams
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

    # Weather control — downweight actual goal diff rows for extreme conditions
    if "weather_adj" in subset.columns:
        weather_w = subset["weather_adj"].fillna(1.0).values.astype(float)
    else:
        weather_w = np.ones(n_games)

    X_games = np.vstack([X, X])
    y_games = np.concatenate([y_actual, y_implied])
    w_games = np.concatenate([
        decay_w     * (1 - market_weight) * weather_w,  # actual — downweighted by weather
        decay_w_mkt * market_weight,                     # implied — not weather-adjusted
    ])

    # ── Prior rows — decoupled team shrinkage and HFA anchor ─────────────────
    # Two independent priors with separate weights:
    #
    # Team prior:  [team=+1, HFA=0] = 0
    #   → rating_team = 0  → shrinks team toward league average
    #   → HFA NOT included — team shrinkage does not drag HFA estimate
    #   → weighted by team_prior_w (3.0 / games_per_team)
    #
    # HFA prior:   [HFA=+1] = ha_prior
    #   → HFA = ha_prior  → pure HFA anchor, independent of team ratings
    #   → weighted by hfa_prior_w (5.0 / games_per_team — harder anchor)
    team_prior_w = adaptive_prior          # 3.0 / games_per_team
    hfa_prior_w  = adaptive_prior * 5/3   # 5.0 / games_per_team

    Xp, yp, wp = [], [], []

    # Team shrinkage rows — one per team, HFA column = 0
    for t in free_teams:
        r = {c: 0 for c in col_order}   # HFA stays 0
        r[t] = 1
        Xp.append([r[c] for c in col_order]); yp.append(0.0); wp.append(team_prior_w)

    # HFA anchor row — pure HFA signal, no team indicators
    r = {c: 0 for c in col_order}; r["Home_Adv"] = 1
    Xp.append([r[c] for c in col_order]); yp.append(ha_prior); wp.append(hfa_prior_w)

    # Preseason prior rows — fades naturally with adaptive_prior
    if preseason_priors:
        ps_weight = PRESEASON_WEIGHT * adaptive_prior
        for t, prior_val in preseason_priors.items():
            if t not in free_teams: continue
            r = {c: 0 for c in col_order}
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

    # COVID correction
    is_covid = subset["season"].isin(COVID_SEASONS) if "season" in subset.columns \
               else pd.Series(False, index=subset.index)
    if is_covid.any() and not is_covid.all():
        Xnc = X[~is_covid.values]
        ync = y_actual[~is_covid.values]
        wnc = decay_w[~is_covid.values]
        coefs["Home_Adv"] = float(np.average(
            ync - Xnc[:,:-1] @ beta[:-1], weights=wnc))

    base_ratings = {t: coefs[t] for t in free_teams}

    # Form/momentum blend
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
    overall_arr = np.array([final_ratings.get(t, 0.0) for t in teams])
    n = len(teams)
    mean_goals  = (MEAN_HOME + MEAN_AWAY) / 2

    hg = subset["HomeGoals"].values.astype(int)
    ag = subset["AwayGoals"].values.astype(int)
    has_sot = ("HST" in subset.columns and "AST" in subset.columns and
               subset["HST"].notna().mean() > 0.5)
    if has_sot:
        valid = subset[subset["HST"].notna() & subset["AST"].notna() &
                       (subset["HST"] > 0) & (subset["AST"] > 0)]
        hconv = valid["HomeGoals"].sum()/valid["HST"].sum() if len(valid)>=MIN_GAMES else 0.316
        aconv = valid["AwayGoals"].sum()/valid["AST"].sum() if len(valid)>=MIN_GAMES else 0.306
        xg_h  = np.maximum(subset["HST"].fillna(pd.Series(hg/max(hconv,0.1), index=subset.index)).values * hconv, 0.05)
        xg_a  = np.maximum(subset["AST"].fillna(pd.Series(ag/max(aconv,0.1), index=subset.index)).values * aconv, 0.05)
    else:
        xg_h = np.maximum(hg.astype(float), 0.05)
        xg_a = np.maximum(ag.astype(float), 0.05)

    G = np.zeros((n, n))
    for k,(ht,at,xgh,xga,wk) in enumerate(zip(
            subset["HomeTeam"],subset["AwayTeam"],xg_h,xg_a,decay_w)):
        hi_k=teams.index(ht); ai_k=teams.index(at)
        G[ai_k,hi_k]+=wk*xgh; G[hi_k,ai_k]+=wk*xga

    od_off=np.ones(n); od_def=np.ones(n)
    for _ in range(50):
        od_off_new=np.array([sum(G[i,j]/max(od_def[i],1e-6) for i in range(n)) for j in range(n)])
        od_def_new=np.array([sum(G[i,j]/max(od_off_new[j],1e-6) for j in range(n)) for i in range(n)])
        if np.max(np.abs(od_off_new-od_off))<1e-6 and np.max(np.abs(od_def_new-od_def))<1e-6: break
        od_off,od_def=od_off_new,od_def_new

    atk_raw=np.log(np.maximum(od_off,1e-6)); atk_raw-=atk_raw.mean()
    dfc_raw=np.log(np.maximum(od_def,1e-6)); dfc_raw-=dfc_raw.mean()
    SPLIT_DAMP = 0.15
    split=( atk_raw-dfc_raw)*SPLIT_DAMP
    atk_fin=overall_arr/2+split; dfc_fin=-overall_arr/2+split

    return {
        "ratings":  final_ratings,
        "attack":   {teams[i]: atk_fin[i] for i in range(n)},
        "defence":  {teams[i]: dfc_fin[i] for i in range(n)},
        "home_adv": coefs["Home_Adv"],
    }

# ── 3b. PRESEASON PRIORS ─────────────────────────────────────────────────────
def build_preseason_priors(div, season, all_df):
    """Same logic as part7_ratings — see that file for full comments."""
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


# ── 4. PROBABILITY CONVERSION ─────────────────────────────────────────────────
def poisson_probs(mu, max_goals=10, lam_atk=None, mu_atk=None):
    """
    H/D/A from symmetric Poisson (market-anchored).
    p_over25 from asymmetric LM lambdas if provided, else symmetric.
    """
    total = MEAN_HOME + MEAN_AWAY
    lH = np.clip((total + mu) / 2, 0.1, None)
    lA = np.clip((total - mu) / 2, 0.1, None)
    g  = np.arange(max_goals + 1)
    M_sym = np.outer(np.exp(-lH) * lH**g / factorial(g),
                     np.exp(-lA) * lA**g / factorial(g))
    pH, pD, pA = np.clip([np.tril(M_sym,-1).sum(), M_sym.trace(),
                           np.triu(M_sym,1).sum()], 1e-6, 1-1e-6)
    # p_over25 uses asymmetric lambdas for meaningful variation per game
    if lam_atk is not None and mu_atk is not None:
        M_atk = np.outer(np.exp(-lam_atk) * lam_atk**g / factorial(g),
                         np.exp(-mu_atk)  * mu_atk**g  / factorial(g))
        p_over25 = float(sum(M_atk[i,j] for i in range(max_goals+1)
                                         for j in range(max_goals+1) if i+j>2))
        xgh_out = round(float(lam_atk), 2)
        xga_out = round(float(mu_atk),  2)
    else:
        p_over25 = float(sum(M_sym[i,j] for i in range(max_goals+1)
                                         for j in range(max_goals+1) if i+j>2))
        xgh_out = round(float(lH), 2)
        xga_out = round(float(lA), 2)
    return pH, pD, pA, p_over25, xgh_out, xga_out

def odds_to_probs(h, d, a):
    raw = np.array([1/h, 1/d, 1/a])
    return raw / raw.sum()

# ── 5. FIT ALL CURRENT RATINGS ────────────────────────────────────────────────
print("Fitting current season ratings...")
league_ratings = {}
for div in sorted(df[df["season"]==CURRENT_SEASON]["Div"].unique()):
    subset = df[(df["Div"]==div) & (df["season"]==CURRENT_SEASON)].copy()
    if len(subset) < MIN_GAMES: continue
    res = fit_ratings(subset,
                      market_half_life=MARKET_HALF_LIFE,
                      form_weight=FORM_WEIGHT,
                      form_half_life=FORM_HALF_LIFE,
                      preseason_priors=build_preseason_priors(div, CURRENT_SEASON, df))
    if res:
        # Recentre ratings so league mean = 0 (ghost-team prior causes slight offset)
        mean_r = np.mean(list(res["ratings"].values()))
        res["ratings"] = {t: r - mean_r for t, r in res["ratings"].items()}
        league_ratings[div] = res
        print(f"  {div:<6} {DIV_NAMES.get(div, div):<25} "
              f"{len(subset)} games  ha={res['home_adv']:+.3f}")

# ── 6. LOAD FIXTURES ──────────────────────────────────────────────────────────
try:
    fixtures = pd.read_csv("data/fixtures.csv")
    fixtures["Date"] = pd.to_datetime(fixtures["Date"])
    print(f"\nLoaded {len(fixtures)} fixtures from data/fixtures.csv")
except FileNotFoundError:
    print("\nNo fixtures.csv found — creating sample template...")
    sample = pd.DataFrame([
        {"Date":"2025-03-22","Div":"E0","HomeTeam":"Arsenal",
         "AwayTeam":"Chelsea","OddsH":2.10,"OddsD":3.40,"OddsA":3.60},
        {"Date":"2025-03-22","Div":"E0","HomeTeam":"Liverpool",
         "AwayTeam":"Man City","OddsH":2.80,"OddsD":3.20,"OddsA":2.60},
        {"Date":"2025-03-22","Div":"D1","HomeTeam":"Bayern Munich",
         "AwayTeam":"Dortmund","OddsH":1.55,"OddsD":4.00,"OddsA":6.00},
        {"Date":"2025-03-22","Div":"SP1","HomeTeam":"Real Madrid",
         "AwayTeam":"Barcelona","OddsH":2.20,"OddsD":3.30,"OddsA":3.40},
    ])
    sample.to_csv("data/fixtures.csv", index=False)
    fixtures = sample
    print("Created data/fixtures.csv — edit with your fixtures and rerun.")

# Merge injury adjustments if available
if os.path.exists("data/tm_injuries.csv"):
    inj = pd.read_csv("data/tm_injuries.csv")[
        ["HomeTeam","AwayTeam","injury_adj_home",
         "injury_adj_away","injury_adj_net","injury_notes"]
    ]
    fixtures = fixtures.merge(inj, on=["HomeTeam","AwayTeam"], how="left")
    fixtures["injury_adj_net"] = pd.to_numeric(fixtures["injury_adj_net"], errors="coerce").fillna(0.0)
    n_inj = fixtures["injury_adj_net"].notna().sum()
    if n_inj > 0:
        nonzero = (fixtures["injury_adj_net"].fillna(0) != 0).sum()
        print(f"  Injury adjustments loaded: {n_inj} fixtures "
              f"({nonzero} with non-zero adjustment)")
else:
    fixtures["injury_adj_net"] = 0.0

# ── 7. GENERATE PREDICTIONS ───────────────────────────────────────────────────
has_odds = all(c in fixtures.columns for c in ["OddsH","OddsD","OddsA"])
pred_rows = []

for _, fix in fixtures.iterrows():
    div  = fix["Div"]
    home = fix["HomeTeam"]
    away = fix["AwayTeam"]

    if div not in league_ratings:
        print(f"  WARNING: no ratings for {div} — skipping {home} vs {away}")
        continue

    ratings  = league_ratings[div]["ratings"]
    home_adv = league_ratings[div]["home_adv"]

    # Warn on unknown teams but don't crash — use league average (0.0)
    if home not in ratings:
        print(f"  WARNING: '{home}' not found in {div} ratings — using 0.0")
    if away not in ratings:
        print(f"  WARNING: '{away}' not found in {div} ratings — using 0.0")

    h_rat = ratings.get(home, 0.0)
    a_rat = ratings.get(away, 0.0)
    mu    = h_rat - a_rat + home_adv

    # Apply injury adjustment only for RECENT injuries (since_date within 48h)
    # Rationale: closing market odds already price in known injuries.
    # Only players injured in the last 48 hours add genuine new signal.
    inj_adj = float(fix.get("injury_adj_net", 0.0) or 0.0)
    if np.isnan(inj_adj):
        inj_adj = 0.0
    if inj_adj != 0.0 and os.path.exists("data/tm_injuries.csv"):
        try:
            inj_df  = pd.read_csv("data/tm_injuries.csv")
            cutoff  = pd.Timestamp.today() - pd.Timedelta(hours=48)
            # Find this fixture's row
            mask = ((inj_df["HomeTeam"] == fix["HomeTeam"]) &
                    (inj_df["AwayTeam"] == fix["AwayTeam"]))
            fix_row = inj_df[mask]
            # Parse since_dates from injury_notes and check if any are fresh
            fresh_adj = 0.0
            if len(fix_row) > 0:
                notes = str(fix_row.iloc[0].get("injury_notes", ""))
                # Re-read raw cache for since_dates
                cache_path = "data/tm_injuries_cache.json"
                if os.path.exists(cache_path):
                    with open(cache_path) as f:
                        cache = json.load(f)
                    for team, side in [(fix["HomeTeam"], 1), (fix["AwayTeam"], -1)]:
                        key  = f"injuries_{team}"
                        data = cache.get(key, {}).get("data", [])
                        for p in data:
                            since = p.get("since_date", "")
                            if since:
                                try:
                                    # TM format: DD/MM/YYYY
                                    since_dt = pd.to_datetime(since, dayfirst=True)
                                    if since_dt >= cutoff:
                                        # Fresh injury — compute its individual contribution
                                        squad_key = f"squad_vals_{team}"
                                        squad_total = 300.0
                                        sq = cache.get(squad_key, {}).get("data", {})
                                        if sq:
                                            squad_total = sq.get("total", 300.0)
                                        from tm_injuries import POSITION_WEIGHTS, DEFAULT_POSITION_WEIGHT
                                        val_share = p["value_m"] / squad_total
                                        pos_w     = POSITION_WEIGHTS.get(
                                            p["position"], DEFAULT_POSITION_WEIGHT)
                                        contrib   = -(val_share * pos_w * (0.20 / 0.14))
                                        fresh_adj += contrib * side
                                except Exception:
                                    pass
            mu = mu + fresh_adj
            if fresh_adj != 0.0:
                print(f"    Fresh injury adj: {fix['HomeTeam']} vs {fix['AwayTeam']} "
                      f"→ {fresh_adj:+.3f} goals")
        except Exception:
            pass

    # Asymmetric lambdas from LM attack/defence for xGH/xGA and p_over25
    atk     = league_ratings[div].get("attack",  {})
    dfc     = league_ratings[div].get("defence", {})
    mean_g  = (MEAN_HOME + MEAN_AWAY) / 2
    hfa_log = float(np.log(1 + max(home_adv, 0) / mean_g))
    lam_atk = float(np.clip(np.exp(atk.get(home,0) - dfc.get(away,0) + hfa_log), 0.2, 3.5))
    mu_atk  = float(np.clip(np.exp(atk.get(away,0) - dfc.get(home,0)),            0.2, 3.5))

    pH, pD, pA, p_over25, xgh, xga = poisson_probs(mu, lam_atk=lam_atk, mu_atk=mu_atk)

    # Apply weather adjustment if available (scales total goals, not goal diff)
    # Sources: Pavlinovic 2024 (temp), Zhong 2024 (wind/humidity), Bray 2008 (wind)
    weather_adj = fix.get("weather_adj", 1.0)
    if weather_adj and weather_adj != 1.0:
        adj = float(weather_adj)
        # Re-run poisson with scaled lambdas
        pH, pD, pA, p_over25, xgh, xga = poisson_probs(
            mu, lam_atk=lam_atk * adj, mu_atk=mu_atk * adj)

    fair_H    = round(1 / pH, 2)
    fair_D    = round(1 / pD, 2)
    fair_A    = round(1 / pA, 2)
    ah_spread = round(mu * 4) / 4

    row = {
        "Date":        fix["Date"].date() if hasattr(fix["Date"], "date") else fix["Date"],
        "Div":         div,
        "HomeTeam":    home,
        "AwayTeam":    away,
        "mu":          round(mu, 3),
        "xGH":         xgh,
        "xGA":         xga,
        "pH_model":    round(pH * 100, 1),
        "pD_model":    round(pD * 100, 1),
        "pA_model":    round(pA * 100, 1),
        "pOver_model": round(p_over25 * 100, 1),
        "fair_H":      fair_H,
        "fair_D":      fair_D,
        "fair_A":      fair_A,
        "ah_spread":   ah_spread,
    }

    if has_odds and pd.notna(fix.get("OddsH")):
        mH, mD, mA = odds_to_probs(fix["OddsH"], fix["OddsD"], fix["OddsA"])
        # Expected value: model_prob * market_odds - 1
        ev_H = round((pH * fix["OddsH"] - 1) * 100, 1)
        ev_D = round((pD * fix["OddsD"] - 1) * 100, 1)
        ev_A = round((pA * fix["OddsA"] - 1) * 100, 1)
        row.update({
            "pH_mkt":  round(mH * 100, 1),
            "pD_mkt":  round(mD * 100, 1),
            "pA_mkt":  round(mA * 100, 1),
            "diff_H":  round((pH - mH) * 100, 1),
            "diff_D":  round((pD - mD) * 100, 1),
            "diff_A":  round((pA - mA) * 100, 1),
            "ev_H":    ev_H,   # % edge on home bet at market odds
            "ev_D":    ev_D,
            "ev_A":    ev_A,
            "best_ev": max(ev_H, ev_D, ev_A),
            "alert":   max(abs(pH-mH), abs(pD-mD), abs(pA-mA)) > ALERT_THRESHOLD,
        })

    # Over/under market if OddsOver/OddsUnder in fixtures
    if all(c in fix.index for c in ["OddsOver","OddsUnder"]) and pd.notna(fix.get("OddsOver")):
        raw_ou = np.array([1/fix["OddsOver"], 1/fix["OddsUnder"]]); raw_ou /= raw_ou.sum()
        mOver = raw_ou[0]
        row.update({
            "pOver_mkt":  round(mOver * 100, 1),
            "diff_Over":  round((p_over25 - mOver) * 100, 1),
            "ev_Over":    round((p_over25 * fix["OddsOver"] - 1) * 100, 1),
            "ev_Under":   round(((1-p_over25) * fix["OddsUnder"] - 1) * 100, 1),
        })
    pred_rows.append(row)

if not pred_rows:
    print("\nNo predictions generated — check fixtures.csv team names match ratings.")
    exit()

preds = pd.DataFrame(pred_rows)

# ── 8. PRINT PREDICTIONS ──────────────────────────────────────────────────────
print(f"\n{'='*78}")
print(f"  PREDICTIONS  (threshold for alert: >{ALERT_THRESHOLD*100:.0f}pp disagreement with market)")
print(f"{'='*78}")

for div, grp in preds.groupby("Div"):
    print(f"\n  {DIV_NAMES.get(div, div)} ({div})")
    print(f"  {'Date':<12} {'Home':<22} {'Away':<22} "
          f"{'xGH':>5} {'xGA':>5} {'H%':>5} {'D%':>5} {'A%':>5} {'O25%':>6}", end="")
    if has_odds:
        print(f"  {'Mkt_H':>6} {'Mkt_D':>6} {'Mkt_A':>6}  {'ΔH':>5} {'ΔD':>5} {'ΔA':>5}",
              end="")
    print()
    print(f"  {'-'*73}")

    for _, row in grp.iterrows():
        alert_flag = "  ◀ ALERT" if row.get("alert") else ""
        line = (f"  {str(row['Date']):<12} {row['HomeTeam']:<22} "
                f"{row['AwayTeam']:<22} "
                f"{row.get('xGH',0):>4.2f} {row.get('xGA',0):>4.2f} "
                f"{row['pH_model']:>4.1f}% {row['pD_model']:>4.1f}% "
                f"{row['pA_model']:>4.1f}% {row['pOver_model']:>5.1f}%")
        if has_odds and "pH_mkt" in row:
            line += (f"  {row['pH_mkt']:>5.1f}% {row['pD_mkt']:>5.1f}% "
                     f"{row['pA_mkt']:>5.1f}%  "
                     f"{row['diff_H']:>+5.1f} {row['diff_D']:>+5.1f} "
                     f"{row['diff_A']:>+5.1f}")
        print(line + alert_flag)

# ── 9. ALERTS SUMMARY ─────────────────────────────────────────────────────────
if has_odds and "alert" in preds.columns:
    alerts = preds[preds["alert"] == True]
    if len(alerts) > 0:
        print(f"\n{'='*78}")
        print(f"  ALERTS — {len(alerts)} game(s) where model disagrees "
              f"with market by >{ALERT_THRESHOLD*100:.0f}pp")
        print(f"{'='*78}")
        for _, row in alerts.iterrows():
            diffs = {"Home": row["diff_H"], "Draw": row["diff_D"], "Away": row["diff_A"]}
            biggest   = max(diffs, key=lambda k: abs(diffs[k]))
            direction = "higher" if diffs[biggest] > 0 else "lower"
            print(f"\n  {row['HomeTeam']} vs {row['AwayTeam']} ({row['Div']})")
            print(f"    Model sees {biggest} probability as {direction} than market "
                  f"({diffs[biggest]:+.1f}pp)")
            print(f"    Model: H={row['pH_model']}%  D={row['pD_model']}%  "
                  f"A={row['pA_model']}%")
            print(f"    Mkt:   H={row['pH_mkt']}%  D={row['pD_mkt']}%  "
                  f"A={row['pA_mkt']}%")
    else:
        print(f"\n  No alerts — all predictions within "
              f"{ALERT_THRESHOLD*100:.0f}pp of market.")

# ── 10. MARKET PRICING TABLE ─────────────────────────────────────────────────
# Fair odds, EV per outcome, and Asian handicap spread for each fixture.
# EV > 0% means the model sees positive expected value at the market price.
# AH spread rounded to nearest 0.25 (standard bookmaker line increment).
if has_odds and "ev_H" in preds.columns:
    print(f"\n{'='*90}")
    print(f"  MARKET PRICING — Fair Odds  |  Expected Value  |  Asian Handicap Spread")
    print(f"{'='*90}")

    for div, grp in preds.groupby("Div"):
        print(f"\n  {DIV_NAMES.get(div, div)} ({div})")
        print(f"  {'-'*86}")
        print(f"  {'Home':<20} {'Away':<20} {'AH':>6}  "
              f"{'FairH':>6} {'FairD':>6} {'FairA':>6}  "
              f"{'EvH%':>6} {'EvD%':>6} {'EvA%':>6}  {'Best':>9}")
        print(f"  {'-'*86}")
        for _, row in grp.iterrows():
            best_out   = "H" if row["ev_H"] == row["best_ev"] else \
                         "D" if row["ev_D"] == row["best_ev"] else "A"
            value_flag = " ★" if row["best_ev"] > 0 else "  "
            ah         = row["ah_spread"]
            ah_str     = f"+{ah:.2f}" if ah > 0 else f"{ah:.2f}"
            print(f"  {row['HomeTeam']:<20} {row['AwayTeam']:<20} {ah_str:>6}  "
                  f"{row['fair_H']:>6.2f} {row['fair_D']:>6.2f} {row['fair_A']:>6.2f}  "
                  f"{row['ev_H']:>+5.1f}% {row['ev_D']:>+5.1f}% {row['ev_A']:>+5.1f}%  "
                  f"{row['best_ev']:>+5.1f}%{value_flag}({best_out})")

# ── 11. SAVE ──────────────────────────────────────────────────────────────────
preds.to_csv("data/predictions.csv", index=False)
print(f"\nSaved predictions → data/predictions.csv")




# ── 11. APPEND TO PREDICTIONS LOG (persistent history) ────────────────────────
# Used by fetch_fixtures.py --results to score settled predictions
LOG_PATH = "data/predictions_log.csv"
log_cols = ["Date","Div","HomeTeam","AwayTeam",
            "mu","pH_model","pD_model","pA_model","pOver_model",
            "pH_mkt","pD_mkt","pA_mkt",
            "fair_H","fair_D","fair_A","ah_spread"]   # fair odds for CLV tracking
log_row = preds[[c for c in log_cols if c in preds.columns]].copy()
log_row["predicted_at"] = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M")

try:
    existing_log = pd.read_csv(LOG_PATH)
    existing_keys = set(zip(existing_log["Div"],
                            existing_log["HomeTeam"],
                            existing_log["AwayTeam"]))
    log_row["_key"] = list(zip(log_row["Div"], log_row["HomeTeam"], log_row["AwayTeam"]))
    new_rows = log_row[~log_row["_key"].isin(existing_keys)].drop(columns=["_key"])
    updated_log = pd.concat([existing_log, new_rows], ignore_index=True)
    n_new = len(new_rows)
except FileNotFoundError:
    updated_log = log_row.copy()
    n_new = len(log_row)

updated_log.to_csv(LOG_PATH, index=False)
print(f"Appended {n_new} new predictions to log ({len(updated_log)} total) → {LOG_PATH}")
# ── 12. ARGPARSE (for command-line filtering) ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--div",             nargs="*", default=None)
    parser.add_argument("--alert-threshold", type=float, default=ALERT_THRESHOLD)
    parser.add_argument("--season",          default=CURRENT_SEASON)
    args = parser.parse_args()
    ALERT_THRESHOLD = args.alert_threshold
    CURRENT_SEASON  = args.season