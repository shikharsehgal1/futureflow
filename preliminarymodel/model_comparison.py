import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from scipy.optimize import minimize
import pickle
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("data/big5_with_probs.csv")
df["Date"]     = pd.to_datetime(df["Date"])
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]
df["Result"]   = np.where(df["GoalDiff"] > 0, "H",
                 np.where(df["GoalDiff"] < 0, "A", "D"))

with open("data/mean_goals.pkl", "rb") as f:
    mg = pickle.load(f)
MEAN_HOME     = mg["home"]
MEAN_AWAY     = mg["away"]
COVID_SEASONS = ["2019-20", "2020-21"]

TRAIN_SEASONS = [s for s in df["season"].unique()
                 if s not in ("2023-24","2024-25","2025-26")]
TEST_SEASON   = "2023-24"
MIN_GAMES     = 10

BLEND_WEIGHTS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0]

# ══════════════════════════════════════════════════════════════════════════════
# PROBABILITY CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def poisson_probs(mu, max_goals=10):
    total    = MEAN_HOME + MEAN_AWAY
    lH = np.clip((total + mu) / 2, 0.1, None)
    lA = np.clip((total - mu) / 2, 0.1, None)
    g  = np.arange(max_goals + 1)
    pmf_H = np.exp(-lH) * lH**g / factorial(g)
    pmf_A = np.exp(-lA) * lA**g / factorial(g)
    M = np.outer(pmf_H, pmf_A)
    return np.clip([np.tril(M,-1).sum(), M.trace(), np.triu(M,1).sum()],
                   1e-6, 1-1e-6)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log_loss(preds, pH_col="pH", pD_col="pD", pA_col="pA"):
    losses = []
    for _, r in preds.iterrows():
        p = {"H": r[pH_col], "D": r[pD_col], "A": r[pA_col]}[r["Result"]]
        losses.append(-np.log(max(p, 1e-6)))
    return float(np.mean(losses))

def make_pred_row(row, mu, home, away, div, season, date):
    pH, pD, pA = poisson_probs(mu)
    return {
        "Date": date, "Div": div, "season": season,
        "HomeTeam": home, "AwayTeam": away,
        "Result": row["Result"], "GoalDiff": row["GoalDiff"],
        "mu": mu, "pH": pH, "pD": pD, "pA": pA,
        "p_H_mkt": row["p_H_mkt"],
        "p_D_mkt": row["p_D_mkt"],
        "p_A_mkt": row["p_A_mkt"],
        "mu_mkt":  row["mu_mkt"],
    }

def ols_ratings(subset, ref_date, half_life=None, prior_w=0.5, ha_prior=0.3):
    subset = subset.copy().reset_index(drop=True)
    subset["GoalDiff"] = subset["HomeGoals"] - subset["AwayGoals"]
    if half_life is None:
        subset["w"] = 1.0
    else:
        subset["w"] = 0.5 ** (
            (pd.Timestamp(ref_date) - subset["Date"]).dt.days / half_life)

    teams      = sorted(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    anchor     = teams[0]
    free_teams = [t for t in teams if t != anchor]
    col_order  = free_teams + ["Home_Adv"]

    def make_row(home, away):
        r = {t: 0 for t in col_order}; r["Home_Adv"] = 1
        if home in free_teams: r[home] =  1
        if away in free_teams: r[away] = -1
        return [r[c] for c in col_order]

    X_real = np.array([make_row(r.HomeTeam, r.AwayTeam)
                       for _, r in subset.iterrows()])
    y_real = subset["GoalDiff"].values.astype(float)
    w_real = subset["w"].values.astype(float)

    X_p, y_p, w_p = [], [], []
    for t in free_teams:
        r = {c: 0 for c in col_order}; r["Home_Adv"] = 0
        X_p.append([r[c] for c in col_order]); y_p.append(0.0); w_p.append(prior_w)
    r = {c: 0 for c in col_order}; r["Home_Adv"] = 1
    X_p.append([r[c] for c in col_order]); y_p.append(ha_prior); w_p.append(prior_w)

    X_all = np.vstack([X_real, np.array(X_p)])
    y_all = np.concatenate([y_real, np.array(y_p)])
    w_all = np.concatenate([w_real, np.array(w_p)])
    sq    = np.sqrt(w_all)
    beta, _, _, _ = np.linalg.lstsq(
        X_all * sq[:, None], y_all * sq, rcond=None)
    if not np.all(np.isfinite(beta)): return None, None

    coefs = dict(zip(col_order, beta))
    is_covid = subset["season"].isin(COVID_SEASONS)
    if is_covid.any() and not is_covid.all():
        X_nc = X_real[~is_covid.values]; y_nc = y_real[~is_covid.values]
        w_nc = w_real[~is_covid.values]
        coefs["Home_Adv"] = float(np.average(
            y_nc - X_nc[:, :-1] @ beta[:-1], weights=w_nc))

    return {anchor: 0.0, **{t: coefs[t] for t in free_teams}}, coefs["Home_Adv"]

def evaluate(preds, name):
    if len(preds) == 0:
        return {"name": name, "N": 0, "ll_model": np.nan,
                "ll_mkt": np.nan, "gap": np.nan, "rmse": np.nan,
                "draw_mod": np.nan, "draw_act": np.nan}
    ll_m  = log_loss(preds, "p_H_mkt","p_D_mkt","p_A_mkt")
    ll_d  = log_loss(preds)
    rmse  = float(np.sqrt(np.mean((preds["GoalDiff"] - preds["mu"])**2)))
    return {"name": name, "N": len(preds),
            "ll_mkt": ll_m, "ll_model": ll_d, "gap": ll_m - ll_d,
            "rmse": rmse, "draw_mod": preds["pD"].mean(),
            "draw_act": (preds["Result"] == "D").mean()}

def print_results(results, title):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(f"  {'Model':<38} {'N':>6} {'LL_mkt':>8} {'LL_mdl':>8} "
          f"{'Gap':>7} {'RMSE':>7} {'Draw%':>7}")
    best_ll = np.inf; best_name = ""
    for r in results:
        if r["N"] == 0: continue
        marker = ""
        if r["name"] != "1. Market" and r["ll_model"] < best_ll:
            best_ll = r["ll_model"]; best_name = r["name"]
        print(f"  {r['name']:<38} {r['N']:>6,} {r['ll_mkt']:>8.4f} "
              f"{r['ll_model']:>8.4f} {r['gap']:>+7.4f} "
              f"{r['rmse']:>7.4f} {r['draw_mod']:>7.3f}")
    print(f"\n  Best: {best_name}  LL={best_ll:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — PURE MARKET
# ══════════════════════════════════════════════════════════════════════════════

def model_market(df_sub):
    preds = []
    for _, row in df_sub.iterrows():
        preds.append(make_pred_row(row, row["mu_mkt"], row["HomeTeam"],
                     row["AwayTeam"], row["Div"], row["season"], row["Date"]))
    return pd.DataFrame(preds)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — PURE RATINGS (walk-forward OLS)
# ══════════════════════════════════════════════════════════════════════════════

def model_pure_ratings(df_sub, half_life=None):
    preds = []
    for (div, season), group in df_sub.groupby(["Div","season"]):
        group = group.sort_values("Date")
        for gd in group["Date"].unique():
            train   = group[group["Date"] < gd]
            predict = group[group["Date"] == gd]
            if len(train) < MIN_GAMES: continue
            ratings, ha = ols_ratings(train, ref_date=gd, half_life=half_life)
            if ratings is None: continue
            for _, row in predict.iterrows():
                mu = ratings.get(row["HomeTeam"],0.0) - \
                     ratings.get(row["AwayTeam"],0.0) + ha
                preds.append(make_pred_row(row, mu, row["HomeTeam"],
                             row["AwayTeam"], div, season, row["Date"]))
    return pd.DataFrame(preds)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — BLEND: RATINGS + MARKET MU (walk-forward)
# ══════════════════════════════════════════════════════════════════════════════

def model_blend(df_sub, half_life=None, blend_w=0.5):
    preds = []
    for (div, season), group in df_sub.groupby(["Div","season"]):
        group = group.sort_values("Date")
        for gd in group["Date"].unique():
            train   = group[group["Date"] < gd]
            predict = group[group["Date"] == gd]
            if len(train) < MIN_GAMES: continue
            ratings, ha = ols_ratings(train, ref_date=gd, half_life=half_life)
            if ratings is None: continue
            for _, row in predict.iterrows():
                mu_model = ratings.get(row["HomeTeam"],0.0) - \
                           ratings.get(row["AwayTeam"],0.0) + ha
                mu = (1-blend_w) * mu_model + blend_w * row["mu_mkt"]
                preds.append(make_pred_row(row, mu, row["HomeTeam"],
                             row["AwayTeam"], div, season, row["Date"]))
    return pd.DataFrame(preds)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — ELO SEQUENTIAL UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def model_elo(df_sub, K=0.05, home_adv=0.3):
    preds = []
    for (div, season), group in df_sub.groupby(["Div","season"]):
        group = group.sort_values("Date")
        ratings = {}
        for _, row in group.iterrows():
            h = row["HomeTeam"]; a = row["AwayTeam"]
            rH = ratings.get(h, 0.0); rA = ratings.get(a, 0.0)
            mu = rH - rA + home_adv
            preds.append(make_pred_row(row, mu, h, a, div, season, row["Date"]))
            delta = K * (row["HomeGoals"] - row["AwayGoals"] - mu)
            ratings[h] = rH + delta
            ratings[a] = rA - delta
    return pd.DataFrame(preds)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — RATINGS SEEDED FROM PREVIOUS SEASON
# ══════════════════════════════════════════════════════════════════════════════

def model_seeded(df_sub, half_life=None, seed_weight=0.3):
    preds        = []
    prev_ratings = {}
    for season in sorted(df_sub["season"].unique()):
        season_df = df_sub[df_sub["season"] == season]
        for div, group in season_df.groupby("Div"):
            group = group.sort_values("Date")
            prev  = prev_ratings.get(div, {})
            for gd in group["Date"].unique():
                train   = group[group["Date"] < gd]
                predict = group[group["Date"] == gd]
                if len(train) < MIN_GAMES: continue
                ratings, ha = ols_ratings(train, ref_date=gd, half_life=half_life)
                if ratings is None: continue
                if prev:
                    anchor_prev = prev.get(list(ratings.keys())[0], 0.0)
                    for t in ratings:
                        prev_r = prev.get(t, anchor_prev) - anchor_prev
                        ratings[t] = (1-seed_weight)*ratings[t] + seed_weight*prev_r
                for _, row in predict.iterrows():
                    mu = (ratings.get(row["HomeTeam"],0.0) -
                          ratings.get(row["AwayTeam"],0.0) + ha)
                    preds.append(make_pred_row(row, mu, row["HomeTeam"],
                                 row["AwayTeam"], div, season, row["Date"]))
            r, _ = ols_ratings(group, ref_date=group["Date"].max(),
                               half_life=half_life)
            if r: prev_ratings[div] = r
    return pd.DataFrame(preds)

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_from_latest(test_df, train_df, latest_season, half_life=None, blend_w=0.0):
    preds = []
    for (div, _), group in test_df.groupby(["Div","season"]):
        train_sub = train_df[(train_df["Div"]==div) &
                              (train_df["season"]==latest_season)]
        if len(train_sub) == 0: continue
        r, ha = ols_ratings(train_sub, ref_date=train_sub["Date"].max(),
                            half_life=half_life)
        if r is None: continue
        for _, row in group.iterrows():
            mu_model = r.get(row["HomeTeam"],0.0) - r.get(row["AwayTeam"],0.0) + ha
            mu       = (1-blend_w) * mu_model + blend_w * row["mu_mkt"]
            preds.append(make_pred_row(row, mu, row["HomeTeam"],
                         row["AwayTeam"], div, row["season"], row["Date"]))
    return pd.DataFrame(preds)

# ══════════════════════════════════════════════════════════════════════════════
# RUN — WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

wf_df = df[df["season"].isin(["2022-23","2023-24"])].copy()

print("Running walk-forward backtest on 2023-24...")
wf_results = []

print("  Model 1: Market...")
p = model_market(wf_df[wf_df["season"]==TEST_SEASON])
wf_results.append(evaluate(p, "1. Market"))

print("  Model 2: Pure ratings...")
p = model_pure_ratings(wf_df)
p = p[p["season"]==TEST_SEASON]
wf_results.append(evaluate(p, "2. Pure ratings (w=0.0)"))

for bw in BLEND_WEIGHTS[1:]:
    label = f"Blend (w={bw})"
    print(f"  {label}...")
    p = model_blend(wf_df, blend_w=bw)
    p = p[p["season"]==TEST_SEASON]
    wf_results.append(evaluate(p, label))

print("  Model 4: Elo...")
p = model_elo(wf_df)
p = p[p["season"]==TEST_SEASON]
wf_results.append(evaluate(p, "4. Elo (K=0.05)"))

print("  Model 5: Seeded ratings...")
p = model_seeded(wf_df)
p = p[p["season"]==TEST_SEASON]
wf_results.append(evaluate(p, "5. Seeded ratings"))

print_results(wf_results, f"WALK-FORWARD BACKTEST — {TEST_SEASON}")

# ══════════════════════════════════════════════════════════════════════════════
# RUN — SINGLE TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

train_df     = df[df["season"].isin(TRAIN_SEASONS)].copy()
test_df      = df[df["season"] == TEST_SEASON].copy()
latest_season = sorted(TRAIN_SEASONS)[-1]

print("\nRunning single train/test split...")
split_results = []

print("  Model 1: Market...")
p = model_market(test_df)
split_results.append(evaluate(p, "1. Market"))

print("  Model 2: Pure ratings...")
p = predict_from_latest(test_df, train_df, latest_season, blend_w=0.0)
split_results.append(evaluate(p, "2. Pure ratings (w=0.0)"))

for bw in BLEND_WEIGHTS[1:]:
    label = f"Blend (w={bw})"
    print(f"  {label}...")
    p = predict_from_latest(test_df, train_df, latest_season, blend_w=bw)
    split_results.append(evaluate(p, label))

print("  Model 4: Elo...")
p = model_elo(pd.concat([train_df.tail(len(test_df)*2), test_df]))
p = p[p["season"]==TEST_SEASON]
split_results.append(evaluate(p, "4. Elo (K=0.05)"))

print("  Model 5: Seeded ratings...")
p = model_seeded(pd.concat([train_df[train_df["season"]==latest_season], test_df]))
p = p[p["season"]==TEST_SEASON]
split_results.append(evaluate(p, "5. Seeded ratings"))

print_results(split_results, f"SINGLE TRAIN/TEST SPLIT — {TEST_SEASON}")

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-SEASON STABILITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

print("\nMulti-season stability check...")
print(f"  {'Season':<10} {'Best_w':>8} {'LL_best':>9} {'LL_mkt':>9} {'Gap':>8}")

completed = sorted([s for s in df["season"].unique()
                    if s not in ("2024-25","2025-26")])

stability_rows = []
check_weights = [0.0, 0.7, 0.9, 0.95, 0.99, 1.0]

for i, test_s in enumerate(completed[2:], start=2):
    ctx_s  = completed[i-1]
    sub    = df[df["season"].isin([ctx_s, test_s])].copy()
    season_results = {}

    for bw in check_weights:
        if bw == 0.0:
            p = model_pure_ratings(sub)
        elif bw == 1.0:
            p = model_market(sub[sub["season"]==test_s])
        else:
            p = model_blend(sub, blend_w=bw)
        p = p[p["season"]==test_s]
        if len(p) == 0: continue
        season_results[bw] = evaluate(p, str(bw))

    if not season_results: continue
    best_bw  = min(season_results, key=lambda w: season_results[w]["ll_model"])
    best     = season_results[best_bw]
    mkt_ll   = season_results.get(1.0, {}).get("ll_model", np.nan)
    stability_rows.append({
        "season": test_s, "best_w": best_bw,
        "ll_best": best["ll_model"], "ll_mkt": mkt_ll,
        "gap": mkt_ll - best["ll_model"]
    })
    print(f"  {test_s:<10} {best_bw:>8.2f} {best['ll_model']:>9.4f} "
          f"{mkt_ll:>9.4f} {mkt_ll - best['ll_model']:>+8.4f}")

stab_df = pd.DataFrame(stability_rows)
if len(stab_df) > 0:
    print(f"\n  Mean best blend weight : {stab_df['best_w'].mean():.3f}")
    print(f"  Std  best blend weight : {stab_df['best_w'].std():.3f}")
    print(f"  Seasons model beat mkt : {(stab_df['gap'] > 0).sum()}/{len(stab_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

pd.DataFrame(wf_results).to_csv("data/model_comparison_walkforward.csv", index=False)
pd.DataFrame(split_results).to_csv("data/model_comparison_split.csv", index=False)
if len(stab_df) > 0:
    stab_df.to_csv("data/blend_stability.csv", index=False)
print(f"\nSaved results to data/model_comparison_*.csv")