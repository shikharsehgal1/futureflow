import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/big5_football.csv")
df["Date"]     = pd.to_datetime(df["Date"])
df             = df.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]
df["Result"]   = np.where(df["GoalDiff"] > 0, "H",
                 np.where(df["GoalDiff"] < 0, "A", "D"))

# ── 2. ODDS → NORMALISED PROBABILITIES ───────────────────────────────────────
for side, ps, bc, bo in [
    ("H", "PSCH", "B365CH", "B365H"),
    ("D", "PSCD", "B365CD", "B365D"),
    ("A", "PSCA", "B365CA", "B365A"),
]:
    df[f"odds_{side}"] = (
        df[ps].where(df[ps].notna() & (df[ps] > 1))
        .fillna(df[bc].where(df[bc].notna() & (df[bc] > 1)))
        .fillna(df[bo].where(df[bo].notna() & (df[bo] > 1)))
    )

# Over/under 2.5 closing odds
df["ou_over"]  = df["B365C>2.5"] if "B365C>2.5" in df.columns else np.nan
df["ou_under"] = df["B365C<2.5"] if "B365C<2.5" in df.columns else np.nan

df = df.dropna(subset=["odds_H","odds_D","odds_A"])
df["raw_sum"]   = 1/df["odds_H"] + 1/df["odds_D"] + 1/df["odds_A"]
df["overround"] = df["raw_sum"] - 1
df = df[df["overround"] >= 0].copy().reset_index(drop=True)

raw = np.column_stack([1/df["odds_H"], 1/df["odds_D"], 1/df["odds_A"]])
raw = raw / raw.sum(axis=1, keepdims=True)
df["p_H_mkt"] = raw[:, 0]
df["p_D_mkt"] = raw[:, 1]
df["p_A_mkt"] = raw[:, 2]

# Normalised O/U probs
has_ou = (df["ou_over"].notna() & df["ou_under"].notna()).values
ou_raw = np.where(
    has_ou[:, None],
    np.column_stack([1/df["ou_over"].fillna(2).values,
                     1/df["ou_under"].fillna(2).values]),
    np.nan)
ou_sum = np.nansum(ou_raw, axis=1, keepdims=True)
ou_norm = np.where(ou_sum > 0, ou_raw / ou_sum, np.nan)
df["p_over_mkt"]  = ou_norm[:, 0]
df["p_under_mkt"] = ou_norm[:, 1]

print(f"Rows: {len(df):,}  |  Mean overround: {df['overround'].mean():.3%}")
print(f"O/U 2.5 coverage: {has_ou.mean():.1%}")

# ── 3. GLOBAL STATS (from full df — used for feature construction only) ───────
MEAN_HOME_ALL = df["HomeGoals"].mean()
MEAN_AWAY_ALL = df["AwayGoals"].mean()

# ── 4. BASE FEATURES (computed on full df before split) ───────────────────────
def clip_p(p): return np.clip(p, 1e-6, 1-1e-6)

for col, p in [("z_H","p_H_mkt"), ("z_A","p_A_mkt"), ("z_D","p_D_mkt")]:
    df[col] = norm.ppf(clip_p(df[p].values))

df["strength_adj"] = (df["p_H_mkt"] - df["p_A_mkt"]) / (1 - df["p_D_mkt"] + 1e-6)
df["z_strength"]   = norm.ppf(clip_p((df["strength_adj"] + 1) / 2))
df["log_odds_ratio"] = np.log(clip_p(df["p_H_mkt"]) / clip_p(df["p_A_mkt"]))
df["z_D_sq"] = df["z_D"] ** 2

_mean_total = MEAN_HOME_ALL + MEAN_AWAY_ALL
df["exp_total"] = np.where(
    ~np.isnan(df["p_over_mkt"].values),
    norm.ppf(clip_p(df["p_over_mkt"].fillna(0.5).values)) *
    _mean_total / 2 + _mean_total,
    np.nan)

# ── 5. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
train_mask = ~df["season"].isin(["2023-24","2024-25","2025-26"])
train = df[train_mask].copy()
test  = df[df["season"] == "2023-24"].copy()

MEAN_HOME    = train["HomeGoals"].mean()
MEAN_AWAY    = train["AwayGoals"].mean()
SIGMA_GLOBAL = train["GoalDiff"].std()
sigma_by_div = train.groupby("Div")["GoalDiff"].std()

print(f"Train: {len(train):,}  |  Test: {len(test):,}")
print(f"Mean goals: home={MEAN_HOME:.3f}  away={MEAN_AWAY:.3f}\n")

# ── 6. DIVISION DUMMIES ───────────────────────────────────────────────────────
div_dummies_train = pd.get_dummies(train["Div"], prefix="div", drop_first=True)
div_dummies_test  = pd.get_dummies(test["Div"],  prefix="div", drop_first=True)
for col in div_dummies_train.columns:
    if col not in div_dummies_test.columns:
        div_dummies_test[col] = 0
div_dummies_test = div_dummies_test[div_dummies_train.columns]
div_cols = list(div_dummies_train.columns)

# ── 7. PROBABILITY CONVERSION ─────────────────────────────────────────────────
def poisson_probs(mu, max_goals=10):
    total = MEAN_HOME + MEAN_AWAY
    lH = np.clip((total + mu) / 2, 0.1, None)
    lA = np.clip((total - mu) / 2, 0.1, None)
    g  = np.arange(max_goals + 1)
    M  = np.outer(np.exp(-lH) * lH**g / factorial(g),
                  np.exp(-lA) * lA**g / factorial(g))
    return np.clip([np.tril(M,-1).sum(), M.trace(), np.triu(M,1).sum()],
                   1e-6, 1-1e-6)

def normal_probs(mu, sigma):
    pH = 1 - norm.cdf(0.5/sigma - mu/sigma)
    pA = norm.cdf(-0.5/sigma - mu/sigma)
    return np.clip([pH, 1-pH-pA, pA], 1e-6, 1-1e-6)

def sigma_for(div):
    return float(sigma_by_div.get(div, SIGMA_GLOBAL))

# ── 8. EVALUATION ─────────────────────────────────────────────────────────────
def eval_model(eval_df, mu_col, prob_fn="poisson"):
    losses = []; draws = []
    for _, r in eval_df.iterrows():
        if pd.isna(r[mu_col]): continue
        if prob_fn == "poisson":
            pH, pD, pA = poisson_probs(r[mu_col])
        else:
            pH, pD, pA = normal_probs(r[mu_col], sigma_for(r["Div"]))
        p = {"H": pH, "D": pD, "A": pA}[r["Result"]]
        losses.append(-np.log(max(p, 1e-6)))
        draws.append(pD)
    return float(np.mean(losses)), float(np.mean(draws))

def fit_and_predict(feat_train, feat_test, y_train, mu_col, base_feat_cols):
    reg = LinearRegression().fit(feat_train, y_train)
    df.loc[train_mask, mu_col]              = reg.predict(feat_train)
    df.loc[df["season"]=="2023-24", mu_col] = reg.predict(feat_test)
    # Apply to 2024-25 and 2025-26 using same fitted model
    future_mask = df["season"].isin(["2024-25","2025-26"])
    if future_mask.any():
        future_div = pd.get_dummies(df.loc[future_mask,"Div"], prefix="div", drop_first=True)\
                       .reindex(columns=div_dummies_train.columns, fill_value=0).values
        feat_future = np.column_stack([
            df.loc[future_mask, base_feat_cols].values,
            future_div
        ])
        df.loc[future_mask, mu_col] = reg.predict(feat_future)
    cv = cross_val_score(reg, feat_train, y_train, cv=5, scoring="r2")
    return reg, cv.mean(), cv.std()

# ── 9. MODEL DEFINITIONS ──────────────────────────────────────────────────────
models = {}

# Model A: baseline probit
feat_A_tr = np.column_stack([train[["z_H","z_A","z_D"]].values, div_dummies_train.values])
feat_A_te = np.column_stack([test[["z_H","z_A","z_D"]].values,  div_dummies_test.values])
reg_A, cv_A, _ = fit_and_predict(feat_A_tr, feat_A_te, train["GoalDiff"].values, "mu_A", ["z_H","z_A","z_D"])
models["A: Probit (baseline)"] = ("mu_A", cv_A, reg_A)

feat_B_tr = np.column_stack([train[["z_strength","z_D","z_D_sq"]].values, div_dummies_train.values])
feat_B_te = np.column_stack([test[["z_strength","z_D","z_D_sq"]].values,  div_dummies_test.values])
reg_B, cv_B, _ = fit_and_predict(feat_B_tr, feat_B_te, train["GoalDiff"].values, "mu_B", ["z_strength","z_D","z_D_sq"])
models["B: Draw-adjusted strength"] = ("mu_B", cv_B, reg_B)

feat_C_tr = np.column_stack([train[["log_odds_ratio","z_D","z_D_sq"]].values, div_dummies_train.values])
feat_C_te = np.column_stack([test[["log_odds_ratio","z_D","z_D_sq"]].values,  div_dummies_test.values])
reg_C, cv_C, _ = fit_and_predict(feat_C_tr, feat_C_te, train["GoalDiff"].values, "mu_C", ["log_odds_ratio","z_D","z_D_sq"])
models["C: Log odds ratio"] = ("mu_C", cv_C, reg_C)

feat_D_tr = np.column_stack([train[["z_H","z_A","z_D","z_D_sq","z_strength","log_odds_ratio"]].values, div_dummies_train.values])
feat_D_te = np.column_stack([test[["z_H","z_A","z_D","z_D_sq","z_strength","log_odds_ratio"]].values,  div_dummies_test.values])
reg_D, cv_D, _ = fit_and_predict(feat_D_tr, feat_D_te, train["GoalDiff"].values, "mu_D", ["z_H","z_A","z_D","z_D_sq","z_strength","log_odds_ratio"])
models["D: Combined features"] = ("mu_D", cv_D, reg_D)

mean_exp = float(train["exp_total"].mean())
exp_tr = train["exp_total"].fillna(mean_exp).values
exp_te = test["exp_total"].fillna(mean_exp).values
feat_E_tr = np.column_stack([train[["z_H","z_A","z_D"]].values, exp_tr[:, None], div_dummies_train.values])
feat_E_te = np.column_stack([test[["z_H","z_A","z_D"]].values,  exp_te[:, None], div_dummies_test.values])
reg_E, cv_E, _ = fit_and_predict(feat_E_tr, feat_E_te, train["GoalDiff"].values, "mu_E", ["z_H","z_A","z_D"])
models["E: Probit + O/U 2.5"] = ("mu_E", cv_E, reg_E)

# Model F: Poisson MLE inversion (test set only — slow)
def poisson_mle_mu(pH_obs, pD_obs, pA_obs, max_goals=10):
    def neg_ll(params):
        lH = np.exp(params[0]); lA = np.exp(params[1])
        g  = np.arange(max_goals + 1)
        pH_pmf = np.exp(-lH) * lH**g / factorial(g)
        pA_pmf = np.exp(-lA) * lA**g / factorial(g)
        M = np.outer(pH_pmf, pA_pmf)
        eps = 1e-6
        return -(pH_obs * np.log(max(np.tril(M,-1).sum(), eps)) +
                 pD_obs * np.log(max(M.trace(), eps)) +
                 pA_obs * np.log(max(np.triu(M,1).sum(), eps)))
    res = minimize(neg_ll, [np.log(MEAN_HOME), np.log(MEAN_AWAY)],
                   method="Nelder-Mead",
                   options={"xatol":1e-4, "fatol":1e-4, "maxiter":200})
    return np.exp(res.x[0]) - np.exp(res.x[1])

print("Computing Poisson MLE inversion (test set — may take a minute)...")
test_mle = []
for _, row in test.iterrows():
    try:
        mu = poisson_mle_mu(row["p_H_mkt"], row["p_D_mkt"], row["p_A_mkt"])
    except Exception:
        mu = np.nan
    test_mle.append(mu)
df.loc[df["season"]=="2023-24", "mu_F"] = test_mle
models["F: Poisson MLE inversion"] = ("mu_F", None, None)

# ── 10. COMPARE ALL MODELS ────────────────────────────────────────────────────
eval_df     = df[df["season"] == "2023-24"].copy()
draw_actual = (eval_df["Result"] == "D").mean()

print(f"\n{'='*72}")
print(f"  MODEL COMPARISON — Test set 2023-24 (N={len(eval_df):,})")
print(f"{'='*72}")
print(f"  {'Model':<35} {'CV_R²':>7} {'LL_pois':>9} {'LL_norm':>9} "
      f"{'Draw%':>7} {'RMSE':>8}")

mkt_losses = [-np.log(max({"H":r.p_H_mkt,"D":r.p_D_mkt,"A":r.p_A_mkt}[r.Result],1e-6))
              for _, r in eval_df.iterrows()]
print(f"  {'Market (direct)':35} {'n/a':>7} {np.mean(mkt_losses):>9.4f} "
      f"{'n/a':>9} {eval_df['p_D_mkt'].mean():>7.3f} {'n/a':>8}")

best_ll = np.inf; best_model_name = ""
for name, (mu_col, cv_r2, _) in models.items():
    sub = eval_df.dropna(subset=[mu_col])
    if len(sub) == 0: continue
    ll_p, draw_p = eval_model(sub, mu_col, "poisson")
    ll_n, _      = eval_model(sub, mu_col, "normal")
    rmse = float(np.sqrt(np.mean((sub["GoalDiff"] - sub[mu_col])**2)))
    cv_str = f"{cv_r2:.4f}" if cv_r2 is not None else "  n/a"
    print(f"  {name:<35} {cv_str:>7} {ll_p:>9.4f} {ll_n:>9.4f} "
          f"{draw_p:>7.3f} {rmse:>8.4f}")
    if ll_p < best_ll:
        best_ll = ll_p; best_model_name = name

print(f"\n  Actual draw rate : {draw_actual:.3f}")
print(f"  Best model       : {best_model_name}  (LL={best_ll:.4f})")
print(f"  Market LL        : {np.mean(mkt_losses):.4f}  "
      f"Gap: {np.mean(mkt_losses)-best_ll:+.4f}")

# ── 11. DRAW CALIBRATION ──────────────────────────────────────────────────────
print(f"\n  DRAW CALIBRATION")
print(f"  {'Model':<35} {'Mean p_D':>9} {'vs actual':>10}")
print(f"  {'Market':35} {eval_df['p_D_mkt'].mean():>9.3f} "
      f"{eval_df['p_D_mkt'].mean()-draw_actual:>+10.3f}")
for name, (mu_col, _, __) in models.items():
    sub = eval_df.dropna(subset=[mu_col])
    if len(sub) == 0: continue
    _, draw_p = eval_model(sub, mu_col, "poisson")
    print(f"  {name:<35} {draw_p:>9.3f} {draw_p-draw_actual:>+10.3f}")

# ── 12. SAVE BEST MODEL ───────────────────────────────────────────────────────
best_mu_col = models[best_model_name][0]
df["mu_mkt"] = df[best_mu_col].fillna(df["mu_A"])  # fallback to A if NaN

sigma_by_div.to_csv("data/sigma_by_div.csv", header=True)
df.to_csv("data/big5_with_probs.csv", index=False)

with open("data/mean_goals.pkl", "wb") as f:
    pickle.dump({"home": MEAN_HOME, "away": MEAN_AWAY}, f)

best_reg = models[best_model_name][2]
if best_reg is not None:
    with open("data/probit_reg.pkl", "wb") as f: pickle.dump(best_reg, f)
with open("data/div_cols.pkl", "wb") as f: pickle.dump(div_cols, f)

pd.DataFrame([{"best_model": best_model_name, "best_mu_col": best_mu_col}]
             ).to_csv("data/part5_best.csv", index=False)

print(f"\nSaved best model '{best_model_name}' as mu_mkt → data/big5_with_probs.csv")