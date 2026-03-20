import pandas as pd
import numpy as np
from scipy.special import factorial
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/big5_with_probs.csv")
df["Date"]     = pd.to_datetime(df["Date"])
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]

with open("data/mean_goals.pkl", "rb") as f:
    mg = pickle.load(f)
MEAN_HOME     = mg["home"]
MEAN_AWAY     = mg["away"]
COVID_SEASONS = ["2019-20", "2020-21"]
MIN_GAMES     = 10

# Completed seasons only — exclude current in-progress seasons
ALL_SEASONS = sorted([s for s in df["season"].unique()
                      if s not in ("2024-25","2025-26")])

# ── 2. CORE FUNCTIONS ─────────────────────────────────────────────────────────
def fit_ratings(subset, ref_date, half_life=None, market_weight=0.0,
                prior_weight=0.5, ha_prior=0.3):
    subset = subset.copy().reset_index(drop=True)
    subset["GoalDiff"] = subset["HomeGoals"] - subset["AwayGoals"]
    subset["y_target"] = ((1 - market_weight) * subset["GoalDiff"] +
                               market_weight   * subset["mu_mkt"])

    n_games        = len(subset)
    n_teams        = len(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    games_per_team = max(n_games * 2 / n_teams, 0.1)
    adaptive_prior = 3.0 / games_per_team

    if half_life is None:
        subset["w"] = 1.0
    else:
        subset["w"] = 0.5 ** (
            (pd.Timestamp(ref_date) - subset["Date"]).dt.days / half_life)

    teams      = sorted(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    anchor     = teams[0]
    free_teams = [t for t in teams if t != anchor]
    col_order  = free_teams + ["Home_Adv"]

    def make_row(h, a):
        r = {t: 0 for t in col_order}; r["Home_Adv"] = 1
        if h in free_teams: r[h] =  1
        if a in free_teams: r[a] = -1
        return [r[c] for c in col_order]

    X  = np.array([make_row(r.HomeTeam, r.AwayTeam) for _, r in subset.iterrows()])
    y  = subset["y_target"].values.astype(float)
    w  = subset["w"].values.astype(float)

    Xp, yp, wp = [], [], []
    for t in free_teams:
        r = {c: 0 for c in col_order}; r["Home_Adv"] = 0
        Xp.append([r[c] for c in col_order]); yp.append(0.0); wp.append(adaptive_prior)
    r = {c: 0 for c in col_order}; r["Home_Adv"] = 1
    Xp.append([r[c] for c in col_order]); yp.append(ha_prior); wp.append(adaptive_prior)

    Xa = np.vstack([X, np.array(Xp)])
    ya = np.concatenate([y, np.array(yp)])
    wa = np.concatenate([w, np.array(wp)])
    sq = np.sqrt(wa)

    beta, _, _, _ = np.linalg.lstsq(Xa * sq[:,None], ya * sq, rcond=None)
    if not np.all(np.isfinite(beta)): return None, None

    coefs = dict(zip(col_order, beta))
    is_covid = subset["season"].isin(COVID_SEASONS)
    if is_covid.any() and not is_covid.all():
        Xnc = X[~is_covid.values]; ync = subset["GoalDiff"].values[~is_covid.values]
        wnc = w[~is_covid.values]
        coefs["Home_Adv"] = float(np.average(ync - Xnc[:,:-1] @ beta[:-1], weights=wnc))

    return {anchor: 0.0, **{t: coefs[t] for t in free_teams}}, coefs["Home_Adv"]

def walk_forward_rmse(subset_df, half_life, market_weight):
    """Walk-forward RMSE for a single (div, season) or multi-season df."""
    actual = []; pred = []
    for (div, season), group in subset_df.groupby(["Div","season"]):
        group    = group.sort_values("Date").reset_index(drop=True)
        gamedays = group["Date"].unique()
        for gd in gamedays:
            train   = group[group["Date"] < gd]
            predict = group[group["Date"] == gd]
            if len(train) < MIN_GAMES: continue
            ratings, ha = fit_ratings(train, ref_date=gd,
                                      half_life=half_life,
                                      market_weight=market_weight)
            if ratings is None: continue
            for _, row in predict.iterrows():
                mu = (ratings.get(row["HomeTeam"], 0.0) -
                      ratings.get(row["AwayTeam"], 0.0) + ha)
                actual.append(row["GoalDiff"])
                pred.append(mu)
    if not actual: return np.nan
    return float(np.sqrt(np.mean((np.array(actual) - np.array(pred))**2)))

# ── 3. MONTE CARLO SETUP ──────────────────────────────────────────────────────
# Instead of a fixed grid, sample (half_life, market_weight) pairs randomly.
# This covers the space more efficiently and avoids overfitting to specific
# grid points. We run each sample on a RANDOM SUBSET of seasons to get a
# distribution of performance, not a single point estimate.

N_SAMPLES    = 200    # number of random hyperparameter pairs to try
N_SEASONS    = 4      # seasons to randomly sample per evaluation
RANDOM_SEED  = 42

rng = np.random.default_rng(RANDOM_SEED)

# Sample hyperparameters from reasonable ranges
# half_life: log-uniform from 30 to inf (None represented as very large number)
# market_weight: uniform from 0 to 1
hl_samples = []
for _ in range(N_SAMPLES):
    u = rng.uniform(0, 1)
    if u < 0.15:
        hl_samples.append(None)          # ~15% chance of no decay
    else:
        # Log-uniform between 30 and 730 days
        hl_samples.append(int(np.exp(rng.uniform(np.log(30), np.log(730)))))

mw_samples = rng.uniform(0.0, 1.0, N_SAMPLES)

print(f"Monte Carlo hyperparameter search")
print(f"  Samples       : {N_SAMPLES}")
print(f"  Seasons/eval  : {N_SEASONS} (randomly sampled each time)")
print(f"  Available seasons: {ALL_SEASONS}\n")

# ── 4. RUN MONTE CARLO ────────────────────────────────────────────────────────
results = []

for i, (hl, mw) in enumerate(zip(hl_samples, mw_samples)):
    # Sample N_SEASONS random seasons
    chosen = rng.choice(ALL_SEASONS, size=min(N_SEASONS, len(ALL_SEASONS)),
                        replace=False).tolist()
    subset = df[df["season"].isin(chosen)].copy()

    rmse = walk_forward_rmse(subset, half_life=hl, market_weight=mw)
    results.append({
        "sample":       i,
        "half_life":    hl,
        "market_weight": mw,
        "rmse":         rmse,
        "seasons":      str(chosen),
    })

    hl_label = str(hl) if hl else "∞"
    if (i+1) % 20 == 0:
        best_so_far = min(r["rmse"] for r in results if not np.isnan(r["rmse"]))
        print(f"  [{i+1:>3}/{N_SAMPLES}]  last: hl={hl_label:>4} mw={mw:.2f} "
              f"rmse={rmse:.4f}  best so far={best_so_far:.4f}")

results_df = pd.DataFrame(results).dropna(subset=["rmse"])

# ── 5. AGGREGATE RESULTS ──────────────────────────────────────────────────────
# Each sample was evaluated on different seasons — aggregate by binning
# hyperparameters and taking the mean RMSE across samples in each bin

print(f"\n{'='*65}")
print(f"  MONTE CARLO RESULTS — {len(results_df)} valid samples")
print(f"{'='*65}")

# ── Half-life marginal effect ─────────────────────────────────────────────────
print(f"\n  Half-life marginal effect (averaged over all market_weight values):")
print(f"  {'HalfLife':>10} {'N':>4} {'Mean RMSE':>10} {'Std':>8} {'Min RMSE':>10}")

results_df["hl_bin"] = results_df["half_life"].apply(
    lambda x: "∞" if x is None else
    "30-60"   if x <= 60  else
    "61-120"  if x <= 120 else
    "121-270" if x <= 270 else
    "271-730")

for bin_label in ["30-60","61-120","121-270","271-730","∞"]:
    grp = results_df[results_df["hl_bin"] == bin_label]
    if len(grp) == 0: continue
    print(f"  {bin_label:>10} {len(grp):>4} {grp['rmse'].mean():>10.4f} "
          f"{grp['rmse'].std():>8.4f} {grp['rmse'].min():>10.4f}")

# ── Market weight marginal effect ─────────────────────────────────────────────
print(f"\n  Market weight marginal effect (averaged over all half_life values):")
print(f"  {'MW range':>10} {'N':>4} {'Mean RMSE':>10} {'Std':>8} {'Min RMSE':>10}")

mw_bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
mw_labels = ["0.0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1.0"]
results_df["mw_bin"] = pd.cut(results_df["market_weight"],
                               bins=mw_bins, labels=mw_labels, right=False)
for label in mw_labels:
    grp = results_df[results_df["mw_bin"] == label]
    if len(grp) == 0: continue
    print(f"  {label:>10} {len(grp):>4} {grp['rmse'].mean():>10.4f} "
          f"{grp['rmse'].std():>8.4f} {grp['rmse'].min():>10.4f}")

# ── Top 10 individual samples ─────────────────────────────────────────────────
print(f"\n  Top 10 parameter combinations:")
print(f"  {'HalfLife':>10} {'MktWeight':>10} {'RMSE':>8} {'Seasons'}")
top10 = results_df.nsmallest(10, "rmse")
for _, r in top10.iterrows():
    hl_l = str(r["half_life"]) if r["half_life"] else "∞"
    print(f"  {hl_l:>10} {r['market_weight']:>10.3f} {r['rmse']:>8.4f}  {r['seasons']}")

# ── Best params ───────────────────────────────────────────────────────────────
# Use the top-N samples to estimate robust optimal params
TOP_N = 20
top_n = results_df.nsmallest(TOP_N, "rmse")
best_hl = top_n["half_life"].mode()[0]
best_mw = round(top_n["market_weight"].mean(), 2)

print(f"\n  Robust estimate from top {TOP_N} samples:")
print(f"  Most common half_life : {best_hl}")
print(f"  Mean market_weight    : {best_mw:.3f}")
print(f"  (These are the recommended operational hyperparameters)")

# ── Correlation ───────────────────────────────────────────────────────────────
hl_numeric = results_df["half_life"].fillna(9999).astype(float)
corr_hl = np.corrcoef(hl_numeric, results_df["rmse"])[0,1]
corr_mw = np.corrcoef(results_df["market_weight"], results_df["rmse"])[0,1]
print(f"\n  Correlation with RMSE:")
print(f"  half_life      : {corr_hl:+.3f}  "
      f"({'longer decay → better' if corr_hl < 0 else 'shorter decay → better'})")
print(f"  market_weight  : {corr_mw:+.3f}  "
      f"({'higher mw → better' if corr_mw < 0 else 'lower mw → better'})")

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
results_df.to_csv("data/monte_carlo_results.csv", index=False)
pd.DataFrame([{"half_life": best_hl, "market_weight": best_mw,
               "method": "monte_carlo_top20"}]).to_csv(
    "data/mc_best_params.csv", index=False)

print(f"\nSaved full results → data/monte_carlo_results.csv")
print(f"Saved best params  → data/mc_best_params.csv")