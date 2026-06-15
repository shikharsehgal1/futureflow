"""
In-tournament momentum analysis for a World Cup model.

Question: how much do a team's EARLIER games in the same World Cup predict their
LATER games in that same tournament, beyond pre-tournament strength?

Approach:
  - Filter results.csv to FIFA World Cup finals (exclude qualification).
  - Build a pre-tournament Elo from ALL international matches played BEFORE each
    World Cup starts. This is the "pre-tournament strength" control.
  - For each team within each WC, order matches chronologically.
    perf metric = goal difference (team perspective).
  - Pool (perf_so_far = mean GD over games 1..k, pretournament_rating_diff,
    perf_next = GD in game k+1) across all teams/tournaments.
  - Regress perf_next ~ perf_so_far + pretournament_rating_diff.
    Coefficient on perf_so_far = momentum weight.
  - Report decay by k, group->KO correlation, lag-1 autocorrelation.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "data/intl_results.csv"
OUT = "data/momentum_weights.json"

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------
df = pd.read_csv(RESULTS, parse_dates=["date"])
df = df.dropna(subset=["home_score", "away_score"])
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)

# WC finals only (exclude qualification, and the non-FIFA "Viva"/"CONIFA")
wc = df[df["tournament"] == "FIFA World Cup"].copy()
wc = wc.sort_values("date").reset_index(drop=True)
wc["year"] = wc["date"].dt.year

print(f"WC-finals matches: {len(wc)}  across years: {sorted(wc['year'].unique())}")

# ---------------------------------------------------------------------------
# Pre-tournament Elo from ALL matches before each WC (point-in-time)
# ---------------------------------------------------------------------------
allm = df.sort_values("date").reset_index(drop=True)
K = 30.0
HFA = 65.0  # home-field advantage in Elo points

def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

# We need each team's Elo as of the START of each WC. Replay all history,
# snapshotting ratings on the day each WC's first match occurs.
wc_start_dates = wc.groupby("year")["date"].min().to_dict()
snapshot_dates = sorted(wc_start_dates.values())

elo = {}
def get(t):
    return elo.get(t, 1500.0)

snapshots = {}  # date -> dict copy
si = 0
for _, r in allm.iterrows():
    d = r["date"]
    # take snapshots for any WC start date we've reached
    while si < len(snapshot_dates) and d >= snapshot_dates[si]:
        snapshots[snapshot_dates[si]] = dict(elo)
        si += 1
    h, a = r["home_team"], r["away_team"]
    hs, as_ = r["home_score"], r["away_score"]
    neutral = str(r["neutral"]).upper() == "TRUE"
    ra, rb = get(h), get(a)
    eh = expected(ra + (0 if neutral else HFA), rb)
    if hs > as_:
        sh = 1.0
    elif hs < as_:
        sh = 0.0
    else:
        sh = 0.5
    # goal-difference multiplier (mild)
    gd = abs(hs - as_)
    mult = 1.0 + 0.15 * max(gd - 1, 0)
    elo[h] = ra + K * mult * (sh - eh)
    elo[a] = rb + K * mult * ((1 - sh) - (1 - eh))
# any remaining snapshots
while si < len(snapshot_dates):
    snapshots[snapshot_dates[si]] = dict(elo)
    si += 1

def pretourn_rating(team, year):
    snap = snapshots[wc_start_dates[year]]
    return snap.get(team, 1500.0)

# ---------------------------------------------------------------------------
# Build per-team within-tournament match sequences
# ---------------------------------------------------------------------------
# Long format: one row per (team, match) with team perspective GD + opp.
rows = []
for _, r in wc.iterrows():
    rows.append(dict(year=r["year"], date=r["date"], team=r["home_team"],
                     opp=r["away_team"], gd=r["home_score"] - r["away_score"]))
    rows.append(dict(year=r["year"], date=r["date"], team=r["away_team"],
                     opp=r["home_team"], gd=r["away_score"] - r["home_score"]))
long = pd.DataFrame(rows).sort_values(["year", "team", "date"]).reset_index(drop=True)

# attach ratings
long["team_rating"] = long.apply(lambda x: pretourn_rating(x["team"], x["year"]), axis=1)
long["opp_rating"] = long.apply(lambda x: pretourn_rating(x["opp"], x["year"]), axis=1)
long["rating_diff"] = long["team_rating"] - long["opp_rating"]

# ---------------------------------------------------------------------------
# Pooled "running form" predicts next game
# ---------------------------------------------------------------------------
pool = []  # perf_next, perf_so_far, rating_diff_next, k (games already played)
for (year, team), g in long.groupby(["year", "team"]):
    g = g.sort_values("date").reset_index(drop=True)
    gds = g["gd"].values
    rds = g["rating_diff"].values
    n = len(g)
    for k in range(1, n):           # predict game k+1 (index k) from games 1..k
        perf_so_far = gds[:k].mean()
        perf_next = gds[k]
        pool.append(dict(year=year, team=team, k=k,
                         perf_so_far=perf_so_far, perf_next=perf_next,
                         rating_diff=rds[k]))
P = pd.DataFrame(pool)
print(f"\nPooled prediction samples (team, next-game pairs): {len(P)}")
print(P["k"].value_counts().sort_index().to_dict(), "<- count by #games already played")

def ols(y, X):
    """OLS with intercept. X is (n,p). Returns beta, se, t, p, r2."""
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    n, p = Xc.shape
    dof = n - p
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(Xc.T @ Xc)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    return beta, se, t, pval, r2, dof

y = P["perf_next"].values.astype(float)
X = P[["perf_so_far", "rating_diff"]].values.astype(float)
beta, se, t, pval, r2, dof = ols(y, X)
momentum_coef = float(beta[1])      # coef on perf_so_far
rating_coef = float(beta[2])

print("\n=== Regression: perf_next ~ perf_so_far + rating_diff ===")
print(f"  intercept      = {beta[0]:+.4f}  (se {se[0]:.4f})")
print(f"  perf_so_far    = {beta[1]:+.4f}  (se {se[1]:.4f}, t {t[1]:+.2f}, p {pval[1]:.4f})  <-- MOMENTUM")
print(f"  rating_diff    = {beta[2]:+.6f} (se {se[2]:.6f}, t {t[2]:+.2f}, p {pval[2]:.4f})")
print(f"  R^2={r2:.4f}  n={len(y)}  dof={dof}")

# rating_diff per 100 Elo (interpretability)
print(f"  -> rating effect per 100 Elo = {rating_coef*100:+.3f} goals")

# ---------------------------------------------------------------------------
# Decay: does momentum coef change with number of games already played (k)?
# ---------------------------------------------------------------------------
decay = {}
for k in sorted(P["k"].unique()):
    sub = P[P["k"] == k]
    if len(sub) < 8:
        decay[int(k)] = {"coef": None, "n": int(len(sub)), "note": "too few"}
        continue
    bb, ss, tt, pp, rr, dd = ols(sub["perf_next"].values.astype(float),
                                 sub[["perf_so_far", "rating_diff"]].values.astype(float))
    decay[int(k)] = {"coef": round(float(bb[1]), 4), "se": round(float(ss[1]), 4),
                     "p": round(float(pp[1]), 4), "n": int(len(sub))}
print("\n=== Momentum coef by #games already played (decay) ===")
for k, v in decay.items():
    print(f"  after {k} game(s): coef={v.get('coef')} n={v['n']} p={v.get('p')}")

# ---------------------------------------------------------------------------
# Group-stage GD vs Knockout GD correlation
# ---------------------------------------------------------------------------
# Heuristic split: in modern format group stage = first 3 matches per team.
# Use first 3 as "group", remainder as "knockout".
gk = []
for (year, team), g in long.groupby(["year", "team"]):
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) <= 3:
        continue  # no knockout games
    group_gd = g["gd"].values[:3].mean()
    ko_gd = g["gd"].values[3:].mean()
    gk.append((group_gd, ko_gd, year, team))
gk = pd.DataFrame(gk, columns=["group_gd", "ko_gd", "year", "team"])
if len(gk) >= 3:
    r_gk, p_gk = stats.pearsonr(gk["group_gd"], gk["ko_gd"])
else:
    r_gk, p_gk = float("nan"), float("nan")
print(f"\n=== Group-stage GD vs Knockout GD ===")
print(f"  corr = {r_gk:+.4f}  p = {p_gk:.4f}  n = {len(gk)} teams (reached KO)")

# ---------------------------------------------------------------------------
# Game-to-game (lag-1) autocorrelation of within-tournament GD
# ---------------------------------------------------------------------------
pairs_cur, pairs_next = [], []
for (year, team), g in long.groupby(["year", "team"]):
    gds = g.sort_values("date")["gd"].values
    for i in range(len(gds) - 1):
        pairs_cur.append(gds[i])
        pairs_next.append(gds[i + 1])
r_ac, p_ac = stats.pearsonr(pairs_cur, pairs_next)
print(f"\n=== Lag-1 autocorrelation of GD within tournaments ===")
print(f"  corr = {r_ac:+.4f}  p = {p_ac:.4f}  n = {len(pairs_cur)} consecutive pairs")

# Also: residual autocorrelation after removing rating_diff (cleaner momentum)
# regress gd on rating_diff, take residuals, then lag-1 corr
gg = long.copy()
b2, _, _, _, _, _ = ols(gg["gd"].values.astype(float),
                        gg[["rating_diff"]].values.astype(float))
gg["resid"] = gg["gd"] - (b2[0] + b2[1] * gg["rating_diff"])
rc, rn = [], []
for (year, team), g in gg.groupby(["year", "team"]):
    rv = g.sort_values("date")["resid"].values
    for i in range(len(rv) - 1):
        rc.append(rv[i]); rn.append(rv[i + 1])
r_resid, p_resid = stats.pearsonr(rc, rn)
print(f"  residual (rating-adjusted) lag-1 corr = {r_resid:+.4f}  p = {p_resid:.4f}  n={len(rc)}")

# ---------------------------------------------------------------------------
# Translate momentum_coef into a per-game rating nudge for a goals model
# ---------------------------------------------------------------------------
# momentum_coef says: each +1 goal of mean in-cup GD -> +momentum_coef goals
# expected in the next game (beyond pre-tournament strength). A team that has
# over/under-performed pre-tournament expectation by `e` goals per game should
# have its effective goals-rating nudged by ~ momentum_coef * e per game.
# We express per_game_rating_update as the momentum_coef itself (goals of
# next-game GD per goal of running over/underperformance). To use *over-
# performance vs expectation*, define e = perf_so_far - expected_gd. Here we
# report the slope; recommended practical nudge is momentum_coef per goal of
# average over/underperformance, capped.
per_game_rating_update = round(momentum_coef, 3)

significant = bool(pval[1] < 0.05)

notes = (
    f"Momentum coef = {momentum_coef:+.3f} goals of next-game GD per +1 goal of "
    f"average in-tournament GD so far, controlling for pre-tournament Elo. "
    f"n={len(y)} (team, next-game) pairs from {len(wc)} WC-finals matches across "
    f"{len(wc['year'].unique())} tournaments. "
    f"{'STATISTICALLY SIGNIFICANT at p<0.05' if significant else 'NOT significant at p<0.05'} "
    f"(p={pval[1]:.3f}). Pre-tournament rating_diff is the dominant predictor "
    f"({rating_coef*100:+.3f} goals per 100 Elo). "
    f"Group->KO GD corr = {r_gk:+.3f} (p={p_gk:.3f}, n={len(gk)}); "
    f"raw lag-1 GD autocorr = {r_ac:+.3f} (p={p_ac:.3f}); "
    f"rating-adjusted lag-1 autocorr = {r_resid:+.3f} (p={p_resid:.3f}). "
    f"This is a small-sample question: in-cup form carries a modest signal that "
    f"is weak-to-moderate and largely subsumed by pre-tournament strength. "
    f"Weight group results lightly when forecasting knockout games."
)

# decay summary as a compact dict
decay_summary = {str(k): v.get("coef") for k, v in decay.items()}

out = {
    "momentum_coef": round(momentum_coef, 4),
    "momentum_coef_se": round(float(se[1]), 4),
    "momentum_coef_p": round(float(pval[1]), 4),
    "momentum_significant": significant,
    "decay": decay_summary,
    "decay_detail": decay,
    "rating_coef_per100elo_goals": round(rating_coef * 100, 4),
    "per_game_rating_update": per_game_rating_update,
    "correlation_group_to_ko": round(float(r_gk), 4),
    "correlation_group_to_ko_p": round(float(p_gk), 4),
    "lag1_autocorr_gd": round(float(r_ac), 4),
    "lag1_autocorr_gd_p": round(float(p_ac), 4),
    "lag1_autocorr_rating_adjusted": round(float(r_resid), 4),
    "lag1_autocorr_rating_adjusted_p": round(float(p_resid), 4),
    "regression_r2": round(float(r2), 4),
    "n_samples": int(len(y)),
    "n_wc_matches": int(len(wc)),
    "n_tournaments": int(len(wc["year"].unique())),
    "n_ko_teams": int(len(gk)),
    "notes": notes,
}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {OUT}")
print(json.dumps(out, indent=2))
