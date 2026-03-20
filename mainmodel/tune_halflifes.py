"""
tune_halflifes.py — Grid search over MARKET_HALF_LIFE and FORM_HALF_LIFE

Optimisations vs naive version:
  1. Design matrix X built ONCE per (div, season) — not rebuilt each gameday
  2. Grid search parallelised across CPU cores via multiprocessing
  3. Decay vectors precomputed per gameday — no per-row Python loops

Holds HALF_LIFE=90 and MARKET_WEIGHT=0.95 fixed (already tuned in part6.py).

Usage:
    python3.10 tune_halflifes.py
    python3.10 tune_halflifes.py --div E0 D1   # faster: specific leagues only
    python3.10 tune_halflifes.py --workers 4   # override CPU count
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import warnings
import multiprocessing as mp
from itertools import product
warnings.filterwarnings("ignore")

# ── CONFIG (fixed) ────────────────────────────────────────────────────────────
HALF_LIFE     = 90
MARKET_WEIGHT = 0.95
FORM_WEIGHT   = 0.15
MIN_GAMES     = 10
HA_PRIOR      = 0.3

# ── SEARCH GRID ───────────────────────────────────────────────────────────────
MARKET_HALF_LIVES = [7, 14, 21, 30, 45, 60, 90]
FORM_HALF_LIVES   = [7, 14, 21, 30, 45, 60, 90]

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/big5_with_probs.csv", encoding="utf-8", low_memory=False)
df["Date"]     = pd.to_datetime(df["Date"])
df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]


def preprocess_group(group):
    """Build design matrix X once per (div, season)."""
    group  = group.sort_values("Date").reset_index(drop=True)
    teams  = sorted(set(group["HomeTeam"]) | set(group["AwayTeam"]))
    t_idx  = {t: i for i, t in enumerate(teams)}
    n      = len(teams)
    m      = len(group)

    # Design matrix (m, n+1) — last col = Home_Adv
    X = np.zeros((m, n + 1), dtype=np.float64)
    X[:, -1] = 1.0
    for k, (ht, at) in enumerate(zip(group["HomeTeam"], group["AwayTeam"])):
        X[k, t_idx[ht]] =  1.0
        X[k, t_idx[at]] = -1.0

    y_actual = group["GoalDiff"].values.astype(np.float64)
    y_impl   = group["mu_mkt"].values.astype(np.float64)
    dates    = pd.to_datetime(group["Date"]).values.astype("datetime64[D]")
    gamedays = np.unique(dates)

    # Prior matrix (n+1 rows): team rows + HFA row
    Xp = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n):
        Xp[i, i] = 1.0   # team shrinkage: [team=+1, HFA=0] = 0
    Xp[n, -1] = 1.0      # HFA anchor:    [HFA=+1] = HA_PRIOR

    yp = np.concatenate([np.zeros(n), [HA_PRIOR]])

    return dict(n=n, X=X, y_actual=y_actual, y_impl=y_impl,
                dates=dates, gamedays=gamedays, Xp=Xp, yp=yp, m=m)


def evaluate_combo(args):
    """Worker: evaluate one (market_hl, form_hl) combo across all groups."""
    preprocessed, market_hl, form_hl = args
    all_actual, all_pred = [], []

    for pg in preprocessed:
        n   = pg["n"]
        X   = pg["X"]
        ya  = pg["y_actual"]
        yi  = pg["y_impl"]
        Xp  = pg["Xp"]
        yp  = pg["yp"]
        dates    = pg["dates"]
        gamedays = pg["gamedays"]

        for gd in gamedays:
            mask_tr  = dates < gd
            mask_pr  = dates == gd
            n_train  = mask_tr.sum()
            if n_train < MIN_GAMES:
                continue

            X_tr  = X[mask_tr]
            ya_tr = ya[mask_tr]
            yi_tr = yi[mask_tr]
            X_pr  = X[mask_pr]
            ya_pr = ya[mask_pr]

            # Days ago (vectorised)
            days = (gd - dates[mask_tr]).astype(float)

            # Decay vectors
            dw      = 0.5 ** (days / HALF_LIFE)
            dw_mkt  = 0.5 ** (days / market_hl)
            dw_form = 0.5 ** (days / form_hl)

            # Adaptive prior weights
            gpt    = max(n_train * 2 / n, 0.1)
            tpw    = 3.0 / gpt
            hpw    = tpw * 5/3
            wp     = np.concatenate([np.full(n, tpw), [hpw]])

            # Stacked system
            X_g = np.vstack([X_tr, X_tr])
            y_g = np.concatenate([ya_tr, yi_tr])
            w_g = np.concatenate([dw * (1-MARKET_WEIGHT), dw_mkt * MARKET_WEIGHT])

            Xa  = np.vstack([X_g, Xp])
            ya_ = np.concatenate([y_g, yp])
            wa  = np.concatenate([w_g, wp])
            sq  = np.sqrt(np.clip(wa, 0, None))

            beta, _, _, _ = np.linalg.lstsq(Xa * sq[:,None], ya_ * sq, rcond=None)
            if not np.all(np.isfinite(beta)):
                continue

            # Form blend
            if FORM_WEIGHT > 0 and n_train >= MIN_GAMES * 2:
                w_f = np.concatenate([dw_form*(1-MARKET_WEIGHT), dw_form*MARKET_WEIGHT])
                wa_f = np.concatenate([w_f, wp])
                sq_f = np.sqrt(np.clip(wa_f, 0, None))
                beta_f, _, _, _ = np.linalg.lstsq(Xa * sq_f[:,None], ya_ * sq_f, rcond=None)
                if np.all(np.isfinite(beta_f)):
                    beta = (1 - FORM_WEIGHT) * beta + FORM_WEIGHT * beta_f

            preds = (X_pr @ beta).tolist()
            all_actual.extend(ya_pr.tolist())
            all_pred.extend(preds)

    if not all_actual:
        return market_hl, form_hl, np.nan, np.nan

    actual = np.array(all_actual)
    pred   = np.array(all_pred)
    rmse   = float(np.sqrt(np.mean((actual - pred)**2)))
    ss_res = np.sum((actual - pred)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return market_hl, form_hl, rmse, r2


def main(divs=None, n_workers=None):
    tune_df = df[~df["season"].isin(["2024-25", "2025-26"])].copy()
    if divs:
        tune_df = tune_df[tune_df["Div"].isin(divs)]
        print(f"Restricting to: {divs}")

    print("Pre-building design matrices...", flush=True)
    preprocessed = [preprocess_group(g)
                    for _, g in tune_df.groupby(["Div", "season"])]
    print(f"  {len(preprocessed)} league-seasons ready\n")

    combos  = list(product(MARKET_HALF_LIVES, FORM_HALF_LIVES))
    workers = n_workers or max(1, mp.cpu_count() - 1)

    print(f"Grid search: {len(combos)} combos  |  {workers} workers")
    print(f"Fixed: HALF_LIFE={HALF_LIFE}d  MARKET_WEIGHT={MARKET_WEIGHT}  "
          f"FORM_WEIGHT={FORM_WEIGHT}\n")
    print(f"  {'MktHL':>6} {'FormHL':>7} {'RMSE':>8} {'R²':>8}")
    print(f"  {'-'*37}")

    work = [(preprocessed, mhl, fhl) for mhl, fhl in combos]

    best_rmse   = np.inf
    best_params = {}
    rows        = []

    with mp.Pool(workers) as pool:
        for market_hl, form_hl, rmse, r2 in pool.imap(evaluate_combo, work):
            marker = " ◀" if rmse < best_rmse else ""
            print(f"  {market_hl:>6}d {form_hl:>6}d  {rmse:>8.4f}  {r2:>8.4f}{marker}",
                  flush=True)
            rows.append({"market_hl": market_hl, "form_hl": form_hl,
                         "rmse": rmse, "r2": r2})
            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = {"market_hl": market_hl, "form_hl": form_hl}

    print(f"\n  Best: MARKET_HALF_LIFE={best_params['market_hl']}d  "
          f"FORM_HALF_LIFE={best_params['form_hl']}d  "
          f"(RMSE={best_rmse:.4f})")

    out = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    out.to_csv("data/tune_halflifes.csv", index=False)
    print(f"  Saved → data/tune_halflifes.csv")

    print(f"\n  Top 10 combos:")
    print(f"  {'MktHL':>6} {'FormHL':>7} {'RMSE':>8} {'R²':>8}")
    for _, r in out.head(10).iterrows():
        print(f"  {int(r.market_hl):>6}d {int(r.form_hl):>6}d  "
              f"{r.rmse:>8.4f}  {r.r2:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--div",     nargs="*", default=None)
    parser.add_argument("--workers", type=int,  default=None)
    args = parser.parse_args()
    main(args.div, args.workers)