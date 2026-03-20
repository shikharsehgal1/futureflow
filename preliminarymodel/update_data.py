"""
update_data.py — Part 7 data updater

Scrapes the latest results from football-data.co.uk for all 10 divisions,
finds rows not yet in big5_with_probs.csv, applies the Part 5 probit model
to compute mu_mkt, and appends them to the master CSV.

Usage:
    python update_data.py                  # scrape + auto-merge
    python update_data.py --manual         # also prompt to add manual rows
    python update_data.py --dry-run        # show new rows without saving

Manual fallback:
    If football-data.co.uk hasn't updated yet, create a file
    data/manual_results.csv with columns:
        Date, Div, HomeTeam, AwayTeam, HomeGoals, AwayGoals,
        odds_H, odds_D, odds_A
    and run: python update_data.py --manual
"""

import argparse
import pickle
import warnings
import requests
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/big5_with_probs.csv"
REG_PATH     = "data/probit_reg.pkl"
DIV_COLS_PATH= "data/div_cols.pkl"
MG_PATH      = "data/mean_goals.pkl"
MANUAL_PATH  = "data/manual_results.csv"

# Current seasons to scrape
CURRENT_SEASONS = {
    "2025-26": "2526",
    "2024-25": "2425",
}

DIVISION_URLS = {
    "E0":  "england/E0",
    "E1":  "england/E1",
    "D1":  "germany/D1",
    "D2":  "germany/D2",
    "SP1": "spain/SP1",
    "SP2": "spain/SP2",
    "I1":  "italy/I1",
    "I2":  "italy/I2",
    "F1":  "france/F1",
    "F2":  "france/F2",
}

COUNTRY_MAP = {
    "E0": "england", "E1": "england",
    "D1": "germany", "D2": "germany",
    "SP1": "spain",  "SP2": "spain",
    "I1": "italy",   "I2": "italy",
    "F1": "france",  "F2": "france",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281"

KEEP_COLS = [
    "Date", "Time", "Div", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
    "B365CH", "B365CD", "B365CA", "PSCH", "PSCD", "PSCA",
    "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA",
    "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
    "B365C>2.5", "B365C<2.5", "MaxC>2.5", "MaxC<2.5",
    "AHCh", "B365CAHH", "B365CAHA", "MaxCAHH", "MaxCAHA",
]

# ── LOAD SAVED MODELS ─────────────────────────────────────────────────────────
def load_models():
    with open(REG_PATH, "rb") as f:
        reg = pickle.load(f)
    with open(DIV_COLS_PATH, "rb") as f:
        div_cols = pickle.load(f)
    with open(MG_PATH, "rb") as f:
        mg = pickle.load(f)
    return reg, div_cols, mg["home"], mg["away"]


# ── SCRAPE ────────────────────────────────────────────────────────────────────
def scrape_division(div, season_code):
    url = f"{BASE_URL}/{season_code}/{div}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
        df.dropna(how="all", inplace=True)
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        if "Div" not in df.columns or df["Div"].isna().all():
            df["Div"] = div
        return df
    except Exception as e:
        print(f"    WARNING: could not fetch {url} — {e}")
        return None


def scrape_all():
    frames = []
    for season_label, season_code in CURRENT_SEASONS.items():
        for div, _ in DIVISION_URLS.items():
            print(f"  Scraping {div} {season_label}...")
            df = scrape_division(div, season_code)
            if df is None or df.empty:
                continue
            df["season"]  = season_label
            df["country"] = COUNTRY_MAP[div]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── PROCESS NEW ROWS ──────────────────────────────────────────────────────────
def parse_dates(series):
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.isna().sum() == 0:
            return parsed
        mask = parsed.isna()
        parsed2 = pd.to_datetime(series[mask], format="%d/%m/%Y", errors="coerce")
        parsed = parsed.copy()
        parsed[mask] = parsed2
        return parsed


def compute_odds_features(df):
    """Add normalised market probabilities and probit features."""
    df = df.copy()

    for side, ps, bc, bo in [
        ("H", "PSCH", "B365CH", "B365H"),
        ("D", "PSCD", "B365CD", "B365D"),
        ("A", "PSCA", "B365CA", "B365A"),
    ]:
        df[f"odds_{side}"] = (
            df[ps].where(df[ps].notna() & (df[ps] > 1))
            .fillna(df[bc].where(df[bc].notna() & (df[bc] > 1)))
            .fillna(df[bo].where(df[bo].notna() & (df[bo] > 1)))
            if all(c in df.columns for c in [ps, bc, bo])
            else df[bo] if bo in df.columns else np.nan
        )

    df["ou_over"]  = df["B365C>2.5"] if "B365C>2.5" in df.columns else np.nan
    df["ou_under"] = df["B365C<2.5"] if "B365C<2.5" in df.columns else np.nan

    mask_odds = df["odds_H"].notna() & df["odds_D"].notna() & df["odds_A"].notna()
    df = df[mask_odds & (df["odds_H"] > 1) & (df["odds_D"] > 1) & (df["odds_A"] > 1)].copy()

    raw = np.column_stack([1/df["odds_H"], 1/df["odds_D"], 1/df["odds_A"]])
    df["raw_sum"]   = raw.sum(axis=1)
    df["overround"] = df["raw_sum"] - 1
    raw = raw / raw.sum(axis=1, keepdims=True)
    df["p_H_mkt"] = raw[:, 0]
    df["p_D_mkt"] = raw[:, 1]
    df["p_A_mkt"] = raw[:, 2]

    has_ou = df["ou_over"].notna() & df["ou_under"].notna()
    ou_raw = np.where(
        has_ou.values[:, None],
        np.column_stack([1/df["ou_over"].fillna(2).values,
                         1/df["ou_under"].fillna(2).values]),
        np.nan)
    ou_sum = np.nansum(ou_raw, axis=1, keepdims=True)
    ou_norm = np.where(ou_sum > 0, ou_raw / ou_sum, np.nan)
    df["p_over_mkt"]  = ou_norm[:, 0]
    df["p_under_mkt"] = ou_norm[:, 1]

    # Probit features
    def clip_p(p): return np.clip(p, 1e-6, 1-1e-6)
    for col, p in [("z_H","p_H_mkt"), ("z_A","p_A_mkt"), ("z_D","p_D_mkt")]:
        df[col] = norm.ppf(clip_p(df[p].values))

    df["strength_adj"]   = (df["p_H_mkt"] - df["p_A_mkt"]) / (1 - df["p_D_mkt"] + 1e-6)
    df["z_strength"]     = norm.ppf(clip_p((df["strength_adj"] + 1) / 2))
    df["log_odds_ratio"] = np.log(clip_p(df["p_H_mkt"]) / clip_p(df["p_A_mkt"]))
    df["z_D_sq"]         = df["z_D"] ** 2

    return df


def apply_probit(df, reg, div_cols, mean_home, mean_away):
    """Apply the saved Part 5 probit regression to get mu_mkt."""
    df = df.copy()

    div_dummies = pd.get_dummies(df["Div"], prefix="div", drop_first=True)
    for col in div_cols:
        if col not in div_dummies.columns:
            div_dummies[col] = 0
    div_dummies = div_dummies[div_cols]

    base_cols = ["z_H", "z_A", "z_D"]
    feat = np.column_stack([df[base_cols].values, div_dummies.values])

    df["mu_mkt"] = reg.predict(feat)

    # Model mu columns for completeness (use mu_B as mu_mkt source)
    df["mu_A"] = df["mu_mkt"]
    df["mu_B"] = df["mu_mkt"]

    return df


def process_scraped(raw_df, reg, div_cols, mean_home, mean_away):
    """Full pipeline: parse dates, clean, add features, apply probit."""
    df = raw_df.copy()
    df["Date"] = parse_dates(df["Date"])
    df = df.dropna(subset=["Date"])

    # Rename goals columns
    if "FTHG" in df.columns: df = df.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})
    df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]
    df["Result"]   = np.where(df["GoalDiff"] > 0, "H",
                     np.where(df["GoalDiff"] < 0, "A", "D"))

    df = compute_odds_features(df)
    df = apply_probit(df, reg, div_cols, mean_home, mean_away)

    return df


# ── FIND NEW ROWS ─────────────────────────────────────────────────────────────
def find_new_rows(existing_df, scraped_df):
    """Return rows in scraped_df not already in existing_df."""
    key = ["Date", "Div", "HomeTeam", "AwayTeam"]
    existing_keys = set(
        zip(existing_df["Date"].astype(str),
            existing_df["Div"],
            existing_df["HomeTeam"],
            existing_df["AwayTeam"])
    )
    scraped_df = scraped_df.copy()
    scraped_df["_key"] = list(zip(
        scraped_df["Date"].astype(str),
        scraped_df["Div"],
        scraped_df["HomeTeam"],
        scraped_df["AwayTeam"]
    ))
    new = scraped_df[~scraped_df["_key"].isin(existing_keys)].drop(columns=["_key"])
    return new


# ── MANUAL FALLBACK ───────────────────────────────────────────────────────────
def process_manual(reg, div_cols, mean_home, mean_away):
    """
    Load data/manual_results.csv and process it.
    Required columns: Date, Div, HomeTeam, AwayTeam, HomeGoals, AwayGoals,
                      odds_H, odds_D, odds_A
    Optional:         season (defaults to current)
    """
    try:
        manual = pd.read_csv(MANUAL_PATH)
    except FileNotFoundError:
        print(f"  No manual file found at {MANUAL_PATH} — skipping.")
        return pd.DataFrame()

    manual["Date"] = pd.to_datetime(manual["Date"])
    manual["GoalDiff"] = manual["HomeGoals"] - manual["AwayGoals"]
    manual["Result"]   = np.where(manual["GoalDiff"] > 0, "H",
                         np.where(manual["GoalDiff"] < 0, "A", "D"))

    if "season" not in manual.columns:
        manual["season"] = "2025-26"
    if "country" not in manual.columns:
        manual["country"] = manual["Div"].map(COUNTRY_MAP)

    # Normalise odds
    raw = np.column_stack([1/manual["odds_H"], 1/manual["odds_D"], 1/manual["odds_A"]])
    raw = raw / raw.sum(axis=1, keepdims=True)
    manual["p_H_mkt"] = raw[:, 0]
    manual["p_D_mkt"] = raw[:, 1]
    manual["p_A_mkt"] = raw[:, 2]
    manual["raw_sum"]   = 1/manual["odds_H"] + 1/manual["odds_D"] + 1/manual["odds_A"]
    manual["overround"] = manual["raw_sum"] - 1

    def clip_p(p): return np.clip(p, 1e-6, 1-1e-6)
    for col, p in [("z_H","p_H_mkt"), ("z_A","p_A_mkt"), ("z_D","p_D_mkt")]:
        manual[col] = norm.ppf(clip_p(manual[p].values))

    manual["strength_adj"]   = (manual["p_H_mkt"] - manual["p_A_mkt"]) / (1 - manual["p_D_mkt"] + 1e-6)
    manual["z_strength"]     = norm.ppf(clip_p((manual["strength_adj"] + 1) / 2))
    manual["log_odds_ratio"] = np.log(clip_p(manual["p_H_mkt"]) / clip_p(manual["p_A_mkt"]))
    manual["z_D_sq"]         = manual["z_D"] ** 2

    manual = apply_probit(manual, reg, div_cols, mean_home, mean_away)
    print(f"  Loaded {len(manual)} manual rows from {MANUAL_PATH}")
    return manual


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(dry_run=False, include_manual=False):
    print("Loading existing data...")
    existing = pd.read_csv(DATA_PATH)
    existing["Date"] = pd.to_datetime(existing["Date"])
    print(f"  Existing rows: {len(existing):,}  |  Latest date: {existing['Date'].max().date()}")

    print("\nLoading saved models...")
    reg, div_cols, mean_home, mean_away = load_models()

    print("\nScraping latest results...")
    raw_scraped = scrape_all()

    new_rows = pd.DataFrame()

    if not raw_scraped.empty:
        print("\nProcessing scraped data...")
        processed = process_scraped(raw_scraped, reg, div_cols, mean_home, mean_away)
        new_rows = find_new_rows(existing, processed)
        print(f"  New rows from scrape: {len(new_rows)}")

    if include_manual:
        print("\nChecking manual results file...")
        manual_rows = process_manual(reg, div_cols, mean_home, mean_away)
        if not manual_rows.empty:
            manual_new = find_new_rows(existing, manual_rows)
            print(f"  New rows from manual: {len(manual_new)}")
            new_rows = pd.concat([new_rows, manual_new], ignore_index=True)

    if new_rows.empty:
        print("\nNo new rows to add — data is already up to date.")
        return

    print(f"\nTotal new rows to add: {len(new_rows)}")
    if dry_run:
        print("\nDRY RUN — not saving. New rows preview:")
        print(new_rows[["Date","Div","HomeTeam","AwayTeam",
                         "HomeGoals","AwayGoals","mu_mkt"]].to_string(index=False))
        return

    # Align columns with existing before concat
    for col in existing.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan
    new_rows = new_rows[existing.columns]

    updated = pd.concat([existing, new_rows], ignore_index=True)
    updated = updated.sort_values(["country","season","Date"]).reset_index(drop=True)
    updated.to_csv(DATA_PATH, index=False)

    print(f"\nSaved {len(updated):,} rows → {DATA_PATH}")
    print(f"Added {len(new_rows)} new rows.")
    print(f"Latest date now: {pd.to_datetime(updated['Date']).max().date()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="Preview new rows without saving")
    parser.add_argument("--manual",   action="store_true", help="Also load data/manual_results.csv")
    args = parser.parse_args()
    main(dry_run=args.dry_run, include_manual=args.manual)
