import requests
from io import StringIO
import pandas as pd

# One recent season per league to inspect available columns
URLS = {
    "EPL":       "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "Championship": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "Bundesliga":"https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "La Liga":   "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "Serie A":   "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "Ligue 1":   "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
}

all_cols = {}

for league, url in URLS.items():
    print(f"\n{'='*55}")
    print(f"  {league}")
    print(f"{'='*55}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
        df = df.dropna(how="all")
        n = len(df)
        all_cols[league] = set(df.columns)
        print(f"  {n} rows, {len(df.columns)} columns\n")
        for col in df.columns:
            null_pct = df[col].isna().mean()
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else "N/A"
            print(f"  {col:<14} {null_pct:5.1%} null   sample={sample}")
    except Exception as e:
        print(f"  ERROR: {e}")

# Show which columns are common across all leagues vs league-specific
print(f"\n{'='*55}")
print("  COLUMN AVAILABILITY ACROSS LEAGUES")
print(f"{'='*55}")
if all_cols:
    common = set.intersection(*all_cols.values())
    print(f"\n  In ALL leagues ({len(common)} cols):")
    for c in sorted(common):
        print(f"    {c}")

    for league, cols in all_cols.items():
        unique = cols - common
        if unique:
            print(f"\n  Only in {league} ({len(unique)} cols):")
            for c in sorted(unique):
                print(f"    {c}")