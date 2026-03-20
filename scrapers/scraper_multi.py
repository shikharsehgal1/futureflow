"""
Scrapes football-data.co.uk for the big 5 European leagues, top 2 divisions each.
Saves one combined CSV: big5_football.csv

Division codes:
  England  : E0 (Premier League), E1 (Championship)
  Germany  : D1 (Bundesliga),     D2 (2. Bundesliga)
  Spain    : SP1 (La Liga),        SP2 (Segunda)
  Italy    : I1 (Serie A),         I2 (Serie B)
  France   : F1 (Ligue 1),         F2 (Ligue 2)

Usage:
  python scraper_multi.py                        # defaults: 2015-16 onward
  python scraper_multi.py --from-season 2010-11
  python scraper_multi.py --output mydata/big5.csv
"""

import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import re
import os

BASE_URL = "https://www.football-data.co.uk"

COUNTRY_PAGES = {
    "england": "englandm.php",
    "germany": "germanym.php",
    "spain":   "spainm.php",
    "italy":   "italym.php",
    "france":  "francem.php",
}

# Top 2 division codes per country
DIVISIONS = {
    "england": ["E0", "E1"],
    "germany": ["D1", "D2"],
    "spain":   ["SP1", "SP2"],
    "italy":   ["I1", "I2"],
    "france":  ["F1", "F2"],
}

KEEP_COLS = [
    # Match facts
    "Date", "Time", "Div", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",

    # Closing 1X2 odds — primary for model evaluation
    "B365CH", "B365CD", "B365CA",
    "PSCH",   "PSCD",   "PSCA",
    "MaxCH",  "MaxCD",  "MaxCA",
    "AvgCH",  "AvgCD",  "AvgCA",

    # Opening 1X2 odds — useful for tracking line movement
    "B365H",  "B365D",  "B365A",
    "PSH",    "PSD",    "PSA",

    # Over/under 2.5 closing — for total goals model later
    "B365C>2.5", "B365C<2.5",
    "MaxC>2.5",  "MaxC<2.5",

    # Asian handicap closing — for margin model later
    "AHCh",
    "B365CAHH", "B365CAHA",
    "MaxCAHH",  "MaxCAHA",
]

def parse_season_from_url(url):
    # Matches both:
    #   mmz4281/2526/E0.csv  (current format)
    #   mmz4281/E0/2526.csv  (old format, just in case)
    match = re.search(r"/(\d{4})/[A-Z]", url) or re.search(r"/(\d{4})\.csv", url)
    if not match:
        return "unknown"
    code = match.group(1)
    start, end = code[:2], code[2:]
    start_year = int(f"{'19' if int(start) >= 90 else '20'}{start}")
    return f"{start_year}-{end}"

def season_start_year(label):
    return int(label.split("-")[0])

def get_csv_links(page_url):
    resp = requests.get(page_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".csv"):
            full = href if href.startswith("http") else f"{BASE_URL}/{href.lstrip('/')}"
            links.append(full)
    return links

def parse_div_from_url(url):
    """Extract division code from URL, e.g. mmz4281/2526/E0.csv -> E0"""
    match = re.search(r"/\d{4}/([^/]+)\.csv", url)
    return match.group(1) if match else None

def fetch_csv(url, season, country):
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        df["season"]  = season
        df["country"] = country
        # If Div column missing or all null, infer from URL
        if "Div" not in df.columns or df["Div"].isna().all():
            df["Div"] = parse_div_from_url(url)
        df.dropna(how="all", inplace=True)
        return df
    except Exception as e:
        print(f"    WARNING: skipped {url} — {e}")
        return None

def parse_dates(series):
    # Try DD/MM/YY first (most common), then DD/MM/YYYY for older seasons
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        still_bad = parsed.isna().sum()
        if still_bad == 0:
            return parsed
        # Fill in what we got, retry remaining with next format
        if fmt == "%d/%m/%y":
            fallback_mask = parsed.isna()
            parsed2 = pd.to_datetime(series[fallback_mask], format="%d/%m/%Y", errors="coerce")
            parsed = parsed.copy()
            parsed[fallback_mask] = parsed2
            still_bad = parsed.isna().sum()
            if still_bad > 0:
                print(f"  WARNING: {still_bad} rows have unparseable dates (set to NaT)")
            return parsed
    return parsed

def scrape_country(country, min_start, keep_divs):
    page_url = f"{BASE_URL}/{COUNTRY_PAGES[country]}"
    print(f"\n{'='*55}")
    print(f"  {country.upper()}")
    print(f"{'='*55}")
    print(f"  Fetching links from {page_url} ...")

    links = get_csv_links(page_url)

    filtered = [
        l for l in links
        if parse_season_from_url(l) != "unknown"
        and season_start_year(parse_season_from_url(l)) >= min_start
    ]
    print(f"  Found {len(filtered)} season files from {min_start} onward.")

    frames = []
    for url in filtered:
        season = parse_season_from_url(url)
        df = fetch_csv(url, season, country)
        if df is None or df.empty:
            continue
        # Filter to desired divisions if Div column exists
        if "Div" in df.columns and keep_divs:
            df = df[df["Div"].isin(keep_divs)]
        if not df.empty:
            frames.append(df)
            divs = df["Div"].unique().tolist() if "Div" in df.columns else ["?"]
            print(f"    {season}: {len(df):4d} rows  divisions={divs}")

    return frames

def main(from_season, output_path):
    min_start = season_start_year(from_season)

    print(f"From season : {from_season}")
    print(f"Output      : {output_path}")

    all_frames = []
    for country, divs in DIVISIONS.items():
        frames = scrape_country(country, min_start, divs)
        all_frames.extend(frames)

    if not all_frames:
        raise RuntimeError("No data downloaded — check connection.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined["Date"] = parse_dates(combined["Date"])

    numeric_cols = [c for c in KEEP_COLS if c not in ("Date", "Div", "HomeTeam", "AwayTeam")]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined.sort_values(["country", "season", "Date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"  Total rows    : {len(combined):,}")
    print(f"  Date range    : {combined['Date'].min().date()} – {combined['Date'].max().date()}")
    print(f"  B365 coverage : {combined['B365H'].notna().mean():.1%}")
    print()
    summary = (
        combined.groupby(["country", "Div"])
        .agg(games=("FTHG", "count"), seasons=("season", "nunique"))
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"\n  Saved to {output_path}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape big 5 European leagues (top 2 divisions each)"
    )
    parser.add_argument(
        "--from-season", default="2015-16", metavar="YYYY-YY",
        help="Earliest season to include (default: 2015-16)",
    )
    parser.add_argument(
        "--output", default="big5_football.csv",
        help="Output CSV path (default: big5_football.csv)",
    )
    args = parser.parse_args()
    main(from_season=args.from_season, output_path=args.output)