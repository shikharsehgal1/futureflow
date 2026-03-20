import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import re
import os

BASE_URL = "https://www.football-data.co.uk"

COUNTRY_PAGES = {
    "england":     "englandm.php",
    "scotland":    "scotlandm.php",
    "germany":     "germanym.php",
    "italy":       "italym.php",
    "spain":       "spainm.php",
    "france":      "francem.php",
    "netherlands": "netherlandsm.php",
    "belgium":     "belgiumm.php",
    "portugal":    "portugalm.php",
    "turkey":      "turkeym.php",
    "greece":      "greecem.php",
}

KEEP_COLS = [
    "Date", "Div", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "HTHG", "HTAG",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
    "B365H", "B365A", "B365D",
    "PSH",  "PSA",  "PSD",
    "MaxH", "MaxD", "MaxA",
]

def parse_season_from_url(url):
    """
    Extract a human-readable season label from a CSV URL.
    E.g. '.../E0/9394.csv' -> '1993-94'
         '.../E0/0102.csv' -> '2001-02'
         '.../E0/2425.csv' -> '2024-25'
    """
    match = re.search(r"/(\d{4})\.csv", url)
    if not match:
        return "unknown"
    code  = match.group(1)
    start, end = code[:2], code[2:]
    start_year = int(f"{'19' if int(start) >= 90 else '20'}{start}")
    return f"{start_year}-{end}"

def season_start_year(label):
    """Return the start year integer from a label like '2010-11'."""
    return int(label.split("-")[0])

def get_csv_links(page_url):
    """Scrape all .csv hrefs from the page."""
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

def fetch_csv(url, season):
    """Download a single CSV, keep only desired columns, add season label."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        df["season"] = season
        df.dropna(how="all", inplace=True)
        return df
    except Exception as e:
        print(f"  WARNING: skipped {url} — {e}")
        return None

def parse_dates(series):
    """
    Try dayfirst (DD/MM/YY) first — standard for football-data.co.uk.
    Older seasons occasionally mix formats, so fall back to inferring format
    on any rows that failed the first pass.
    """
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    n_failed = parsed.isna().sum()
    if n_failed > 0:
        # retry failed rows with format inference
        fallback = pd.to_datetime(series[parsed.isna()], infer_datetime_format=True, errors="coerce")
        parsed = parsed.copy()
        parsed[parsed.isna()] = fallback
        still_bad = parsed.isna().sum()
        if still_bad > 0:
            print(f"  WARNING: {still_bad} rows have unparseable dates and will be NaT — check raw data")
    return parsed

def main(country, from_season, output_path, div=None):
    country = country.lower()
    if country not in COUNTRY_PAGES:
        raise ValueError(
            f"Unknown country '{country}'. "
            f"Available: {', '.join(sorted(COUNTRY_PAGES))}"
        )

    page_url  = f"{BASE_URL}/{COUNTRY_PAGES[country]}"
    min_start = season_start_year(from_season)

    print(f"Country  : {country}")
    print(f"From     : {from_season} (start year >= {min_start})")
    print(f"Division : {div or 'all'}")
    print(f"Output   : {output_path}")
    print(f"Fetching links from {page_url} ...\n")

    links = get_csv_links(page_url)
    print(f"Found {len(links)} CSV files total.")

    filtered = [
        l for l in links
        if parse_season_from_url(l) != "unknown"
        and season_start_year(parse_season_from_url(l)) >= min_start
    ]
    print(f"Keeping  {len(filtered)} files from {from_season} onward.\n")

    frames = []
    for url in filtered:
        season = parse_season_from_url(url)
        print(f"  Downloading {season}: {url}")
        df = fetch_csv(url, season)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded — check the URL or your connection.")

    combined = pd.concat(frames, ignore_index=True)

    # ── Date parsing with fallback ────────────────────────────────────────────
    combined["Date"] = parse_dates(combined["Date"])

    # ── Numeric coercion ──────────────────────────────────────────────────────
    numeric_cols = [c for c in KEEP_COLS if c not in ("Date", "Div", "HomeTeam", "AwayTeam")]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # ── Optional division filter ──────────────────────────────────────────────
    if div and "Div" in combined.columns:
        before = len(combined)
        combined = combined[combined["Div"] == div].reset_index(drop=True)
        print(f"\nFiltered to division '{div}': {before:,} -> {len(combined):,} rows")

    combined.sort_values(["season", "Date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nSaved {len(combined):,} rows to {output_path}")
    print(f"Seasons  : {combined['season'].nunique()} "
          f"({combined['season'].min()} – {combined['season'].max()})")
    if "Div" in combined.columns:
        print(f"Divisions: {sorted(combined['Div'].dropna().unique().tolist())}")
    print(f"Date range: {combined['Date'].min().date()} – {combined['Date'].max().date()}")
    odds_coverage = combined["B365H"].notna().mean()
    print(f"B365 odds coverage: {odds_coverage:.1%}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape football results from football-data.co.uk"
    )
    parser.add_argument(
        "--country", default="england",
        help=f"Options: {', '.join(sorted(COUNTRY_PAGES))}",
    )
    parser.add_argument(
        "--from-season", default="2010-11", metavar="YYYY-YY",
        help="Earliest season to include, e.g. 2010-11 (default: 2010-11)",
    )
    parser.add_argument(
        "--div", default=None,
        help="Filter to a single division code, e.g. E0 for EPL (default: all)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: <country>_football.csv)",
    )
    args = parser.parse_args()

    output = args.output or f"{args.country.lower()}_football.csv"
    main(
        country=args.country,
        from_season=args.from_season,
        output_path=output,
        div=args.div,
    )