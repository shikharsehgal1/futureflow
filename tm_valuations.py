"""
tm_valuations.py — Preseason squad valuation scraper from Transfermarkt

Runs once at the start of each season (August). Fetches squad market values
for all Big 5 teams and uses the year-on-year change to modulate carry weights
in build_preseason_priors().

Integration with part7_ratings.py:
  - Writes data/tm_valuations.csv with columns:
      Team, Div, season, squad_value_m, squad_value_prev_m,
      value_change_pct, carry_weight_adjusted
  - part7_ratings.py reads this in build_preseason_priors() and uses
    carry_weight_adjusted instead of the flat CARRY_WEIGHT_TIER1/2

Carry weight adjustment model:
  Base carry = CARRY_WEIGHT_TIER1 (0.50) for top divisions
  If squad value increased by 30%+ → carry up to 0.65 (major investment)
  If squad value decreased by 30%+ → carry down to 0.35 (major departures)
  Smooth interpolation in between.

Usage:
    python3.10 tm_valuations.py                    # all Big 5 teams
    python3.10 tm_valuations.py --div E0 D1        # specific leagues
    python3.10 tm_valuations.py --season 2025-26   # specific season
"""

import os, time, re, argparse, json
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup

try:
    import cloudscraper
except ImportError:
    print("Install: pip3.10 install cloudscraper beautifulsoup4 lxml")
    raise

# ── CONFIG ────────────────────────────────────────────────────────────────────
VALUATIONS_PATH = "data/tm_valuations.csv"
CACHE_PATH      = "data/tm_valuations_cache.json"
SLEEP           = 3.5    # TM is aggressive — be polite
TM_BASE         = "https://www.transfermarkt.com"
CURRENT_SEASON  = "2025-26"

# Carry weight bounds (modulated by squad value change)
CARRY_TIER1_BASE = 0.50
CARRY_TIER1_MAX  = 0.68   # major investment
CARRY_TIER1_MIN  = 0.32   # major departures
CARRY_TIER2_BASE = 0.10
CARRY_TIER2_MAX  = 0.20
CARRY_TIER2_MIN  = 0.04

TIER1_DIVS = {"E0", "D1", "SP1", "I1", "F1"}

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.transfermarkt.com/",
}

# ── TEAM SLUGS (same as tm_injuries.py) ──────────────────────────────────────
TEAM_SLUGS = {
    # Premier League
    "Arsenal":         ("arsenal-fc",               "11"),
    "Aston Villa":     ("aston-villa",              "405"),
    "Bournemouth":     ("afc-bournemouth",          "989"),
    "Brentford":       ("brentford-fc",           "7941"),
    "Brighton":        ("brighton-amp-hove-albion", "1237"),
    "Burnley":         ("fc-burnley",              "1132"),
    "Chelsea":         ("fc-chelsea",               "631"),
    "Crystal Palace":  ("crystal-palace",           "873"),
    "Everton":         ("fc-everton",               "29"),
    "Fulham":          ("fc-fulham",                "931"),
    "Ipswich":         ("ipswich-town",             "677"),
    "Leeds":           ("leeds-united",             "399"),
    "Liverpool":       ("fc-liverpool",             "31"),
    "Man City":        ("manchester-city",          "281"),
    "Man United":      ("manchester-united",        "985"),
    "Newcastle":       ("newcastle-united",         "762"),
    "Nott'm Forest":   ("nottingham-forest",        "703"),
    "Sunderland":      ("sunderland-afc",           "289"),
    "Tottenham":       ("tottenham-hotspur",        "148"),
    "West Ham":        ("west-ham-united",          "379"),
    "Wolves":          ("wolverhampton-wanderers",  "543"),
    # Bundesliga
    "Augsburg":        ("fc-augsburg",              "167"),
    "Bayern Munich":   ("fc-bayern-munchen",        "27"),
    "Bochum":          ("vfl-bochum",               "80"),
    "Dortmund":        ("borussia-dortmund",        "16"),
    "Ein Frankfurt":   ("eintracht-frankfurt",      "24"),
    "FC Koln":         ("1-fc-koln",                "3"),
    "Freiburg":        ("sc-freiburg",              "7"),
    "Heidenheim":      ("1-fc-heidenheim-1846",  "2036"),
    "Hoffenheim":      ("tsg-hoffenheim",          "533"),
    "Holstein Kiel":   ("holstein-kiel",           "2192"),
    "Leverkusen":      ("bayer-04-leverkusen",      "15"),
    "M'gladbach":      ("borussia-monchengladbach", "18"),
    "Mainz":           ("1-fsv-mainz-05",           "27"),
    "RB Leipzig":      ("rb-leipzig",           "23826"),
    "St Pauli":        ("fc-st-pauli",              "35"),
    "Stuttgart":       ("vfb-stuttgart",            "79"),
    "Union Berlin":    ("1-fc-union-berlin",        "89"),
    "Werder Bremen":   ("sv-werder-bremen",         "86"),
    "Wolfsburg":       ("vfl-wolfsburg",            "82"),
    # La Liga
    "Ath Bilbao":      ("athletic-club",            "621"),
    "Ath Madrid":      ("atletico-de-madrid",       "13"),
    "Barcelona":       ("fc-barcelona",            "131"),
    "Betis":           ("real-betis-balompie",      "150"),
    "Celta":           ("rc-celta-de-vigo",         "940"),
    "Espanol":         ("rcd-espanyol-barcelona",   "714"),
    "Getafe":          ("getafe-cf",               "3709"),
    "Girona":          ("girona-fc",              "12321"),
    "Levante":         ("levante-ud",             "3769"),
    "Mallorca":        ("real-mallorca",            "237"),
    "Osasuna":         ("ca-osasuna",               "331"),
    "Real Madrid":     ("real-madrid",             "418"),
    "Sevilla":         ("fc-sevilla",              "368"),
    "Sociedad":        ("real-sociedad",            "681"),
    "Valencia":        ("fc-valencia",            "1049"),
    "Vallecano":       ("rayo-vallecano",           "366"),
    "Villarreal":      ("villarreal-cf",           "1050"),
    "Alaves":          ("deportivo-alaves",        "1108"),
    "Elche":           ("elche-cf",                "969"),
    "Oviedo":          ("real-oviedo",            "3628"),
    # Serie A
    "Atalanta":        ("atalanta-bc",              "800"),
    "Bologna":         ("fc-bologna",             "1025"),
    "Como":            ("como-1907",              "6195"),
    "Fiorentina":      ("acf-fiorentina",           "430"),
    "Genoa":           ("genoa-cfc",               "252"),
    "Inter":           ("inter-mailand",            "46"),
    "Juventus":        ("juventus-turin",           "506"),
    "Lazio":           ("ss-lazio",                "398"),
    "Lecce":           ("us-lecce",               "4827"),
    "Milan":           ("ac-mailand",               "5"),
    "Napoli":          ("ssc-napoli",             "6195"),
    "Parma":           ("parma-calcio-1913",        "130"),
    "Roma":            ("as-rom",                   "12"),
    "Torino":          ("fc-torino",               "416"),
    "Udinese":         ("udinese-calcio",           "410"),
    "Verona":          ("hellas-verona",            "276"),
    "Cagliari":        ("cagliari-calcio",         "1390"),
    # Ligue 1
    "Angers":          ("sco-angers",             "3497"),
    "Auxerre":         ("aj-auxerre",              "671"),
    "Brest":           ("stade-brestois-29",       "3911"),
    "Le Havre":        ("le-havre-ac",            "3891"),
    "Lens":            ("rc-lens",                 "826"),
    "Lille":           ("losc-lille",             "1082"),
    "Lorient":         ("fc-lorient",             "3913"),
    "Lyon":            ("olympique-lyon",          "1041"),
    "Marseille":       ("olympique-marseille",      "244"),
    "Monaco":          ("as-monaco",               "162"),
    "Montpellier":     ("montpellier-hsc",          "969"),
    "Nantes":          ("fc-nantes",               "995"),
    "Nice":            ("ogc-nice",                "417"),
    "Paris FC":        ("paris-fc",              "30773"),
    "Paris SG":        ("paris-saint-germain",     "583"),
    "Reims":           ("stade-de-reims",          "1421"),
    "Rennes":          ("stade-rennais-fc",         "273"),
    "Strasbourg":      ("rc-strasbourg-alsace",     "667"),
    "Toulouse":        ("toulouse-fc",              "415"),
}

# Which division each team plays in currently
TEAM_DIV = {}  # populated from fixtures/league data at runtime


def make_scraper():
    s = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False}
    )
    s.headers.update(HEADERS)
    return s


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def parse_value(text):
    text = str(text).strip().replace("€","").replace(",","").replace(" ","")
    if not text or text == "-":
        return 0.0
    try:
        if "bn" in text.lower():
            return float(re.sub(r"[^\d.]","",text)) * 1000
        if "m" in text.lower():
            return float(re.sub(r"[^\d.]","",text))
        if "k" in text.lower():
            return float(re.sub(r"[^\d.]","",text)) / 1000
        return float(re.sub(r"[^\d.]","",text)) / 1_000_000
    except (ValueError, TypeError):
        return 0.0


def get_squad_value(scraper, team_name, cache, force=False):
    """Fetch current squad market value from TM team page."""
    slug_data = TEAM_SLUGS.get(team_name)
    if not slug_data:
        return None

    slug, team_id = slug_data
    cache_key = f"squad_val_{team_name}"
    now = datetime.now()

    if not force and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached["timestamp"])
        if (now - cached_time).total_seconds() < 86400 * 7:
            return cached["data"]

    url = f"{TM_BASE}/{slug}/startseite/verein/{team_id}"
    print(f"  {team_name:<25}", end="", flush=True)

    try:
        r = scraper.get(url, timeout=15)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            return None
        soup = BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"Error: {e}")
        return None

    # Try multiple selectors TM uses for squad value
    val = None
    for selector in [
        "span.data-header__market-value-number",
        "a.data-header__market-value",
        "div.data-header__details span",
    ]:
        el = soup.select_one(selector)
        if el:
            v = parse_value(el.get_text(strip=True))
            if v > 0:
                val = v
                break

    if val:
        print(f"€{val:.0f}m")
        cache[cache_key] = {"timestamp": now.isoformat(), "data": val}
        save_cache(cache)
    else:
        print("value not found")

    time.sleep(SLEEP)
    return val


def adjusted_carry_weight(value_m, prev_value_m, div):
    """
    Compute carry weight adjusted for squad value change.

    If squad value grew significantly → team strengthened → higher carry
    If squad value dropped significantly → team weakened → lower carry

    Returns float carry weight.
    """
    is_tier1  = div in TIER1_DIVS
    base      = CARRY_TIER1_BASE  if is_tier1 else CARRY_TIER2_BASE
    max_carry = CARRY_TIER1_MAX   if is_tier1 else CARRY_TIER2_MAX
    min_carry = CARRY_TIER1_MIN   if is_tier1 else CARRY_TIER2_MIN

    if not value_m or not prev_value_m or prev_value_m == 0:
        return base

    pct_change = (value_m - prev_value_m) / prev_value_m

    # Smooth sigmoid-like mapping: ±50% change → ±0.18 adjustment on carry
    # pct_change = +0.30 → +0.10 carry adjustment
    # pct_change = -0.30 → -0.10 carry adjustment
    adjustment = np.clip(pct_change * 0.33, -0.18, 0.18)
    carry      = np.clip(base + adjustment, min_carry, max_carry)

    return round(carry, 3)


def main(divs=None, season=CURRENT_SEASON, force=False):
    os.makedirs("data", exist_ok=True)

    # Get team→div mapping from league data
    try:
        lg = pd.read_csv("data/big5_with_probs.csv", low_memory=False)
        lg = lg[lg["season"] == season]
        team_div = {}
        for _, row in lg.iterrows():
            team_div[row["HomeTeam"]] = row["Div"]
            team_div[row["AwayTeam"]] = row["Div"]
    except Exception:
        team_div = {}
        print("Could not load league data — div info unavailable")

    # Load previous season valuations if available
    prev_vals = {}
    if os.path.exists(VALUATIONS_PATH):
        prev_df = pd.read_csv(VALUATIONS_PATH)
        if "season" in prev_df.columns:
            # Get most recent season before current
            prev_seasons = sorted(prev_df["season"].unique())
            if season in prev_seasons:
                idx = prev_seasons.index(season)
                if idx > 0:
                    prev_s = prev_seasons[idx - 1]
                    for _, row in prev_df[prev_df["season"] == prev_s].iterrows():
                        prev_vals[row["Team"]] = row["squad_value_m"]

    # Filter teams
    teams_to_fetch = [t for t in TEAM_SLUGS if not divs or team_div.get(t) in divs]
    print(f"Fetching squad valuations for {len(teams_to_fetch)} teams...\n")

    scraper = make_scraper()
    cache   = load_cache()
    rows    = []

    for team in sorted(teams_to_fetch):
        div       = team_div.get(team, "?")
        value     = get_squad_value(scraper, team, cache, force)
        prev_val  = prev_vals.get(team)

        if value is None:
            continue

        carry = adjusted_carry_weight(value, prev_val, div)

        rows.append({
            "Team":                 team,
            "Div":                  div,
            "season":               season,
            "squad_value_m":        round(value, 1),
            "squad_value_prev_m":   round(prev_val, 1) if prev_val else None,
            "value_change_pct":     round((value - prev_val) / prev_val * 100, 1)
                                    if prev_val else None,
            "carry_weight_adjusted": carry,
        })

    if not rows:
        print("No data fetched.")
        return

    # Load existing and append/update
    out_df = pd.DataFrame(rows)
    if os.path.exists(VALUATIONS_PATH):
        existing = pd.read_csv(VALUATIONS_PATH)
        # Remove current season rows and replace
        existing = existing[existing["season"] != season]
        out_df   = pd.concat([existing, out_df], ignore_index=True)

    out_df.to_csv(VALUATIONS_PATH, index=False)
    print(f"\nSaved {len(rows)} teams → {VALUATIONS_PATH}")

    # Print summary
    df = pd.DataFrame(rows).sort_values("squad_value_m", ascending=False)
    print(f"\n  {'Team':<22} {'Div':>4} {'Value':>8} {'Prev':>8} "
          f"{'Chg%':>7} {'Carry':>7}")
    print(f"  {'-'*60}")
    for _, r in df.iterrows():
        chg = f"{r['value_change_pct']:>+6.1f}%" if pd.notna(r.get("value_change_pct")) else "    N/A"
        prev = f"€{r['squad_value_prev_m']:.0f}m" if pd.notna(r.get("squad_value_prev_m")) else "    N/A"
        print(f"  {r['Team']:<22} {r['Div']:>4} "
              f"€{r['squad_value_m']:>6.0f}m {prev:>8} "
              f"{chg:>7} {r['carry_weight_adjusted']:>7.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--div",    nargs="*", default=None)
    parser.add_argument("--season", default=CURRENT_SEASON)
    parser.add_argument("--force",  action="store_true",
                        help="Bypass cache — re-fetch all")
    args = parser.parse_args()
    main(args.div, args.season, args.force)