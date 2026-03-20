"""
tm_injuries.py — Weekly injury/suspension scraper from Transfermarkt

Runs Friday alongside fetch_fixtures.py. For each upcoming fixture,
scrapes current injuries and suspensions for both teams and computes
a mu adjustment based on the value and position of missing players.

Integration with part7_predict.py:
  - Writes data/tm_injuries.csv with columns:
      HomeTeam, AwayTeam, Date, injury_adj_home, injury_adj_away, injury_notes
  - part7_predict.py reads this and adjusts mu before Poisson calculation:
      mu_adjusted = mu + injury_adj_home - injury_adj_away

Adjustment model:
  Missing player value as % of squad affects mu proportionally.
  Calibrated so losing a €100m player (e.g. top striker) from a €700m squad
  (14% of squad value) → ~-0.20 goals adjustment.
  Position weighting: ST/CF/CAM most impactful, CB/GK least.

Usage:
    python3.10 tm_injuries.py                    # all fixtures in data/fixtures.csv
    python3.10 tm_injuries.py --team Arsenal     # single team
    python3.10 tm_injuries.py --div E0 D1        # specific leagues
"""

import os, time, re, argparse, json
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup

try:
    import cloudscraper
except ImportError:
    print("Install cloudscraper: pip3.10 install cloudscraper")
    raise

# ── CONFIG ────────────────────────────────────────────────────────────────────
FIXTURES_PATH  = "data/fixtures.csv"
INJURIES_PATH  = "data/tm_injuries.csv"
CACHE_PATH     = "data/tm_injuries_cache.json"
SLEEP          = 3.0    # TM rate limiting — be polite
TM_BASE        = "https://www.transfermarkt.com"

# Max mu adjustment from injuries (prevents extreme values)
MAX_INJURY_ADJ = 0.40   # goals

# Position weights — how much each position contributes to goal scoring
# Based on research: attackers most impactful, defenders least
POSITION_WEIGHTS = {
    "Centre-Forward":      1.0,
    "Second Striker":      0.9,
    "Attacking Midfield":  0.85,
    "Left Winger":         0.8,
    "Right Winger":        0.8,
    "Central Midfield":    0.5,
    "Defensive Midfield":  0.35,
    "Left-Back":           0.25,
    "Right-Back":          0.25,
    "Centre-Back":         0.20,
    "Goalkeeper":          0.15,
}
DEFAULT_POSITION_WEIGHT = 0.5

# ── TEAM SLUGS ────────────────────────────────────────────────────────────────
# Format: our_name -> (tm_slug, tm_id)
TEAM_SLUGS = {
    # Premier League
    "Arsenal":         ("fc-arsenal",               "11"),
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
    # Championship
    "Birmingham":      ("birmingham-city",          "103"),
    "Blackburn":       ("blackburn-rovers",         "164"),
    "Bristol City":    ("bristol-city",             "986"),
    "Cardiff":         ("cardiff-city",            "1032"),
    "Charlton":        ("charlton-athletic",        "525"),
    "Coventry":        ("coventry-city",            "322"),
    "Derby":           ("derby-county",             "211"),
    "Hull":            ("hull-city",                "176"),
    "Leicester":       ("leicester-city",           "1003"),
    "Middlesbrough":   ("middlesbrough-fc",         "174"),
    "Millwall":        ("millwall-fc",              "590"),
    "Norwich":         ("norwich-city",             "97"),
    "Oxford":          ("oxford-united",            "440"),
    "Portsmouth":      ("portsmouth-fc",            "136"),
    "Preston":         ("preston-north-end",        "270"),
    "QPR":             ("queens-park-rangers",      "241"),
    "Sheffield United":("sheffield-united",         "350"),
    "Sheffield Weds":  ("sheffield-wednesday",      "558"),
    "Southampton":     ("southampton-fc",           "180"),
    "Stoke":           ("stoke-city",               "739"),
    "Swansea":         ("swansea-city",             "450"),
    "Watford":         ("fc-watford",               "1010"),
    "West Brom":       ("west-bromwich-albion",     "188"),
    "Wrexham":         ("wrexham-afc",             "4580"),
    # Bundesliga
    "Augsburg":        ("fc-augsburg",              "167"),
    "Bayern Munich":   ("fc-bayern-munchen",        "27"),
    "Bochum":          ("vfl-bochum",               "80"),
    "Dortmund":        ("borussia-dortmund",        "16"),
    "Ein Frankfurt":   ("eintracht-frankfurt",      "24"),
    "FC Koln":         ("1-fc-koln",                "3"),
    "Fortuna Dusseldorf":("fortuna-dusseldorf",     "39"),
    "Freiburg":        ("sc-freiburg",              "7"),
    "Hamburg":         ("hamburger-sv",             "41"),
    "Heidenheim":      ("1-fc-heidenheim-1846",  "2036"),
    "Hertha":          ("hertha-bsc",               "86"),
    "Hoffenheim":      ("tsg-hoffenheim",          "533"),
    "Holstein Kiel":   ("holstein-kiel",           "2192"),
    "Leverkusen":      ("bayer-04-leverkusen",      "15"),
    "M'gladbach":      ("borussia-monchengladbach", "18"),
    "Mainz":           ("1-fsv-mainz-05",           "39"),
    "RB Leipzig":      ("rb-leipzig",           "23826"),
    "St Pauli":        ("fc-st-pauli",              "35"),
    "Stuttgart":       ("vfb-stuttgart",            "79"),
    "Union Berlin":    ("1-fc-union-berlin",       "89"),
    "Werder Bremen":   ("sv-werder-bremen",         "86"),
    "Wolfsburg":       ("vfl-wolfsburg",            "82"),
    # La Liga
    "Ath Bilbao":      ("athletic-club",            "621"),
    "Ath Madrid":      ("atletico-de-madrid",       "13"),
    "Barcelona":       ("fc-barcelona",            "131"),
    "Betis":           ("real-betis-balompie",      "150"),
    "Celta":           ("rc-celta-de-vigo",         "940"),
    "Espanol":         ("rcd-espanyol-barcelona",   "714"),
    "Getafe":          ("getafe-cf",                "3709"),
    "Girona":          ("girona-fc",               "12321"),
    "Levante":         ("levante-ud",              "3769"),
    "Mallorca":        ("real-mallorca",            "237"),
    "Osasuna":         ("ca-osasuna",               "331"),
    "Real Madrid":     ("real-madrid",             "418"),
    "Sevilla":         ("fc-sevilla",              "368"),
    "Sociedad":        ("real-sociedad",            "681"),
    "Valencia":        ("fc-valencia",             "1049"),
    "Vallecano":       ("rayo-vallecano",           "366"),
    "Villarreal":      ("villarreal-cf",            "1050"),
    "Alaves":          ("deportivo-alaves",         "1108"),
    "Elche":           ("elche-cf",                "969"),
    "Oviedo":          ("real-oviedo",             "3628"),
    # Serie A
    "Atalanta":        ("atalanta-bc",              "800"),
    "Bologna":         ("fc-bologna",              "1025"),
    "Como":            ("como-1907",              "6195"),
    "Fiorentina":      ("acf-fiorentina",           "430"),
    "Genoa":           ("genoa-cfc",               "252"),
    "Inter":           ("inter-mailand",            "46"),
    "Juventus":        ("juventus-turin",           "506"),
    "Lazio":           ("ss-lazio",                "398"),
    "Lecce":           ("us-lecce",                "4827"),
    "Milan":           ("ac-mailand",              "5"),
    "Napoli":          ("ssc-napoli",             "6195"),
    "Parma":           ("parma-calcio-1913",       "130"),
    "Roma":            ("as-rom",                  "12"),
    "Sassuolo":        ("us-sassuolo",            "6574"),
    "Torino":          ("fc-torino",               "416"),
    "Udinese":         ("udinese-calcio",          "410"),
    "Verona":          ("hellas-verona",           "276"),
    "Cagliari":        ("cagliari-calcio",         "1390"),
    # Ligue 1
    "Angers":          ("sco-angers",             "3497"),
    "Auxerre":         ("aj-auxerre",              "671"),
    "Brest":           ("stade-brestois-29",       "3911"),
    "Le Havre":        ("le-havre-ac",            "3891"),
    "Lens":            ("rc-lens",                "826"),
    "Lille":           ("losc-lille",             "1082"),
    "Lorient":         ("fc-lorient",             "3913"),
    "Lyon":            ("olympique-lyon",          "1041"),
    "Marseille":       ("olympique-marseille",     "244"),
    "Metz":            ("fc-metz",                "347"),
    "Monaco":          ("as-monaco",              "162"),
    "Montpellier":     ("montpellier-hsc",        "969"),
    "Nantes":          ("fc-nantes",              "995"),
    "Nice":            ("ogc-nice",               "417"),
    "Paris FC":        ("paris-fc",              "30773"),
    "Paris SG":        ("paris-saint-germain",    "583"),
    "Reims":           ("stade-de-reims",         "1421"),
    "Rennes":          ("stade-rennais-fc",       "273"),
    "Strasbourg":      ("rc-strasbourg-alsace",   "667"),
    "Toulouse":        ("toulouse-fc",            "415"),
}

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.transfermarkt.com/",
}


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
    """Parse '€150m', '€4.50m', '€800k' → float millions EUR."""
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


def get_team_injuries(scraper, team_name, cache, force=False):
    """
    Fetch current injuries + suspensions for a team from TM.
    Returns list of {name, position, value_m, reason, return_date}.
    Caches results for 24 hours.

    TM page structure (as of 2026):
      URL:  /{slug}/sperrenundverletzungen/verein/{id}
      Rows: inline-table with name/position, age, injury type, since, return, games missed
      Note: market value NOT on this page — fetched separately via get_squad_values()
    """
    slug_data = TEAM_SLUGS.get(team_name)
    if not slug_data:
        return None

    slug, team_id = slug_data
    cache_key = f"injuries_{team_name}"
    now = datetime.now()

    # Use cache if fresh (< 24 hours)
    if not force and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached["timestamp"])
        if (now - cached_time).total_seconds() < 86400:
            return cached["data"]

    url = f"{TM_BASE}/{slug}/sperrenundverletzungen/verein/{team_id}/plus/1"
    print(f"    Fetching injuries: {team_name}...", flush=True)

    try:
        r = scraper.get(url, timeout=15)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code} for {team_name}")
            return None
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"    Error: {e}")
        return None

    injured = []
    for row in soup.select("table.items tbody tr"):
        # Name — inside inline-table
        name_td = row.select_one("td.hauptlink a")
        if not name_td:
            continue
        name = name_td.get_text(strip=True)

        # Position — second tr inside inline-table
        pos_td = row.select_one("table.inline-table tr:last-child td")
        pos    = pos_td.get_text(strip=True) if pos_td else "Unknown"

        # All tds as flat list — structure:
        # [name+pos, blank, name, pos, age, injury, since, return, games, apps, value]
        all_tds = row.select("td")
        td_texts = [td.get_text(strip=True) for td in all_tds]

        # Value — last td contains €XXm
        value = 0.0
        for t in reversed(td_texts):
            if "€" in t:
                value = parse_value(t)
                break

        # Injury reason — td.links
        inj_td = row.select_one("td.links")
        reason = inj_td.get_text(strip=True) if inj_td else "Unknown"

        # Dates — td.zentriert containing dd/mm/yyyy
        since_date = return_date = ""
        for td in row.select("td.zentriert"):
            t = td.get_text(strip=True)
            if re.match(r'\d{2}/\d{2}/\d{4}', t):
                if not since_date:
                    since_date = t
                else:
                    return_date = t

        injured.append({
            "name":        name,
            "position":    pos,
            "value_m":     value,
            "reason":      reason,
            "since_date":  since_date,
            "return_date": return_date,
        })

    time.sleep(SLEEP)

    # Deduplicate by name — TM inline-table renders some rows twice
    seen  = {}
    dedup = []
    for p in injured:
        if p["name"] not in seen:
            seen[p["name"]] = True
            dedup.append(p)
    injured = dedup

    # Cache result
    cache[cache_key] = {"timestamp": now.isoformat(), "data": injured}
    save_cache(cache)


    return injured


def get_squad_values(scraper, team_name, cache, force=False):
    """
    Fetch player-level market values from TM squad page.
    Returns dict: {player_name: value_m}
    """
    slug_data = TEAM_SLUGS.get(team_name)
    if not slug_data:
        return {}

    slug, team_id = slug_data
    cache_key = f"squad_vals_{team_name}"
    now = datetime.now()

    if not force and cache_key in cache:
        cached = cache[cache_key]
        if (now - datetime.fromisoformat(cached["timestamp"])).total_seconds() < 86400 * 7:
            return cached["data"]

    url = f"{TM_BASE}/{slug}/kader/verein/{team_id}/saison_id/2025/plus/1"
    print(f"    Fetching squad values: {team_name}...", flush=True)

    try:
        r = scraper.get(url, timeout=15)
        if r.status_code != 200:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"    Error: {e}")
        return {}

    values = {}
    total  = 0.0
    for row in soup.select("table.items tbody tr.odd, table.items tbody tr.even"):
        name_td = row.select_one("td.hauptlink a")
        val_td  = row.select_one("td.rechts.hauptlink")
        if name_td and val_td:
            name  = name_td.get_text(strip=True)
            value = parse_value(val_td.get_text(strip=True))
            values[name] = value
            total += value

    time.sleep(SLEEP)
    result = {"players": values, "total": total}
    cache[cache_key] = {"timestamp": now.isoformat(), "data": result}
    save_cache(cache)
    return result


def merge_squad_values(injured, squad_data):
    """
    Merge market values into injured player list.
    Matches by name. Falls back to position-average if no match.
    """
    if not squad_data:
        return injured

    player_vals = squad_data.get("players", {})
    total       = squad_data.get("total", 0.0)
    n_players   = max(len(player_vals), 20)
    avg_val     = total / n_players if n_players else 5.0

    for p in injured:
        # Try exact match first, then partial
        val = player_vals.get(p["name"])
        if val is None:
            for k, v in player_vals.items():
                if p["name"].lower() in k.lower() or k.lower() in p["name"].lower():
                    val = v
                    break
        p["value_m"] = val if val is not None else avg_val

    return injured


    slug_data = TEAM_SLUGS.get(team_name)
    if not slug_data:
        return None

    slug, team_id = slug_data
    cache_key = f"squad_{team_name}"
    now = datetime.now()

    if not force and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached["timestamp"])
        if (now - cached_time).total_seconds() < 86400 * 7:  # 1 week cache
            return cached["data"]

    url = f"{TM_BASE}/{slug}/startseite/verein/{team_id}"
    print(f"    Fetching squad value: {team_name}...", flush=True)

    try:
        r = scraper.get(url, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"    Error: {e}")
        return None

    # TM shows total squad value in header
    val_span = soup.find("span", {"class": "data-header__market-value-number"})
    if not val_span:
        val_span = soup.find("a", {"class": lambda c: c and "data-header" in str(c)})
    if val_span:
        val = parse_value(val_span.get_text(strip=True))
        if val > 0:
            cache[cache_key] = {"timestamp": now.isoformat(), "data": val}
            save_cache(cache)
            time.sleep(SLEEP)
            return val

    time.sleep(SLEEP)
    return None


def compute_injury_adjustment(injured_players, squad_value_m):
    """
    Compute mu adjustment from missing players.

    Logic:
    - Each player's impact = (value / squad_value) * position_weight
    - Summed across all injured players
    - Scaled to goals: 0.14 (14% squad value) → -0.20 goals adjustment
      (e.g. top striker at €100m from €700m squad)
    - Capped at MAX_INJURY_ADJ

    Returns negative float (missing players reduce team strength).
    """
    if not injured_players or not squad_value_m or squad_value_m == 0:
        return 0.0

    total_impact = 0.0
    for p in injured_players:
        val_share = p["value_m"] / squad_value_m
        pos_w     = POSITION_WEIGHTS.get(p["position"], DEFAULT_POSITION_WEIGHT)
        total_impact += val_share * pos_w

    # Scale: 0.14 * 1.0 (top striker at 14% squad) → 0.20 goals
    adj = total_impact * (0.20 / 0.14)
    return -round(min(adj, MAX_INJURY_ADJ), 3)


def analyse_fixture(home, away, scraper, cache):
    """Analyse injuries for one fixture. Returns adjustment dict."""
    result = {
        "HomeTeam":         home,
        "AwayTeam":         away,
        "injury_adj_home":  0.0,
        "injury_adj_away":  0.0,
        "home_injured":     [],
        "away_injured":     [],
        "injury_notes":     "",
    }

    for team, side in [(home, "home"), (away, "away")]:
        if team not in TEAM_SLUGS:
            result["injury_notes"] += f"{team}: not in TM slugs. "
            continue

        squad_data = get_squad_values(scraper, team, cache)
        injured    = get_team_injuries(scraper, team, cache)

        if injured is None:
            result["injury_notes"] += f"{team}: fetch failed. "
            continue

        # Merge market values into injured players
        if squad_data:
            injured = merge_squad_values(injured, squad_data)
            squad_total = squad_data.get("total", 300.0)
        else:
            squad_total = 300.0  # fallback

        adj = compute_injury_adjustment(injured, squad_total)
        result[f"injury_adj_{side}"]  = adj
        result[f"{side}_injured"]     = injured

        if injured:
            names = ", ".join(
                f"{p['name']} ({p['position']}, €{p['value_m']:.1f}m)"
                for p in sorted(injured, key=lambda x: -x["value_m"])[:3]
            )
            result["injury_notes"] += f"{team} missing: {names}. "

    # Net adjustment: positive = home benefits, negative = away benefits
    result["injury_adj_net"] = round(
        result["injury_adj_home"] - result["injury_adj_away"], 3
    )

    return result


def print_injury_report(results):
    print(f"\n{'='*70}")
    print(f"  INJURY REPORT")
    print(f"{'='*70}")
    print(f"  {'Home':<22} {'Away':<22} {'Adj_H':>7} {'Adj_A':>7} {'Net':>7}")
    print(f"  {'-'*65}")

    for r in results:
        adj_h = r["injury_adj_home"]
        adj_a = r["injury_adj_away"]
        net   = r["injury_adj_net"]
        flag  = " ◀" if abs(net) >= 0.10 else ""
        print(f"  {r['HomeTeam']:<22} {r['AwayTeam']:<22} "
              f"{adj_h:>+7.3f} {adj_a:>+7.3f} {net:>+7.3f}{flag}")

    alerts = [r for r in results if abs(r["injury_adj_net"]) >= 0.10]
    if alerts:
        print(f"\n  NOTABLE INJURY IMPACTS (|net| ≥ 0.10 goals):")
        for r in alerts:
            print(f"\n  {r['HomeTeam']} vs {r['AwayTeam']}")
            print(f"    Net mu adjustment: {r['injury_adj_net']:+.3f} goals")
            if r["home_injured"]:
                print(f"    {r['HomeTeam']} missing:")
                for p in sorted(r["home_injured"], key=lambda x: -x["value_m"])[:5]:
                    pw = POSITION_WEIGHTS.get(p["position"], DEFAULT_POSITION_WEIGHT)
                    print(f"      {p['name']:<25} {p['position']:<22} "
                          f"€{p['value_m']:>5.1f}m  pos_w={pw:.2f}")
            if r["away_injured"]:
                print(f"    {r['AwayTeam']} missing:")
                for p in sorted(r["away_injured"], key=lambda x: -x["value_m"])[:5]:
                    pw = POSITION_WEIGHTS.get(p["position"], DEFAULT_POSITION_WEIGHT)
                    print(f"      {p['name']:<25} {p['position']:<22} "
                          f"€{p['value_m']:>5.1f}m  pos_w={pw:.2f}")


def main(divs=None, team=None, force=False):
    os.makedirs("data", exist_ok=True)

    # Load fixtures
    if not os.path.exists(FIXTURES_PATH):
        print(f"No fixtures found at {FIXTURES_PATH}. Run fetch_fixtures.py first.")
        return

    fixtures = pd.read_csv(FIXTURES_PATH)
    if divs:
        fixtures = fixtures[fixtures["Div"].isin(divs)]
    if team:
        fixtures = fixtures[
            (fixtures["HomeTeam"] == team) | (fixtures["AwayTeam"] == team)
        ]

    if len(fixtures) == 0:
        print("No fixtures to analyse.")
        return

    print(f"Analysing injuries for {len(fixtures)} fixtures...\n")
    scraper = make_scraper()
    cache   = load_cache()
    results = []

    for _, fix in fixtures.iterrows():
        print(f"  {fix['HomeTeam']} vs {fix['AwayTeam']}")
        r = analyse_fixture(fix["HomeTeam"], fix["AwayTeam"], scraper, cache)
        r["Date"] = fix["Date"]
        r["Div"]  = fix.get("Div", "")
        results.append(r)

    # Save CSV (without nested lists)
    out_rows = []
    for r in results:
        out_rows.append({
            "Date":             r["Date"],
            "Div":              r["Div"],
            "HomeTeam":         r["HomeTeam"],
            "AwayTeam":         r["AwayTeam"],
            "injury_adj_home":  r["injury_adj_home"],
            "injury_adj_away":  r["injury_adj_away"],
            "injury_adj_net":   r["injury_adj_net"],
            "injury_notes":     r["injury_notes"],
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(INJURIES_PATH, index=False)
    print(f"\nSaved → {INJURIES_PATH}")

    print_injury_report(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--div",   nargs="*", default=None)
    parser.add_argument("--team",  default=None)
    parser.add_argument("--force", action="store_true",
                        help="Bypass cache — re-fetch all")
    args = parser.parse_args()
    main(args.div, args.team, args.force)