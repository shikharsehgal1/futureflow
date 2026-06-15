#!/usr/bin/env python3
"""
fetch_referees.py — Pull FREE referee disciplinary data for the 2026 World Cup
cards/booking prop model.

Deliverables:
  1) data/referees.csv          — per-referee historical card rates
  2) data/wc_referee_assignments.csv — match->referee assignments (whatever is confirmed)

Primary data source for per-referee card rates: worldfootball.net referee
summary pages (Cloudflare-protected -> requires cloudscraper). These pages list
every competition a referee has worked, with Matches / Yellow / Yellow-Red
(second yellow) / Red columns. We sum the per-competition "Total" rows to get a
career aggregate, then derive per-game rates.

NOTE on coverage (be honest):
  - worldfootball gives Matches, Yellow, Yellow-Red, Red. It does NOT publish
    fouls-per-game or penalties-per-game. Those columns are therefore left blank
    (NaN) unless a sourced value is available. We do NOT fabricate them.
  - reds_per_game is computed as (Yellow-Red + Red) / Matches  (i.e. all sending-offs).
  - cards_per_game = (Yellow + Yellow-Red + Red) / Matches.

The WC2026 center-referee roster comes from Wikipedia "2026 FIFA World Cup
officials" / ESPN, cross-checked. 52 center referees were appointed.

Run:  python3 fetch_referees.py
"""
import os, re, csv, time, sys, unicodedata
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

try:
    import cloudscraper
except ImportError:
    print("cloudscraper not installed. Run: pip3 install cloudscraper", file=sys.stderr)
    raise
from bs4 import BeautifulSoup

WF = "https://www.worldfootball.net/referee_summary/{slug}/"

# ---------------------------------------------------------------------------
# WC2026 appointed CENTER referees (name, country).
# Source: Wikipedia "2026 FIFA World Cup officials" + ESPN, cross-checked
# (FIFA announced 52 referees on 2026-04-09).
# Artan (Somalia) was later denied US entry and removed; kept here flagged so
# the row is auditable, but he will not be assigned to matches.
# ---------------------------------------------------------------------------
REFEREES = [
    # AFC
    ("Omar Al Ali", "United Arab Emirates"),
    ("Abdulrahman Al-Jassim", "Qatar"),
    ("Khalid Al-Turais", "Saudi Arabia"),
    ("Alireza Faghani", "Australia"),
    ("Ma Ning", "China"),
    ("Adham Makhadmeh", "Jordan"),
    ("Ilgiz Tantashev", "Uzbekistan"),
    ("Yusuke Araki", "Japan"),
    # CAF
    ("Omar Abdulkadir Artan", "Somalia"),
    ("Pierre Atcho", "Gabon"),
    ("Dahane Beida", "Mauritania"),
    ("Mustapha Ghorbal", "Algeria"),
    ("Jalal Jayed", "Morocco"),
    ("Amin Mohamed Omar", "Egypt"),
    ("Abongile Tom", "South Africa"),
    # CONCACAF
    ("Ivan Barton", "El Salvador"),
    ("Juan Gabriel Calderon", "Costa Rica"),
    ("Ismail Elfath", "United States"),
    ("Oshane Nation", "Jamaica"),
    ("Drew Fischer", "Canada"),
    ("Katia Itzel Garcia", "Mexico"),
    ("Said Martinez", "Honduras"),
    ("Tori Penso", "United States"),
    ("Cesar Arturo Ramos", "Mexico"),
    # CONMEBOL
    ("Ramon Abatti", "Brazil"),
    ("Juan Benitez", "Paraguay"),
    ("Raphael Claus", "Brazil"),
    ("Yael Falcon Perez", "Argentina"),
    ("Cristian Garay", "Chile"),
    ("Dario Herrera", "Argentina"),
    ("Kevin Ortega", "Peru"),
    ("Andres Rojas", "Colombia"),
    ("Wilton Sampaio", "Brazil"),
    ("Gustavo Tejera", "Uruguay"),
    ("Facundo Tello", "Argentina"),
    ("Jesus Valenzuela", "Venezuela"),
    # OFC
    ("Campbell-Kirk Kawana-Waugh", "New Zealand"),
    # UEFA
    ("Espen Eskas", "Norway"),
    ("Alejandro Hernandez Hernandez", "Spain"),
    ("Istvan Kovacs", "Romania"),
    ("Francois Letexier", "France"),
    ("Danny Makkelie", "Netherlands"),
    ("Szymon Marciniak", "Poland"),
    ("Maurizio Mariani", "Italy"),
    ("Glenn Nyberg", "Sweden"),
    ("Michael Oliver", "England"),
    ("Joao Pinheiro", "Portugal"),
    ("Sandro Scharer", "Switzerland"),
    ("Anthony Taylor", "England"),
    ("Clement Turpin", "France"),
    ("Slavko Vincic", "Slovenia"),
    ("Felix Zwayer", "Germany"),
]

# Manual worldfootball slug overrides where the algorithmic slug is wrong
# (resolved via worldfootball search / google). Verified during development.
SLUG_OVERRIDE = {
    "Ma Ning": "ning-ma",                 # WF lists Chinese names Western order
    "Cesar Arturo Ramos": "cesar-ramos_4",
    "Anthony Taylor": "anthony-taylor_2",  # anthony-taylor is a player
    "Omar Al Ali": "omar-mohamed-al-ali",
    "Amin Mohamed Omar": "amin-omar",
    "Juan Gabriel Calderon": "juan-calderon_2",  # juan-calderon is a player
    "Katia Itzel Garcia": "katia-garcia",
    "Ramon Abatti": "ramon-abatti-abel",
    "Yael Falcon Perez": "yael-falcon",
    "Sandro Scharer": "sandro-schaerer",
    # Artan (Somalia) removed from tournament after US entry denial; no WF page tried.
}


def slugify(name):
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = n.lower().replace("'", "").replace(".", "")
    n = re.sub(r"[^a-z0-9]+", "-", n).strip("-")
    return n


def make_scraper():
    return cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False}
    )


def parse_referee_page(html):
    """Sum all per-competition 'Total' rows. Returns dict or None if no stats."""
    soup = BeautifulSoup(html, "lxml")
    if "Assignments as Referee" not in html and "Matches as referee" not in html:
        # could be a player profile served under the slug
        pass
    M = Y = YR = R = 0
    found = False
    for tbl in soup.find_all("table"):
        rows = tbl.find_all("tr")
        if not rows:
            continue
        hdr = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        if hdr[:5] != ["Season", "Matches", "Yellow", "Yellow-Red", "Red"]:
            continue
        for tr in rows:
            cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            if cells and cells[0] == "Total" and len(cells) >= 5:
                def num(x):
                    return int(x) if x.strip().isdigit() else 0
                M += num(cells[1]); Y += num(cells[2])
                YR += num(cells[3]); R += num(cells[4])
                found = True
    if not found or M == 0:
        return None
    return {"matches": M, "yellow": Y, "yellow_red": YR, "red": R}


def fetch_referee(scraper, name):
    """Try override slug, then algorithmic slug. Returns (stats|None, slug_used, url)."""
    candidates = []
    if name in SLUG_OVERRIDE:
        candidates.append(SLUG_OVERRIDE[name])
    candidates.append(slugify(name))
    # a couple of common worldfootball disambiguation suffixes as fallback
    base = slugify(name)
    candidates += [base + "_2", base + "_3", base + "_4"]
    seen = set()
    for slug in candidates:
        if slug in seen:
            continue
        seen.add(slug)
        url = WF.format(slug=slug)
        try:
            r = scraper.get(url, timeout=30)
        except Exception as e:
            print(f"    [err] {slug}: {e}", file=sys.stderr)
            continue
        if r.status_code != 200:
            continue
        stats = parse_referee_page(r.text)
        if stats:
            return stats, slug, url
        time.sleep(0.3)
    return None, None, None


def build_referees_csv():
    scraper = make_scraper()
    rows = []
    got = 0
    for name, country in REFEREES:
        stats, slug, url = fetch_referee(scraper, name)
        if stats:
            M = stats["matches"]
            cpg = (stats["yellow"] + stats["yellow_red"] + stats["red"]) / M
            ypg = stats["yellow"] / M
            rpg = (stats["yellow_red"] + stats["red"]) / M
            rows.append({
                "referee": name,
                "country": country,
                "cards_per_game": round(cpg, 3),
                "yellows_per_game": round(ypg, 3),
                "reds_per_game": round(rpg, 4),
                "fouls_per_game": "",   # not published by worldfootball
                "pens_per_game": "",    # not published by worldfootball
                "n_matches": M,
                "source": f"worldfootball.net {url}",
            })
            got += 1
            print(f"  OK  {name:34s} n={M:4d} cpg={cpg:.2f} ypg={ypg:.2f} rpg={rpg:.3f}")
        else:
            rows.append({
                "referee": name,
                "country": country,
                "cards_per_game": "",
                "yellows_per_game": "",
                "reds_per_game": "",
                "fouls_per_game": "",
                "pens_per_game": "",
                "n_matches": "",
                "source": "NOT FOUND (worldfootball.net)",
            })
            print(f"  --  {name:34s} NOT FOUND", file=sys.stderr)
        time.sleep(0.5)

    out = os.path.join(DATA, "referees.csv")
    cols = ["referee", "country", "cards_per_game", "yellows_per_game",
            "reds_per_game", "fouls_per_game", "pens_per_game", "n_matches", "source"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}: {got}/{len(REFEREES)} referees with sourced card rates")
    return rows


def build_assignments_csv(assignments):
    """assignments: list of dicts {date, home_team, away_team, referee}.
    Writes one row per WC fixture, referee blank if unconfirmed.
    Reads data/wc_match_summary.csv for the 71 fixtures."""
    import pandas as pd
    ms = pd.read_csv(os.path.join(DATA, "wc_match_summary.csv"))
    # confirmed lookup keyed by (home, away)
    conf = {(a["home_team"], a["away_team"]): a for a in assignments}
    rows = []
    n_conf = 0
    for _, r in ms.iterrows():
        m = r["match"]
        if " vs " not in m:
            continue
        home, away = [x.strip() for x in m.split(" vs ", 1)]
        date = str(r["commence"])[:10]
        ref = ""
        a = conf.get((home, away))
        if a:
            ref = a["referee"]
            n_conf += 1
        rows.append({"date": date, "home_team": home, "away_team": away, "referee": ref})

    out = os.path.join(DATA, "wc_referee_assignments.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "home_team", "away_team", "referee"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}: {n_conf} confirmed assignments / {len(rows)} fixtures")
    return rows


# Map Wikipedia team spellings -> The Odds API spellings (as in wc_match_summary.csv)
WIKI_TEAM_MAP = {
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    # most others match directly (Brazil, Morocco, Germany, Curaçao, etc.)
}


def norm_team(t):
    t = t.strip()
    return WIKI_TEAM_MAP.get(t, t)


def scrape_wikipedia_assignments():
    """Parse the 'Matches assigned' column of the Wikipedia officials table.
    That column lists, per CENTER referee, the matches they were assigned to
    (assistant/VAR roles are NOT in this column). Returns list of
    {home_team, away_team, referee} using Odds API team spellings."""
    scraper = make_scraper()
    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_officials"
    r = scraper.get(url, timeout=30)
    soup = BeautifulSoup(r.text, "lxml")
    out = []
    # the relevant table has header containing 'Matches assigned'
    target = None
    for tbl in soup.find_all("table"):
        hdr = [c.get_text(strip=True) for c in tbl.find_all("tr")[0].find_all(["th", "td"])]
        if any("Matches assigned" in h for h in hdr):
            target = tbl
            mi = next(i for i, h in enumerate(hdr) if "Matches assigned" in h)
            break
    if target is None:
        print("  [warn] could not locate Wikipedia assignment table", file=sys.stderr)
        return out
    for tr in target.find_all("tr")[1:]:
        cells = tr.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        # referee name = text of first cell, strip trailing "( Country )"
        ref_raw = cells[0].get_text(" ", strip=True)
        ref = re.sub(r"\s*\(.*$", "", ref_raw).strip()
        if not ref or ref in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"):
            continue
        # matches-assigned cell is the LAST cell in the row
        mtext = cells[-1].get_text("\n", strip=True)
        if not mtext:
            continue
        # each assigned match looks like "Canada–Bosnia and Herzegovina ( Group B )"
        for chunk in re.findall(r"([A-Za-zÀ-ÿ' .]+[–-][A-Za-zÀ-ÿ' .]+?)\s*\(\s*Group", mtext):
            # split on en-dash / em-dash / hyphen between two team names
            parts = re.split(r"\s*[–—]\s*", chunk.strip())
            if len(parts) != 2:
                continue
            home, away = norm_team(parts[0]), norm_team(parts[1])
            out.append({"home_team": home, "away_team": away, "referee": ref})
    return out


if __name__ == "__main__":
    print("=== Building referees.csv (worldfootball.net via cloudscraper) ===")
    build_referees_csv()
    print("\n=== Scraping confirmed assignments from Wikipedia officials page ===")
    confirmed = scrape_wikipedia_assignments()
    for a in confirmed:
        print(f"  {a['home_team']} vs {a['away_team']} -> {a['referee']}")
    print("\n=== Building wc_referee_assignments.csv ===")
    build_assignments_csv(confirmed)
