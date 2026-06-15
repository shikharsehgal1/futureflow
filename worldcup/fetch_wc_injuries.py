"""
fetch_wc_injuries.py — 2026 World Cup squad injury / suspension data collector.

PURPOSE
-------
Produce data/wc_injuries.csv for the prediction model, with one row per
genuinely-sourced injured / doubtful / suspended player on a WC2026 squad.

    columns: team, player, position, status, severity, note
      status   in {out, doubtful, suspended}
      severity 0..1  ≈ (importance of player) × (likelihood they miss the match)
      note     source + date so every row is traceable

HONEST COVERAGE NOTE
--------------------
Transfermarkt's national-team "sperrenundverletzungen" page does NOT carry
injury/suspension tables the way club pages do — for every national team it
returns "No information available" (verified live via cloudscraper; the fetch
SUCCEEDS, it's the data that doesn't exist on TM for international squads).
The club scraper in ../tm_injuries.py therefore cannot be pointed at national
teams. So the real payload below is hand-curated from reputable football news
(ESPN WC2026 injuries tracker, ESPN/Sports Mole match previews, Goal.com,
Covers.com, Wikipedia "2026 FIFA World Cup squads" official withdrawals) and
each record carries its source + date in the `note` column. Nothing here is
fabricated; items I could not source are simply absent.

The function `try_transfermarkt_national()` is kept as a live, runnable probe
so anyone can re-confirm that TM has nothing — it adapts the approach from
../tm_injuries.py (cloudscraper + bs4 parse of table.items).

MODELLING NOTE (for the consumer of this CSV)
---------------------------------------------
The betting market already prices KNOWN injuries. This file's value is mainly
as a LATE-NEWS FLAG: injuries / non-recoveries confirmed within ~48h of kickoff
that the market may still be lagging on. Treat long-known season-enders
(Rodrygo, Gnabry, etc.) as already-priced context; weight the recent, opener-
specific items (Neymar out, Enciso doubtful, Morocco's Aguerd/Ezzalzouli, the
late squad withdrawals) more heavily as edges.
"""

import os
import csv
import sys
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_PATH = os.path.join(DATA_DIR, "wc_injuries.csv")

# ── TEAM NAMES must match data/wc_match_summary.csv exactly ──────────────────
# (e.g. "Bosnia & Herzegovina", "Ivory Coast", "South Korea", "Turkey")

# ── Transfermarkt national-team ids (slug, id) — for the live probe only ─────
TM_NATIONAL = {
    "Brazil":               ("brasilien", "3439"),
    "South Korea":          ("sudkorea", "3589"),
    "Czech Republic":       ("tschechien", "3424"),
    "Switzerland":          ("schweiz", "3384"),
    "Netherlands":          ("niederlande", "3379"),
    "Germany":              ("deutschland", "3262"),
    "Japan":                ("japan", "3504"),
}

TM_BASE = "https://www.transfermarkt.com"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.com/",
}


def try_transfermarkt_national(teams=None, sleep=3.0):
    """
    Live probe: confirm whether TM carries any national-team injury rows.
    Adapts ../tm_injuries.py (cloudscraper + table.items parse).
    Returns dict {team: [rows]} — empirically every value is [] because
    TM renders "No information available" for national teams.
    """
    try:
        import cloudscraper
        from bs4 import BeautifulSoup
    except ImportError:
        print("cloudscraper/bs4 missing: pip3 install --user cloudscraper bs4")
        return {}

    teams = teams or list(TM_NATIONAL)
    s = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False})
    s.headers.update(HEADERS)

    import time
    out = {}
    for t in teams:
        if t not in TM_NATIONAL:
            continue
        slug, tid = TM_NATIONAL[t]
        url = f"{TM_BASE}/{slug}/sperrenundverletzungen/verein/{tid}/plus/1"
        rows = []
        try:
            r = s.get(url, timeout=20)
            print(f"  TM {t}: HTTP {r.status_code}", flush=True)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                body = soup.get_text(" ", strip=True).lower()
                if "no information available" in body:
                    print(f"    -> TM has no injury data for {t} (expected).")
                for tr in soup.select("table.items tbody tr"):
                    a = tr.select_one("td.hauptlink a")
                    if a:
                        rows.append(a.get_text(strip=True))
        except Exception as e:
            print(f"    TM error for {t}: {e}")
        out[t] = rows
        time.sleep(sleep)
    return out


# ── CURATED, SOURCED INJURY / SUSPENSION RECORDS ─────────────────────────────
# Each tuple: (team, player, position, status, severity, note)
# severity guidance (importance × likelihood-to-miss):
#   ~0.85+  key/star player, confirmed OUT
#   ~0.55   important player OUT but squad-depth position, or star DOUBTFUL
#   ~0.30   rotation/squad player out, or star likely-fit-but-flagged
# Sources abbreviated in note with date. Pulled 2026-06-11.
RECORDS = [
    # ---- Brazil (Group plays Morocco 2026-06-13) ----
    ("Brazil", "Neymar", "Left Winger", "out", 0.55,
     "ESPN tracker/match preview 2026-06-11: right calf strain, ruled out of opener vs Morocco; may return later in group"),
    ("Brazil", "Wesley", "Right-Back", "out", 0.45,
     "FOX/ESPN 2026-06-07: left thigh muscle injury, replaced on roster by Eder Silva"),
    ("Brazil", "Eder Militao", "Centre-Back", "out", 0.65,
     "ESPN tracker 2026-06: hamstring tear requiring surgery, out of tournament"),
    ("Brazil", "Rodrygo", "Winger", "out", 0.6,
     "ESPN/Goal 2026: torn ACL+meniscus (March), out for 2026"),
    ("Brazil", "Estevao", "Winger", "out", 0.5,
     "ESPN tracker 2026: torn hamstring, ruled out of tournament"),

    # ---- Morocco (vs Brazil 2026-06-13) ----
    ("Morocco", "Nayef Aguerd", "Centre-Back", "out", 0.75,
     "ESPN match preview 2026-06-11: ruled out of tournament (injury)"),
    ("Morocco", "Abde Ezzalzouli", "Winger", "out", 0.6,
     "ESPN/Flashscore 2026-06: knee injury, ruled out of tournament"),
    ("Morocco", "Noussair Mazraoui", "Right-Back", "doubtful", 0.45,
     "ESPN match preview 2026-06-11: shoulder injury in recent friendly, doubtful for opener"),
    ("Morocco", "Anass Salah-Eddine", "Left-Back", "doubtful", 0.3,
     "ESPN match preview 2026-06-11: injury doubt for opener"),
    ("Morocco", "Chemsdine Talbi", "Winger", "doubtful", 0.3,
     "ESPN match preview 2026-06-11: injury doubt for opener"),

    # ---- Netherlands (vs Japan 2026-06-14) ----
    ("Netherlands", "Jurrien Timber", "Centre-Back", "out", 0.6,
     "ESPN tracker 2026: groin strain, ruled out"),
    ("Netherlands", "Xavi Simons", "Attacking Midfield", "out", 0.7,
     "ESPN/Goal 2026: torn ACL (April, at Tottenham), out"),
    ("Netherlands", "Jerdy Schouten", "Defensive Midfield", "out", 0.5,
     "ESPN tracker 2026: torn ACL, ruled out"),
    ("Netherlands", "Matthijs de Ligt", "Centre-Back", "out", 0.7,
     "ESPN/Goal 2026: back injury requiring surgery, ruled out"),

    # ---- Japan (vs Netherlands 2026-06-14) ----
    ("Japan", "Kaoru Mitoma", "Left Winger", "out", 0.7,
     "ESPN/Goal 2026: torn hamstring, ruled out"),
    ("Japan", "Takumi Minamino", "Forward", "out", 0.55,
     "ESPN/Goal 2026: torn ACL (December), ruled out"),
    ("Japan", "Wataru Endo", "Defensive Midfield", "doubtful", 0.35,
     "ESPN tracker 2026: ruptured ankle ligaments, back on grass hoping to play"),

    # ---- Germany (vs Curacao 2026-06-14) ----
    ("Germany", "Serge Gnabry", "Right Winger", "out", 0.55,
     "ESPN/Goal 2026: torn adductor (April), ruled out"),
    ("Germany", "Marc-Andre ter Stegen", "Goalkeeper", "out", 0.45,
     "ESPN tracker 2026: torn thigh muscle, ruled out"),
    ("Germany", "Lennart Karl", "Attacking Midfield", "out", 0.3,
     "ESPN tracker 2026: torn thigh muscle, ruled out"),

    # ---- USA (vs Paraguay 2026-06-13) ----
    ("USA", "Johnny Cardoso", "Defensive Midfield", "out", 0.55,
     "ESPN/Goal 2026: high-grade ankle sprain / surgery (May), ruled out"),
    ("USA", "Patrick Agyemang", "Forward", "out", 0.4,
     "ESPN/Goal 2026: torn Achilles (April), ruled out"),
    ("USA", "Chris Richards", "Centre-Back", "doubtful", 0.3,
     "ESPN/Covers 2026: torn ankle ligaments, expected match-fit soon"),

    # ---- Paraguay (vs USA 2026-06-13) ----
    ("Paraguay", "Julio Enciso", "Attacking Midfield", "doubtful", 0.5,
     "Yahoo/Covers 2026-06: thigh injury, likely to miss opener vs USA, may feature later in group"),

    # ---- Scotland (vs Haiti 2026-06-14) ----
    ("Scotland", "Billy Gilmour", "Central Midfield", "out", 0.55,
     "ESPN/Goal 2026: knee injury, ruled out of tournament"),

    # ---- Australia (vs Turkey 2026-06-14) ----
    ("Australia", "Lewis Miller", "Right-Back", "out", 0.4,
     "ESPN tracker 2026: torn Achilles, ruled out"),
    ("Australia", "Aiden O'Neill", "Central Midfield", "doubtful", 0.35,
     "ESPN tracker 2026: ankle injury, status unclear"),
    ("Australia", "Patrick Yazbek", "Central Midfield", "doubtful", 0.3,
     "ESPN tracker 2026: quad injury, World Cup participation in doubt"),

    # ---- Turkey (vs Australia 2026-06-14) ----
    ("Turkey", "Arda Guler", "Attacking Midfield", "doubtful", 0.25,
     "Daily Sabah/Turkiye Today 2026-06-08: pulled hamstring but back in open training, on track / expected fit"),

    # ---- Ivory Coast (vs Ecuador 2026-06-14) ----
    ("Ivory Coast", "Clement Akpa", "Defender", "out", 0.3,
     "Wikipedia WC2026 squads 2026-05-29: withdrew injured, replaced by Christopher Operi"),

    # ---- Canada (vs Bosnia & Herzegovina 2026-06-12) ----
    ("Canada", "Marcelo Flores", "Winger", "out", 0.35,
     "Wikipedia/ESPN 2026: torn ACL, withdrew (May 31), replaced by Jayden Nelson"),
    ("Canada", "Alphonso Davies", "Left-Back", "doubtful", 0.5,
     "ESPN/Covers 2026-06: hamstring + prior ACL recovery, day-by-day, questionable for group stage"),

    # ---- South Korea (vs Czech Republic 2026-06-12) ----
    ("South Korea", "Cho Yu-min", "Centre-Back", "out", 0.3,
     "Wikipedia WC2026 squads 2026-05-31: withdrew injured, replaced by Cho Wi-je"),

    # ---- Bosnia & Herzegovina (vs Canada 2026-06-12) ----
    ("Bosnia & Herzegovina", "Osman Hadzikic", "Goalkeeper", "out", 0.2,
     "Wikipedia WC2026 squads 2026-06-01: withdrew injured, replaced by Mladen Jurkas"),
    ("Bosnia & Herzegovina", "Nidal Celik", "Midfielder", "out", 0.2,
     "Wikipedia WC2026 squads 2026-06-11: withdrew injured, replaced by Arjan Malic"),

    # ---- Other WC2026 teams (later kickoffs; already largely market-priced) ----
    ("France", "Hugo Ekitike", "Forward", "out", 0.35,
     "ESPN/Goal 2026: ruptured Achilles (April), out for 2026"),
    ("France", "William Saliba", "Centre-Back", "doubtful", 0.2,
     "ESPN/Covers 2026: back injury but cleared / expected to play"),
    ("Spain", "Fermin Lopez", "Central Midfield", "out", 0.4,
     "ESPN/Goal 2026: metatarsal fracture requiring surgery, ruled out"),
    ("Spain", "Lamine Yamal", "Right Winger", "doubtful", 0.3,
     "ESPN/Covers 2026: partially torn hamstring, expected fit for Jun 15 opener"),
    ("Spain", "Mikel Merino", "Central Midfield", "doubtful", 0.35,
     "ESPN tracker 2026: stress fracture in foot, racing to recover"),
    ("Argentina", "Cristian Romero", "Centre-Back", "doubtful", 0.4,
     "ESPN/Covers 2026: partially torn MCL; ESPN 'in jeopardy', Covers 'expected to play' — uncertain"),
    ("Argentina", "Nahuel Molina", "Right-Back", "doubtful", 0.3,
     "ESPN tracker 2026: thigh muscle injury, out ~2-3 weeks from May"),
    ("Argentina", "Lionel Messi", "Forward", "doubtful", 0.25,
     "ESPN/Covers 2026: thigh/hamstring overload, resuming training, expected to play"),
    ("Uruguay", "Jose Gimenez", "Centre-Back", "doubtful", 0.4,
     "ESPN/Covers 2026: severe/high ankle sprain, participation uncertain"),
    ("Croatia", "Luka Modric", "Central Midfield", "doubtful", 0.35,
     "ESPN tracker 2026: fractured cheekbone, confidence he heals in time"),
    ("England", "Tino Livramento", "Right-Back", "out", 0.3,
     "ESPN tracker 2026: thigh injury, season-ending"),
    ("England", "Reece James", "Right-Back", "doubtful", 0.3,
     "ESPN tracker 2026: pulled hamstring, has chance if remains fit"),
    ("Ghana", "Mohammed Kudus", "Attacking Midfield", "doubtful", 0.45,
     "ESPN tracker 2026: hamstring setback, participation in doubt"),
    ("Algeria", "Luca Zidane", "Goalkeeper", "doubtful", 0.3,
     "ESPN tracker 2026: fractured jaw and chin, expected ~3 weeks out post-surgery"),
    ("Mexico", "Luis Angel Malagon", "Goalkeeper", "out", 0.35,
     "Goal 2026: ruptured Achilles tendon, ruled out"),
    ("Austria", "Christoph Baumgartner", "Attacking Midfield", "out", 0.45,
     "Goal 2026: torn thigh muscle requiring surgery, ruled out"),
]


def write_csv(records, path=OUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "player", "position", "status", "severity", "note"])
        for r in records:
            w.writerow(r)
    print(f"Wrote {len(records)} rows -> {path}")


def main():
    if "--probe-tm" in sys.argv:
        print("Probing Transfermarkt national-team injury pages "
              "(expect 'no information available')...")
        res = try_transfermarkt_national()
        total = sum(len(v) for v in res.values())
        print(f"TM national-team injury rows found across probed teams: {total}")
        print("Conclusion: TM does not carry national-team injury tables; "
              "using curated web-sourced records below.\n")
    write_csv(RECORDS)
    teams = sorted({r[0] for r in RECORDS})
    print(f"\nTeams covered ({len(teams)}): {', '.join(teams)}")


if __name__ == "__main__":
    main()
