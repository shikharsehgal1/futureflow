"""
fetch_fixtures.py — Fetches upcoming fixtures + Bet365 odds via The Odds API
and writes them to data/fixtures.csv ready for part7_predict.py.

Setup (one time only):
    1. Go to https://the-odds-api.com and sign up for a free account
    2. Copy your API key from the dashboard
    3. Either paste it below as API_KEY = "your_key_here"
       or set it as an environment variable: export ODDS_API_KEY="your_key_here"

Free tier: 500 requests/month — running this weekly uses ~10 requests total.

Usage:
    python fetch_fixtures.py                  # all 10 leagues, next 14 days
    python fetch_fixtures.py --div E0 D1      # EPL + Bundesliga only
    python fetch_fixtures.py --days 7         # next 7 days only
    python fetch_fixtures.py --results        # score settled predictions
"""

import argparse
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY        = os.environ.get("ODDS_API_KEY", "0ce377e63ef8333b1794c5f26e08b384")
BASE_URL       = "https://api.the-odds-api.com/v4"
FIXTURES_PATH  = "data/fixtures.csv"
PREDICTIONS_LOG= "data/predictions_log.csv"
DATA_PATH      = "data/big5_with_probs.csv"

# Map our division codes to The Odds API sport keys
SPORT_MAP = {
    "E0":  "soccer_epl",
    "E1":  "soccer_england_championship",
    "D1":  "soccer_germany_bundesliga",
    "D2":  "soccer_germany_bundesliga2",
    "SP1": "soccer_spain_la_liga",
    "SP2": "soccer_spain_segunda_division",
    "I1":  "soccer_italy_serie_a",
    "I2":  "soccer_italy_serie_b",
    "F1":  "soccer_france_ligue_one",
    "F2":  "soccer_france_ligue_two",
}

DIV_NAMES = {
    "E0": "Premier League",   "E1": "Championship",
    "D1": "Bundesliga",       "D2": "2. Bundesliga",
    "SP1": "La Liga",         "SP2": "Segunda División",
    "I1": "Serie A",          "I2": "Serie B",
    "F1": "Ligue 1",          "F2": "Ligue 2",
}

# Map Odds API team names to the names in your data
# Add more here if you see WARNING messages in part7_predict.py
TEAM_NAME_MAP = {
    # Premier League
    "Nottingham Forest":          "Nott'm Forest",
    "Manchester City":            "Man City",
    "Manchester United":          "Man United",
    "Newcastle United":           "Newcastle",
    "Tottenham Hotspur":          "Tottenham",
    "West Ham United":            "West Ham",
    "Wolverhampton Wanderers":    "Wolves",
    "Brighton & Hove Albion":     "Brighton",
    # Bundesliga
    "Borussia Dortmund":          "Dortmund",
    "Eintracht Frankfurt":        "Ein Frankfurt",
    "Borussia Mönchengladbach":   "M'gladbach",
    "Borussia Moenchengladbach":  "M'gladbach",
    "FC Köln":                    "FC Koln",
    "Bayer Leverkusen":           "Leverkusen",
    "VfB Stuttgart":              "Stuttgart",
    "FC Augsburg":                "Augsburg",
    "SC Freiburg":                "Freiburg",
    "1. FC Union Berlin":         "Union Berlin",
    "1. FSV Mainz 05":            "Mainz",
    "TSG Hoffenheim":             "Hoffenheim",
    "VfL Wolfsburg":              "Wolfsburg",
    "SV Werder Bremen":           "Werder Bremen",
    "FC St. Pauli":               "St Pauli",
    "1. FC Heidenheim":           "Heidenheim",
    "Hamburger SV":               "Hamburg",
    # La Liga
    "Athletic Club":              "Ath Bilbao",
    "Atletico Madrid":            "Ath Madrid",
    "Atlético Madrid":            "Ath Madrid",
    "Real Sociedad":              "Sociedad",
    "Rayo Vallecano":             "Vallecano",
    "Espanyol":                   "Espanol",
    "Deportivo Alaves":           "Alaves",
    "Deportivo Alavés":           "Alaves",
    "Real Betis":                 "Betis",
    # Serie A
    "AC Milan":                   "Milan",
    "Inter Milan":                "Inter",
    "Hellas Verona":              "Verona",
    # Ligue 1
    "Paris Saint-Germain":        "Paris SG",
    "Stade Brestois 29":          "Brest",
    "RC Lens":                    "Lens",
    "AS Monaco":                  "Monaco",
    "Olympique Marseille":        "Marseille",
    "Olympique Lyonnais":         "Lyon",
    "Stade Rennais":              "Rennes",
    "OGC Nice":                   "Nice",
    "RC Strasbourg":              "Strasbourg",
    "FC Nantes":                  "Nantes",
    "Toulouse FC":                "Toulouse",
    "Le Havre AC":                "Le Havre",
    # Championship
    "Sheffield Wednesday":        "Sheffield Weds",
    "West Bromwich Albion":       "West Brom",
    "Queens Park Rangers":        "QPR",
    "Leeds United":               "Leeds",
    "Brighton and Hove Albion":   "Brighton",
    # Bundesliga extras
    "FSV Mainz 05":               "Mainz",
    "1. FC Köln":                 "FC Koln",
    "1. FC Koeln":                "FC Koln",
    "Borussia Monchengladbach":   "M'gladbach",
    "Hannover 96":                "Hannover",
    "SC Paderborn":               "Paderborn",
    "SV Darmstadt 98":            "Darmstadt",
    "Karlsruher SC":              "Karlsruhe",
    "Eintracht Braunschweig":     "Braunschweig",
    "Greuther Fürth":             "Greuther Furth",
    "1. FC Nürnberg":             "Nurnberg",
    "1. FC Kaiserslautern":       "Kaiserslautern",
    "Arminia Bielefeld":          "Bielefeld",
    "Dynamo Dresden":             "Dresden",
    "SC Preußen Münster":         "Preuss Muenster",
    "Fortuna Düsseldorf":         "Fortuna Dusseldorf",
    "VfL Bochum":                 "Bochum",
    "Holstein Kiel":              "Holstein Kiel",
    "1. FC Magdeburg":            "Magdeburg",
    "Hertha Berlin":              "Hertha",
    # La Liga extras
    "CA Osasuna":                 "Osasuna",
    "Celta Vigo":                 "Celta",
    "Alavés":                     "Alaves",
    "Athletic Bilbao":            "Ath Bilbao",
    "Elche CF":                   "Elche",
    # Serie A extras
    "Atalanta BC":                "Atalanta",
    "AS Roma":                    "Roma",
    # Ligue 1 extras
    "Paris Saint Germain":        "Paris SG",
    "Nice":                       "Nice",
    "FC Schalke 04":              "Schalke 04",
    "Le Mans FC":                 "Le Mans",
    "Stade de Reims":             "Reims",
    "Stade Lavallois":            "Laval",
    "Rodez AF":                   "Rodez",
    "SC Bastia":                  "Bastia",
    "USL Dunkerque":              "Dunkerque",
    "Saint Etienne":              "St Etienne",
    "Annecy FC":                  "Annecy",
    # Serie B extras
    "Cesena FC":                  "Cesena",
    "US Catanzaro 1929":          "Catanzaro",
    "Südtirol":                   "Sudtirol",
    # Segunda extras
    "SD Huesca":                  "Huesca",
    "Almería":                    "Almeria",
    "Real Racing Club de Santander": "Santander",
    "Andorra CF":                 "Andorra",
    "Cádiz CF":                   "Cadiz",
    "Málaga":                     "Malaga",
    "Deportivo La Coruña":        "La Coruna",
    "Burgos CF":                  "Burgos",
    "CD Mirandés":                "Mirandes",
    "Real Valladolid CF":         "Valladolid",
    "Real Sociedad B":            "Sociedad B",
    "Granada CF":                 "Granada",
    "Las Palmas":                 "Las Palmas",
    "Sporting Gijón":             "Sp Gijon",
    "CD Castellón":               "Castellon",
    "Cultural Leonesa":           "Cultural Leonesa",
    "Leganés":                    "Leganes",
    "AD Ceuta FC":                "Ceuta",
    "SD Eibar":                   "Eibar",
    "Córdoba":                    "Cordoba",
}

def normalise_team(name):
    return TEAM_NAME_MAP.get(name, name)


# ── FETCH ODDS FROM API ───────────────────────────────────────────────────────
def fetch_odds_for_league(div, days_ahead=14):
    sport  = SPORT_MAP[div]
    url    = f"{BASE_URL}/sports/{sport}/odds"
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)

    params = {
        "apiKey":           API_KEY,
        "regions":          "eu",
        "markets":          "h2h,totals",   # h2h + over/under 2.5
        "oddsFormat":       "decimal",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo":   cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    try:
        resp      = requests.get(url, params=params, timeout=30)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")

        if resp.status_code == 401:
            print(f"ERROR: Invalid API key — get one free at https://the-odds-api.com")
            return [], remaining, used
        if resp.status_code == 404:
            print(f"not on free tier — skipping")
            return [], remaining, used
        if resp.status_code == 429:
            print(f"monthly quota exceeded — skipping")
            return [], remaining, used
        resp.raise_for_status()
        return resp.json(), remaining, used

    except Exception as e:
        print(f"ERROR: {e}")
        return [], "?", "?"


def parse_fixtures(games, div):
    rows = []
    for game in games:
        home = normalise_team(game["home_team"])
        away = normalise_team(game["away_team"])
        date = datetime.fromisoformat(
            game["commence_time"].replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")

        odds_h = odds_d = odds_a = np.nan
        odds_over = odds_under = np.nan
        # Prefer bet365, fall back to first available bookmaker
        bookmakers = game.get("bookmakers", [])
        preferred  = [b for b in bookmakers if b["key"] == "bet365"]
        bookie_list = preferred if preferred else bookmakers[:1]
        for bookie in bookie_list:
            for market in bookie.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"]
                                for o in market["outcomes"]}
                    odds_h = outcomes.get(game["home_team"],
                             outcomes.get(home, np.nan))
                    odds_a = outcomes.get(game["away_team"],
                             outcomes.get(away, np.nan))
                    for k, v in outcomes.items():
                        if k not in [game["home_team"], game["away_team"],
                                     home, away]:
                            odds_d = v
                elif market["key"] == "totals":
                    # Find Over/Under 2.5 line
                    for o in market["outcomes"]:
                        if o.get("point") == 2.5:
                            if o["name"] == "Over":
                                odds_over = o["price"]
                            elif o["name"] == "Under":
                                odds_under = o["price"]

        rows.append({
            "Date":      date,
            "Div":       div,
            "HomeTeam":  home,
            "AwayTeam":  away,
            "OddsH":     round(float(odds_h), 2) if pd.notna(odds_h) else np.nan,
            "OddsD":     round(float(odds_d), 2) if pd.notna(odds_d) else np.nan,
            "OddsA":     round(float(odds_a), 2) if pd.notna(odds_a) else np.nan,
            "OddsOver":  round(float(odds_over), 2) if pd.notna(odds_over) else np.nan,
            "OddsUnder": round(float(odds_under), 2) if pd.notna(odds_under) else np.nan,
        })
    return rows


# ── WEATHER FORECAST ──────────────────────────────────────────────────────────
# Stadium coordinates for all 10 leagues
# Used to fetch weather forecast on Fridays for upcoming fixtures
STADIUM_COORDS = {
    # Premier League / Championship
    "Arsenal":(51.5549,-0.1084),"Aston Villa":(52.5092,-1.8847),
    "Barnsley":(53.5527,-1.4674),"Birmingham":(52.4752,-1.7865),
    "Blackburn":(53.7286,-2.4890),"Blackpool":(53.8043,-3.0480),
    "Bolton":(53.5805,-2.5357),"Bournemouth":(50.7352,-1.8383),
    "Brentford":(51.4882,-0.3088),"Brighton":(50.8618,-0.0837),
    "Bristol City":(51.4400,-2.6200),"Burnley":(53.7893,-2.2297),
    "Cardiff":(51.4728,-3.2030),"Charlton":(51.4865,0.0366),
    "Chelsea":(51.4816,-0.1910),"Coventry":(52.4484,-1.5576),
    "Crystal Palace":(51.3983,-0.0855),"Derby":(52.9151,-1.4472),
    "Everton":(53.4388,-2.9661),"Fulham":(51.4749,-0.2217),
    "Hull":(53.7460,-0.3675),"Ipswich":(52.0551,1.1449),
    "Leeds":(53.7775,-1.5724),"Leicester":(52.6204,-1.1422),
    "Liverpool":(53.4308,-2.9608),"Luton":(51.8837,-0.4316),
    "Man City":(53.4831,-2.2004),"Man United":(53.4631,-2.2913),
    "Middlesbrough":(54.5782,-1.2175),"Millwall":(51.4858,-0.0508),
    "Newcastle":(54.9756,-1.6217),"Norwich":(52.6221,1.3093),
    "Nott'm Forest":(52.9399,-1.1328),"Oxford":(51.7174,-1.2149),
    "Portsmouth":(50.7965,-1.0641),"Preston":(53.7729,-2.6911),
    "QPR":(51.5093,-0.2321),"Sheffield United":(53.3703,-1.4706),
    "Sheffield Weds":(53.4114,-1.5007),"Southampton":(50.9058,-1.3914),
    "Stoke":(52.9883,-2.1752),"Sunderland":(54.9145,-1.3879),
    "Swansea":(51.6428,-3.9344),"Tottenham":(51.6042,-0.0665),
    "Watford":(51.6498,-0.4017),"West Brom":(52.5090,-1.9642),
    "West Ham":(51.5386,0.0164),"Wolves":(52.5900,-2.1303),
    "Wrexham":(53.0463,-3.0041),"Wigan":(53.5488,-2.6372),
    # Bundesliga / 2. Bundesliga
    "Augsburg":(48.3232,10.8854),"Bayern Munich":(48.2188,11.6247),
    "Bielefeld":(51.9997,8.5248),"Bochum":(51.4900,7.2376),
    "Braunschweig":(52.2733,10.5251),"Darmstadt":(49.8728,8.6530),
    "Dortmund":(51.4926,7.4517),"Dresden":(51.0375,13.7595),
    "Ein Frankfurt":(50.0687,8.6453),"Elversberg":(49.3470,7.0920),
    "FC Koln":(50.9336,6.8750),"Fortuna Dusseldorf":(51.2738,6.7933),
    "Freiburg":(47.9879,7.8955),"Greuther Furth":(49.4812,10.9917),
    "Hamburg":(53.5872,10.0182),"Hannover":(52.3606,9.7285),
    "Heidenheim":(48.6766,10.1551),"Hertha":(52.5147,13.2394),
    "Hoffenheim":(49.2380,8.8892),"Holstein Kiel":(54.3523,10.1332),
    "Kaiserslautern":(49.4345,7.7778),"Karlsruhe":(49.0244,8.4147),
    "Leverkusen":(51.0384,7.0023),"Magdeburg":(52.1350,11.6200),
    "Mainz":(49.9843,8.2242),"M'gladbach":(51.1742,6.3853),
    "Nurnberg":(49.4269,11.1228),"Paderborn":(51.7290,8.7538),
    "Preuss Muenster":(51.9769,7.6440),"RB Leipzig":(51.3457,12.3474),
    "St Pauli":(53.5547,9.9682),"Stuttgart":(48.7925,9.2325),
    "Union Berlin":(52.4575,13.5675),"Werder Bremen":(53.0663,8.8375),
    "Wolfsburg":(52.4344,10.8032),
    # La Liga / Segunda
    "Alaves":(42.8530,-2.6819),"Almeria":(36.8376,-2.4500),
    "Ath Bilbao":(43.2642,-2.9494),"Ath Madrid":(40.4361,-3.5996),
    "Barcelona":(41.3809,2.1228),"Betis":(37.3562,-5.9822),
    "Burgos":(42.3440,-3.6970),"Cadiz":(36.5338,-6.2927),
    "Castellon":(39.9500,-0.0495),"Celta":(42.2117,-8.7388),
    "Eibar":(43.1849,-2.4729),"Elche":(38.2585,-0.7032),
    "Espanol":(41.3479,2.0750),"Getafe":(40.3244,-3.7142),
    "Girona":(41.9618,2.8270),"Granada":(37.1726,-3.5970),
    "Huesca":(42.1467,-0.4165),"Las Palmas":(28.1003,-15.4503),
    "Leganes":(40.3524,-3.7638),"Levante":(39.4927,-0.3480),
    "Mallorca":(39.5896,2.6502),"Malaga":(36.7107,-4.4283),
    "Mirandes":(42.6851,-2.9441),"Osasuna":(42.7968,-1.6376),
    "Oviedo":(43.3566,-5.8456),"Real Madrid":(40.4531,-3.6883),
    "Santander":(43.4659,-3.8264),"La Coruna":(43.3333,-8.4167),
    "Sevilla":(37.3840,-5.9706),"Sociedad":(43.3017,-1.9742),
    "Sp Gijon":(43.5333,-5.6333),"Tenerife":(28.4686,-16.2546),
    "Valencia":(39.4747,-0.3586),"Valladolid":(41.6430,-4.7575),
    "Vallecano":(40.3914,-3.6561),"Villarreal":(39.9444,-0.1031),
    "Zaragoza":(41.6474,-0.9116),"Albacete":(38.9948,-1.8514),
    "Andorra":(42.5000,1.5218),"Almeria":(36.8376,-2.4500),
    # Serie A / Serie B
    "Atalanta":(45.7091,9.6796),"Avellino":(40.9148,14.7794),
    "Bari":(41.0806,16.8683),"Bologna":(44.4924,11.3094),
    "Brescia":(45.5381,10.2179),"Cagliari":(39.2000,9.1340),
    "Carrarese":(44.0833,10.0833),"Catanzaro":(38.9000,16.5870),
    "Cesena":(44.1333,12.2333),"Como":(45.8081,9.0852),
    "Cremonese":(45.1333,10.0333),"Empoli":(43.7167,10.9500),
    "Fiorentina":(43.7805,11.2828),"Frosinone":(41.6333,13.3167),
    "Genoa":(44.4133,8.9526),"Inter":(45.4781,9.1240),
    "Juventus":(45.1096,7.6413),"Lazio":(41.9340,12.4548),
    "Lecce":(40.3667,18.1833),"Mantova":(45.1500,10.7833),
    "Milan":(45.4781,9.1240),"Modena":(44.6400,10.9253),
    "Monza":(45.5833,9.2833),"Napoli":(40.8279,14.1931),
    "Padova":(45.3944,11.8891),"Palermo":(38.1333,13.3667),
    "Parma":(44.7925,10.3958),"Pescara":(42.4500,14.2000),
    "Pisa":(43.6936,10.3885),"Reggiana":(44.7000,10.6333),
    "Roma":(41.9340,12.4548),"Sampdoria":(44.4133,8.9526),
    "Sassuolo":(44.7000,10.6333),"Spezia":(44.1000,9.8333),
    "Sudtirol":(46.5000,11.3500),"Torino":(45.0408,7.6500),
    "Udinese":(46.0792,13.2006),"Venezia":(45.4408,12.3155),
    "Verona":(45.4272,10.9922),"Virtus Entella":(44.3500,9.3167),
    "Juve Stabia":(40.7056,14.4933),
    # Ligue 1 / Ligue 2
    "Ajaccio":(41.9200,8.7381),"Amiens":(49.8941,2.2831),
    "Angers":(47.4667,-0.5500),"Annecy":(45.8992,6.1294),
    "Auxerre":(47.7833,3.5667),"Bastia":(42.6878,9.4500),
    "Boulogne":(50.7200,1.6167),"Brest":(48.3900,-4.4833),
    "Clermont":(45.7731,3.1000),"Dunkerque":(51.0333,2.3667),
    "Grenoble":(45.1667,5.7167),"Guingamp":(48.5608,-3.1500),
    "Laval":(48.0667,-0.7667),"Le Havre":(49.4833,0.1000),
    "Le Mans":(48.0039,0.1956),"Lens":(50.4333,2.8167),
    "Lille":(50.6167,3.1333),"Lorient":(47.7500,-3.3667),
    "Lyon":(45.7654,4.9822),"Marseille":(43.2700,5.3953),
    "Metz":(49.1100,6.1778),"Monaco":(43.7278,7.4153),
    "Montpellier":(43.6222,3.8128),"Nancy":(48.6833,6.2000),
    "Nantes":(47.2558,-1.5247),"Nice":(43.7050,7.2603),
    "Paris FC":(48.8167,2.3000),"Paris SG":(48.8414,2.2530),
    "Pau FC":(43.3000,-0.3667),"Red Star":(48.9136,2.3628),
    "Reims":(49.2389,4.0531),"Rennes":(48.1078,-1.7144),
    "Rodez":(44.3500,2.5667),"St Etienne":(45.4608,4.3906),
    "Strasbourg":(48.5628,7.7519),"Toulouse":(43.5833,1.4333),
    "Troyes":(48.2956,4.0667),
}


def weather_goal_adjustment(temp, wind, precip):
    """
    Returns goal total multiplier based on weather conditions.
    Applied symmetrically to both xGH and xGA.
    Sources: Pavlinovic 2024, Zhong 2024, Bray 2008.
    """
    adj = 1.0
    if temp is not None and not np.isnan(float(temp)):
        t = float(temp)
        if   t >= 28: adj *= 0.85
        elif t >= 21: adj *= 0.92
        elif t <  5:  adj *= 0.97
    if wind is not None and not np.isnan(float(wind)):
        w = float(wind)
        if   w >= 40: adj *= 0.88
        elif w >= 30: adj *= 0.92
        elif w >= 20: adj *= 0.96
        elif w >= 10: adj *= 0.98
    if precip is not None and not np.isnan(float(precip)):
        p = float(precip)
        if   p >= 15: adj *= 0.90
        elif p >= 5:  adj *= 0.95
        elif p >= 2:  adj *= 0.97
    return round(adj, 4)


def fetch_forecast_weather(fixtures_df):
    """
    Fetch weather forecast for each fixture from Open-Meteo forecast API.
    Adds precipitation, wind_speed, temperature, weather_adj columns.
    Returns updated DataFrame.
    """
    print("\nFetching weather forecasts for fixtures...")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    precips = []; winds = []; temps = []; adjs = []

    for _, row in fixtures_df.iterrows():
        coords = STADIUM_COORDS.get(row["HomeTeam"])
        if coords is None:
            precips.append(np.nan); winds.append(np.nan)
            temps.append(np.nan);  adjs.append(1.0)
            continue

        lat, lon  = coords
        date_str  = str(row["Date"])[:10]

        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":        lat,
            "longitude":       lon,
            "daily":           "precipitation_sum,wind_speed_10m_max,temperature_2m_mean",
            "start_date":      date_str,
            "end_date":        date_str,
            "timezone":        "auto",
            "wind_speed_unit": "kmh",
        }

        try:
            r = session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data  = r.json().get("daily", {})
            prec  = data.get("precipitation_sum",  [None])[0]
            wind  = data.get("wind_speed_10m_max", [None])[0]
            temp  = data.get("temperature_2m_mean",[None])[0]
            adj   = weather_goal_adjustment(temp, wind, prec)
            precips.append(prec); winds.append(wind)
            temps.append(temp);   adjs.append(adj)
        except Exception:
            precips.append(np.nan); winds.append(np.nan)
            temps.append(np.nan);   adjs.append(1.0)

        time.sleep(0.2)

    fixtures_df = fixtures_df.copy()
    fixtures_df["precipitation"] = precips
    fixtures_df["wind_speed"]    = winds
    fixtures_df["temperature"]   = temps
    fixtures_df["weather_adj"]   = adjs

    # Summary
    fetched = sum(1 for a in adjs if a != 1.0)
    if fetched:
        print(f"  Weather adjustments applied to {fetched} fixtures:")
        for _, r in fixtures_df[fixtures_df["weather_adj"] != 1.0].iterrows():
            print(f"    {r['HomeTeam']} vs {r['AwayTeam']}: "
                  f"temp={r['temperature']:.0f}°C "
                  f"wind={r['wind_speed']:.0f}km/h "
                  f"rain={r['precipitation']:.1f}mm "
                  f"→ adj={r['weather_adj']:.3f}")
    else:
        print("  No extreme weather conditions — all adjustments = 1.0")

    return fixtures_df
def fetch_all_fixtures(divs, days_ahead=14):
    all_rows      = []
    last_remaining = None

    for div in divs:
        print(f"  {DIV_NAMES.get(div, div):<25} ", end="", flush=True)
        games, remaining, used = fetch_odds_for_league(div, days_ahead)
        last_remaining = remaining

        if not games:
            print("0 fixtures")
            continue

        rows      = parse_fixtures(games, div)
        has_odds  = sum(1 for r in rows if pd.notna(r.get("OddsH")))
        all_rows.extend(rows)
        print(f"{len(rows)} fixtures  ({has_odds} with Bet365 odds)")

    if last_remaining:
        print(f"\n  API quota remaining this month: {last_remaining} requests")

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ── MERGE WITH EXISTING FIXTURES ──────────────────────────────────────────────
def merge_fixtures(new_df):
    try:
        existing      = pd.read_csv(FIXTURES_PATH)
        existing_keys = set(zip(existing["Div"],
                                existing["HomeTeam"],
                                existing["AwayTeam"]))
        new_df["_key"] = list(zip(new_df["Div"],
                                   new_df["HomeTeam"],
                                   new_df["AwayTeam"]))
        truly_new = new_df[~new_df["_key"].isin(existing_keys)].drop(columns=["_key"])
        merged    = pd.concat([existing, truly_new], ignore_index=True)
        print(f"  Added {len(truly_new)} new  |  kept {len(existing)} existing")
    except FileNotFoundError:
        merged = new_df
        print(f"  Created fixtures.csv with {len(merged)} fixtures")

    return merged.sort_values(["Date","Div"]).reset_index(drop=True)


# ── SCORE SETTLED PREDICTIONS ─────────────────────────────────────────────────
def score_predictions():
    if not os.path.exists(PREDICTIONS_LOG):
        print(f"No log at {PREDICTIONS_LOG} — run part7_predict.py first.")
        return

    log = pd.read_csv(PREDICTIONS_LOG)
    log["Date"] = pd.to_datetime(log["Date"])

    results = pd.read_csv(DATA_PATH, encoding="utf-8", low_memory=False)
    results["Date"] = pd.to_datetime(results["Date"])

    merged  = log.merge(
        results[["Date","Div","HomeTeam","AwayTeam",
                 "HomeGoals","AwayGoals","Result"]],
        on=["Date","Div","HomeTeam","AwayTeam"], how="left"
    )
    pending = merged[merged["Result"].isna()]
    settled = merged[merged["Result"].notna()].copy()

    if settled.empty:
        print(f"No settled predictions yet — {len(pending)} still pending.")
        if not pending.empty:
            print("\nPending games:")
            print(pending[["Date","Div","HomeTeam","AwayTeam"]].to_string(index=False))
        return

    def model_ll(row):
        p = {"H": row["pH_model"]/100, "D": row["pD_model"]/100,
             "A": row["pA_model"]/100}.get(row["Result"], np.nan)
        return -np.log(max(p, 1e-6)) if pd.notna(p) else np.nan

    def mkt_ll(row):
        if pd.isna(row.get("pH_mkt", np.nan)): return np.nan
        p = {"H": row["pH_mkt"]/100, "D": row["pD_mkt"]/100,
             "A": row["pA_mkt"]/100}.get(row["Result"], np.nan)
        return -np.log(max(p, 1e-6)) if pd.notna(p) else np.nan

    settled["ll_model"] = settled.apply(model_ll, axis=1)
    settled["ll_mkt"]   = settled.apply(mkt_ll,   axis=1)

    # ── Closing Line Value (CLV) ──────────────────────────────────────────────
    # CLV measures whether your opening odds were better than closing odds.
    # Positive CLV = you were on the right side before the market moved.
    # fair_H/D/A stored at prediction time; compare to closing pH_mkt/pD_mkt/pA_mkt.
    # CLV = log(fair_prob_open / closing_prob) — positive = beat the close.
    def clv(row):
        if pd.isna(row.get("fair_H")) or pd.isna(row.get("pH_mkt")): return np.nan
        result = row["Result"]
        fair_p  = {"H": row["fair_H"], "D": row["fair_D"], "A": row["fair_A"]}.get(result)
        close_p = {"H": row["pH_mkt"]/100, "D": row["pD_mkt"]/100,
                   "A": row["pA_mkt"]/100}.get(result)
        if fair_p and close_p and close_p > 0:
            return round(np.log(fair_p / close_p) * 100, 2)
        return np.nan

    has_fair = "fair_H" in settled.columns
    if has_fair:
        settled["clv"] = settled.apply(clv, axis=1)

    def top_pick(row):
        probs = {"H": row["pH_model"], "D": row["pD_model"], "A": row["pA_model"]}
        return max(probs, key=probs.get)

    settled["model_pick"]   = settled.apply(top_pick, axis=1)
    settled["pick_correct"] = settled["model_pick"] == settled["Result"]

    # ── Game by game ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  SETTLED PREDICTIONS — {len(settled)} games")
    print(f"{'='*75}")
    print(f"  {'Date':<12} {'Home':<20} {'Away':<20} {'Score':>6} {'':>2} {'LL':>7}")
    print(f"  {'-'*68}")
    for _, r in settled.sort_values(["Date","Div"]).iterrows():
        score = f"{int(r['HomeGoals'])}-{int(r['AwayGoals'])}"
        tick  = "✓" if r["pick_correct"] else "✗"
        print(f"  {str(r['Date'].date()):<12} {r['HomeTeam']:<20} "
              f"{r['AwayTeam']:<20} {score:>6}  {tick}  {r['ll_model']:>7.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n        = len(settled)
    ll_model = settled["ll_model"].mean()
    accuracy = settled["pick_correct"].mean()
    has_mkt  = settled["ll_mkt"].notna().any()

    print(f"\n{'='*75}")
    print(f"  SUMMARY  ({n} settled, {len(pending)} pending)")
    print(f"{'='*75}")
    print(f"  Model log-loss  : {ll_model:.4f}")
    if has_mkt:
        ll_mkt = settled["ll_mkt"].dropna().mean()
        gap    = ll_mkt - ll_model
        print(f"  Market log-loss : {ll_mkt:.4f}")
        print(f"  Gap             : {gap:+.4f}  "
              f"({'model better ✓' if gap > 0 else 'market better'})")
    print(f"  Pick accuracy   : {accuracy:.1%}  "
          f"({settled['pick_correct'].sum()}/{n})")

    if has_fair and settled["clv"].notna().any():
        mean_clv = settled["clv"].mean()
        print(f"  Closing Line Val: {mean_clv:+.2f}  "
              f"({'beating close ✓' if mean_clv > 0 else 'behind close'})")

    if n >= 5:
        by_div = settled.groupby("Div").agg(
            N        =("ll_model","count"),
            LL_model =("ll_model","mean"),
            Accuracy =("pick_correct","mean"),
        ).reset_index()
        print(f"\n  {'Div':<6} {'League':<22} {'N':>4} {'LL_model':>9} {'Acc%':>6}")
        print(f"  {'-'*52}")
        for _, r in by_div.sort_values("Div").iterrows():
            print(f"  {r['Div']:<6} {DIV_NAMES.get(r['Div'],''):<22} "
                  f"{r['N']:>4} {r['LL_model']:>9.4f} {r['Accuracy']:>5.1%}")

    settled.to_csv(PREDICTIONS_LOG.replace(".csv","_settled.csv"), index=False)
    print(f"\n  Settled log → {PREDICTIONS_LOG.replace('.csv','_settled.csv')}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(divs=None, days_ahead=14, score=False):
    if score:
        print("Scoring settled predictions...")
        score_predictions()
        return

    if API_KEY == "YOUR_API_KEY_HERE":
        print("=" * 60)
        print("  No API key set!")
        print("  1. Go to https://the-odds-api.com  (free signup)")
        print("  2. Copy your API key from the dashboard")
        print("  3. Open fetch_fixtures.py and replace:")
        print('     API_KEY = "YOUR_API_KEY_HERE"')
        print('     with your actual key')
        print("=" * 60)
        return

    if divs is None:
        divs = list(SPORT_MAP.keys())

    print(f"Fetching fixtures + Bet365 odds (next {days_ahead} days)...\n")
    new_df = fetch_all_fixtures(divs, days_ahead)

    if new_df.empty:
        print("\nNo fixtures found.")
        return

    os.makedirs("data", exist_ok=True)
    merged = merge_fixtures(new_df)

    # Fetch weather forecast for each upcoming fixture
    merged = fetch_forecast_weather(merged)

    merged.to_csv(FIXTURES_PATH, index=False)

    print(f"\n{'='*65}")
    print(f"  {len(merged)} fixtures → {FIXTURES_PATH}")
    print(f"{'='*65}")
    for div, grp in merged.groupby("Div"):
        print(f"\n  {DIV_NAMES.get(div, div)} ({div})")
        for _, r in grp.iterrows():
            has_o = pd.notna(r.get("OddsH"))
            odds  = (f"H={r['OddsH']:.2f} D={r['OddsD']:.2f} A={r['OddsA']:.2f}"
                     if has_o else "no odds yet")
            print(f"    {r['Date']}  {r['HomeTeam']:<22} vs "
                  f"{r['AwayTeam']:<22}  {odds}")

    print(f"\nNow run: python part7_predict.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--div",     nargs="*", default=None)
    parser.add_argument("--days",    type=int,  default=14)
    parser.add_argument("--results", action="store_true")
    args = parser.parse_args()
    main(divs=args.div, days_ahead=args.days, score=args.results)