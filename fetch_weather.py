"""
fetch_weather.py — Fetch historical match-day weather for all Big 5 games

Uses Open-Meteo historical API (free, no API key needed) to get:
  - precipitation_sum (mm) — rain/snow on matchday
  - wind_speed_10m_max (km/h) — max wind speed
  - temperature_2m_mean (°C) — average temperature

Results saved to data/weather.csv and merged into big5_with_probs.csv
as control variables for the ratings model.

Usage:
    python3.10 fetch_weather.py           # fetch all missing games
    python3.10 fetch_weather.py --merge   # merge into big5_with_probs.csv
    python3.10 fetch_weather.py --div E0  # specific league only
"""

import os, time, argparse, json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

WEATHER_PATH = "data/weather.csv"
DATA_PATH    = "data/big5_with_probs.csv"
CACHE_PATH   = "data/weather_cache.json"
SLEEP        = 0.15   # Open-Meteo allows ~10k req/day free

# ── STADIUM COORDINATES ───────────────────────────────────────────────────────
# lat/lon for each team's home ground
# Historical moves noted where relevant
# For B-teams and reserves, uses parent club's stadium

STADIUMS = {
    # ── PREMIER LEAGUE / CHAMPIONSHIP ────────────────────────────────────────
    "Arsenal":         (51.5549,  -0.1084),   # Emirates Stadium
    "Aston Villa":     (52.5092,  -1.8847),   # Villa Park
    "Barnsley":        (53.5527,  -1.4674),   # Oakwell
    "Birmingham":      (52.4752,  -1.7865),   # St Andrew's
    "Blackburn":       (53.7286,  -2.4890),   # Ewood Park
    "Blackpool":       (53.8043,  -3.0480),   # Bloomfield Road
    "Bolton":          (53.5805,  -2.5357),   # University of Bolton Stadium
    "Bournemouth":     (50.7352,  -1.8383),   # Vitality Stadium
    "Brentford":       (51.4882,  -0.3088),   # Gtech Community Stadium
    "Brighton":        (50.8618,  -0.0837),   # Amex Stadium
    "Bristol City":    (51.4400,  -2.6200),   # Ashton Gate
    "Burnley":         (53.7893,  -2.2297),   # Turf Moor
    "Cardiff":         (51.4728,  -3.2030),   # Cardiff City Stadium
    "Charlton":        (51.4865,   0.0366),   # The Valley
    "Chelsea":         (51.4816,  -0.1910),   # Stamford Bridge
    "Coventry":        (52.4484,  -1.5576),   # Coventry Building Society Arena
    "Crystal Palace":  (51.3983,  -0.0855),   # Selhurst Park
    "Derby":           (52.9151,  -1.4472),   # Pride Park
    "Everton":         (53.4388,  -2.9661),   # Goodison Park
    "Fulham":          (51.4749,  -0.2217),   # Craven Cottage
    "Huddersfield":    (53.6543,  -1.7679),   # John Smith's Stadium
    "Hull":            (53.7460,  -0.3675),   # MKM Stadium
    "Ipswich":         (52.0551,   1.1449),   # Portman Road
    "Leeds":           (53.7775,  -1.5724),   # Elland Road
    "Leicester":       (52.6204,  -1.1422),   # King Power Stadium
    "Liverpool":       (53.4308,  -2.9608),   # Anfield
    "Luton":           (51.8837,  -0.4316),   # Kenilworth Road
    "Man City":        (53.4831,  -2.2004),   # Etihad Stadium
    "Man United":      (53.4631,  -2.2913),   # Old Trafford
    "Middlesbrough":   (54.5782,  -1.2175),   # Riverside Stadium
    "Millwall":        (51.4858,  -0.0508),   # The Den
    "Newcastle":       (54.9756,  -1.6217),   # St James' Park
    "Norwich":         (52.6221,   1.3093),   # Carrow Road
    "Nott'm Forest":   (52.9399,  -1.1328),   # City Ground
    "Oxford":          (51.7174,  -1.2149),   # Kassam Stadium
    "Plymouth":        (50.3881,  -4.1236),   # Home Park
    "Portsmouth":      (50.7965,  -1.0641),   # Fratton Park
    "Preston":         (53.7729,  -2.6911),   # Deepdale
    "QPR":             (51.5093,  -0.2321),   # Loftus Road
    "Reading":         (51.4528,  -0.9866),   # Select Car Leasing Stadium
    "Rotherham":       (53.4275,  -1.3632),   # AESSEAL New York Stadium
    "Sheffield United":(53.3703,  -1.4706),   # Bramall Lane
    "Sheffield Weds":  (53.4114,  -1.5007),   # Hillsborough
    "Southampton":     (50.9058,  -1.3914),   # St Mary's Stadium
    "Stoke":           (52.9883,  -2.1752),   # bet365 Stadium
    "Sunderland":      (54.9145,  -1.3879),   # Stadium of Light
    "Swansea":         (51.6428,  -3.9344),   # Swansea.com Stadium
    "Tottenham":       (51.6042,  -0.0665),   # Tottenham Hotspur Stadium
    "Watford":         (51.6498,  -0.4017),   # Vicarage Road
    "West Brom":       (52.5090,  -1.9642),   # The Hawthorns
    "West Ham":        (51.5386,   0.0164),   # London Stadium
    "Wigan":           (53.5488,  -2.6372),   # DW Stadium
    "Wolves":          (52.5900,  -2.1303),   # Molineux
    "Wrexham":         (53.0463,  -3.0041),   # Racecourse Ground

    # ── BUNDESLIGA / 2. BUNDESLIGA ───────────────────────────────────────────
    "Augsburg":        (48.3232,  10.8854),   # WWK Arena
    "Bayern Munich":   (48.2188,  11.6247),   # Allianz Arena
    "Bielefeld":       (51.9997,   8.5248),   # SchücoArena
    "Bochum":          (51.4900,   7.2376),   # Vonovia Ruhrstadion
    "Braunschweig":    (52.2733,  10.5251),   # Eintracht-Stadion
    "Darmstadt":       (49.8728,   8.6530),   # Merck-Stadion
    "Dortmund":        (51.4926,   7.4517),   # Signal Iduna Park
    "Dresden":         (51.0375,  13.7595),   # Rudolf-Harbig-Stadion
    "Duisburg":        (51.4988,   6.7737),   # Schauinsland-Reisen-Arena
    "Ein Frankfurt":   (50.0687,   8.6453),   # Deutsche Bank Park
    "Elversberg":      (49.3470,   7.0920),   # Ursapharm-Arena
    "Erzgebirge Aue":  (50.5771,  12.7113),   # Erzgebirgsstadion
    "FC Koln":         (50.9336,   6.8750),   # RheinEnergieStadion
    "Fortuna Dusseldorf":(51.2738, 6.7933),   # Merkur Spiel-Arena
    "Freiburg":        (47.9879,   7.8955),   # Europa-Park Stadion
    "Greuther Furth":  (49.4812,  10.9917),   # Sportpark Ronhof
    "Hamburg":         (53.5872,  10.0182),   # Volksparkstadion
    "Hannover":        (52.3606,   9.7285),   # Heinz von Heiden Arena
    "Heidenheim":      (48.6766,  10.1551),   # Voith-Arena
    "Hertha":          (52.5147,  13.2394),   # Olympiastadion Berlin
    "Hoffenheim":      (49.2380,   8.8892),   # PreZero Arena
    "Holstein Kiel":   (54.3523,  10.1332),   # Holstein-Stadion
    "Ingolstadt":      (48.7600,  11.4200),   # Audi Sportpark
    "Jahn Regensburg": (49.0167,  12.0833),   # Jahnstadion
    "Kaiserslautern":  (49.4345,   7.7778),   # Fritz Walter Stadion
    "Karlsruhe":       (49.0244,   8.4147),   # BBBank Wildpark
    "Leverkusen":      (51.0384,   7.0023),   # BayArena
    "Magdeburg":       (52.1350,  11.6200),   # MDCC-Arena
    "Mainz":           (49.9843,   8.2242),   # MEWA Arena
    "M'gladbach":      (51.1742,   6.3853),   # Borussia-Park
    "Nurnberg":        (49.4269,  11.1228),   # Max-Morlock-Stadion
    "Paderborn":       (51.7290,   8.7538),   # Benteler Arena
    "Preuss Muenster": (51.9769,   7.6440),   # Preußenstadion
    "RB Leipzig":      (51.3457,  12.3474),   # Red Bull Arena
    "St Pauli":        (53.5547,   9.9682),   # Millerntor-Stadion
    "Stuttgart":       (48.7925,   9.2325),   # MHPArena
    "Union Berlin":    (52.4575,  13.5675),   # An der Alten Försterei
    "Werder Bremen":   (53.0663,   8.8375),   # wohninvest WESERSTADION
    "Wolfsburg":       (52.4344,  10.8032),   # Volkswagen Arena

    # ── LA LIGA / SEGUNDA DIVISIÓN ───────────────────────────────────────────
    "Alaves":          (42.8530,  -2.6819),   # Mendizorroza
    "Albacete":        (38.9948,  -1.8514),   # Estadio Carlos Belmonte
    "Alcorcon":        (40.3436,  -3.8257),   # Estadio Santo Domingo
    "Almeria":         (36.8376,  -2.4500),   # Power Horse Stadium
    "Amorebieta":      (43.2221,  -2.7360),   # Urritxe
    "Andorra":         (42.5000,   1.5218),   # Estadi Nacional
    "Ath Bilbao":      (43.2642,  -2.9494),   # San Mamés
    "Ath Bilbao B":    (43.2642,  -2.9494),
    "Ath Madrid":      (40.4361,  -3.5996),   # Riyadh Air Metropolitano
    "Barcelona":       (41.3809,   2.1228),   # Spotify Camp Nou
    "Barcelona B":     (41.3809,   2.1228),
    "Betis":           (37.3562,  -5.9822),   # Estadio Benito Villamarín
    "Burgos":          (42.3440,  -3.6970),   # El Plantío
    "Cadiz":           (36.5338,  -6.2927),   # Nuevo Mirandilla
    "Castellon":       (39.9500,  -0.0495),   # Estadio Castalia
    "Ceuta":           (35.8894,  -5.3213),   # Alfonso Murube
    "Celta":           (42.2117,  -8.7388),   # Abanca-Balaídos
    "Cordoba":         (37.8768,  -4.7862),   # Estadio El Arcángel
    "Cultural Leonesa":(42.5987,  -5.5671),   # Reino de León
    "Eibar":           (43.1849,  -2.4729),   # Ipurua Municipal Stadium
    "Elche":           (38.2585,  -0.7032),   # Estadio Martínez Valero
    "Espanol":         (41.3479,   2.0750),   # RCDE Stadium
    "Getafe":          (40.3244,  -3.7142),   # Coliseum Alfonso Pérez
    "Girona":          (41.9618,   2.8270),   # Estadio Municipal de Montilivi
    "Granada":         (37.1726,  -3.5970),   # Nuevo Los Cármenes
    "Huesca":          (42.1467,  -0.4165),   # El Alcoraz
    "Las Palmas":      (28.1003, -15.4503),   # Estadio Gran Canaria
    "Leganes":         (40.3524,  -3.7638),   # Estadio Municipal de Butarque
    "Levante":         (39.4927,  -0.3480),   # Estadio Ciudad de Valencia
    "Lugo":            (43.0093,  -7.5554),   # Estadio Anxo Carro
    "Malaga":          (36.7107,  -4.4283),   # La Rosaleda
    "Mallorca":        (39.5896,   2.6502),   # Estadi de Son Moix
    "Mirandes":        (42.6851,  -2.9441),   # Estadio Municipal de Anduva
    "Osasuna":         (42.7968,  -1.6376),   # El Sadar
    "Oviedo":          (43.3566,  -5.8456),   # Carlos Tartiere
    "Ponferradina":    (42.5462,  -6.5964),   # Estadio El Toralin
    "Racing Santander":(43.4659,  -3.8264),   # Campos de Sport de El Sardinero
    "Santander":       (43.4659,  -3.8264),
    "La Coruna":       (43.3333,  -8.4167),   # Estadio Abanca-Riazor
    "Real Madrid":     (40.4531,  -3.6883),   # Estadio Santiago Bernabéu
    "Rayo Vallecano":  (40.3914,  -3.6561),
    "Vallecano":       (40.3914,  -3.6561),   # Estadio de Vallecas
    "Sevilla":         (37.3840,  -5.9706),   # Estadio Ramón Sánchez-Pizjuán
    "Sociedad":        (43.3017,  -1.9742),   # Reale Arena
    "Sp Gijon":        (43.5333,  -5.6333),   # Estadio El Molinón
    "Tenerife":        (28.4686, -16.2546),   # Estadio Heliodoro Rodríguez
    "Valencia":        (39.4747,  -0.3586),   # Estadio Mestalla
    "Valladolid":      (41.6430,  -4.7575),   # Estadio José Zorrilla
    "Villarreal":      (39.9444,  -0.1031),   # Estadio de la Cerámica
    "Zaragoza":        (41.6474,  -0.9116),   # La Romareda
    "Sociedad B":      (43.3017,  -1.9742),
    "Ceuta":           (35.8894,  -5.3213),
    "Alcorcon":        (40.3436,  -3.8257),

    # ── SERIE A / SERIE B ────────────────────────────────────────────────────
    "Atalanta":        (45.7091,   9.6796),   # Gewiss Stadium
    "Avellino":        (40.9148,  14.7794),   # Stadio Partenio-Adriano Lombardi
    "Bari":            (41.0806,  16.8683),   # Stadio San Nicola
    "Benevento":       (41.1139,  14.8031),   # Stadio Ciro Vigorito
    "Bologna":         (44.4924,  11.3094),   # Stadio Renato Dall'Ara
    "Brescia":         (45.5381,  10.2179),   # Stadio Mario Rigamonti
    "Cagliari":        (39.2000,   9.1340),   # Unipol Domus
    "Carpi":           (44.7838,  10.8801),   # Stadio Sandro Cabassi
    "Carrarese":       (44.0833,  10.0833),   # Stadio dei Marmi
    "Catanzaro":       (38.9000,  16.5870),   # Stadio Nicola Ceravolo
    "Cesena":          (44.1333,  12.2333),   # Stadio Dino Manuzzi
    "Chievo":          (45.4343,  10.9938),
    "Como":            (45.8081,   9.0852),   # Stadio Giuseppe Sinigaglia
    "Cosenza":         (39.3000,  16.2500),   # Stadio San Vito-Gigi Marulla
    "Cremonese":       (45.1333,  10.0333),   # Stadio Giovanni Zini
    "Crotone":         (39.0667,  17.1167),   # Stadio Ezio Scida
    "Empoli":          (43.7167,  10.9500),   # Stadio Carlo Castellani
    "Fiorentina":      (43.7805,  11.2828),   # Stadio Artemio Franchi
    "Frosinone":       (41.6333,  13.3167),   # Stadio Benito Stirpe
    "Genoa":           (44.4133,   8.9526),   # Stadio Luigi Ferraris
    "Inter":           (45.4781,   9.1240),   # San Siro
    "Juventus":        (45.1096,   7.6413),   # Allianz Stadium
    "Lazio":           (41.9340,  12.4548),   # Stadio Olimpico
    "Lecce":           (40.3667,  18.1833),   # Stadio Via del Mare
    "Mantova":         (45.1500,  10.7833),   # Stadio Danilo Martelli
    "Milan":           (45.4781,   9.1240),   # San Siro
    "Modena":          (44.6400,  10.9253),   # Stadio Alberto Braglia
    "Monza":           (45.5833,   9.2833),   # Stadio Brianteo
    "Napoli":          (40.8279,  14.1931),   # Stadio Diego Armando Maradona
    "Padova":          (45.3944,  11.8891),   # Stadio Euganeo
    "Palermo":         (38.1333,  13.3667),   # Stadio Renzo Barbera
    "Parma":           (44.7925,  10.3958),   # Stadio Ennio Tardini
    "Pescara":         (42.4500,  14.2000),   # Stadio Adriatico
    "Pisa":            (43.6936,  10.3885),   # Arena Garibaldi
    "Reggiana":        (44.7000,  10.6333),   # Mapei Stadium
    "Roma":            (41.9340,  12.4548),   # Stadio Olimpico
    "Salernitana":     (40.6780,  14.7614),   # Stadio Arechi
    "Sampdoria":       (44.4133,   8.9526),   # Stadio Luigi Ferraris
    "Sassuolo":        (44.7000,  10.6333),   # Mapei Stadium
    "Spal":            (44.8333,  11.6167),   # Stadio Paolo Mazza
    "Spezia":          (44.1000,   9.8333),   # Stadio Alberto Picco
    "Sudtirol":        (46.5000,  11.3500),   # Dreiländerarena
    "Torino":          (45.0408,   7.6500),   # Stadio Olimpico Grande Torino
    "Udinese":         (46.0792,  13.2006),   # Dacia Arena
    "venezia":         (45.4408,  12.3155),
    "Venezia":         (45.4408,  12.3155),   # Stadio Pier Luigi Penzo
    "Verona":          (45.4272,  10.9922),   # Stadio Marcantonio Bentegodi
    "Virtus Entella":  (44.3500,   9.3167),   # Stadio Comunale di Chiavari
    "Virtus Entella":  (44.3500,   9.3167),
    "Juve Stabia":     (40.7056,  14.4933),   # Stadio Romeo Menti

    # ── LIGUE 1 / LIGUE 2 ────────────────────────────────────────────────────
    "Ajaccio":         (41.9200,   8.7381),   # Stade François Coty
    "Ajaccio GFCO":    (41.9200,   8.7381),
    "Amiens":          (49.8941,   2.2831),   # Stade de la Licorne
    "Angers":          (47.4667,  -0.5500),   # Stade Raymond Kopa
    "Annecy":          (45.8992,   6.1294),   # Stade de Genève / Parc des Sports
    "Auxerre":         (47.7833,   3.5667),   # Stade Abbé-Deschamps
    "Bastia":          (42.6878,   9.4500),   # Stade Armand Cesari
    "Bordeaux":        (44.8417,  -0.5731),   # Matmut Atlantique
    "Boulogne":        (50.7200,   1.6167),   # Stade de la Libération
    "Brest":           (48.3900,  -4.4833),   # Stade Francis-Le Blé
    "Caen":            (49.1667,  -0.3667),   # Stade Michel d'Ornano
    "Clermont":        (45.7731,   3.1000),   # Stade Gabriel Montpied
    "Dijon":           (47.3167,   5.0500),   # Stade Gaston Gérard
    "Dunkerque":       (51.0333,   2.3667),   # Stade Marcel-Tribut
    "Grenoble":        (45.1667,   5.7167),   # Stade des Alpes
    "Guingamp":        (48.5608,  -3.1500),   # Stade du Roudourou
    "Laval":           (48.0667,  -0.7667),   # Stade Francis Le Basser
    "Le Havre":        (49.4833,   0.1000),   # Stade Océane
    "Le Mans":         (48.0039,   0.1956),   # MMArena
    "Lens":            (50.4333,   2.8167),   # Stade Bollaert-Delelis
    "Lille":           (50.6167,   3.1333),   # Stade Pierre-Mauroy
    "Lorient":         (47.7500,  -3.3667),   # Stade du Moustoir
    "Lyon":            (45.7654,   4.9822),   # Groupama Stadium
    "Marseille":       (43.2700,   5.3953),   # Stade Vélodrome
    "Metz":            (49.1100,   6.1778),   # Stade Saint-Symphorien
    "Monaco":          (43.7278,   7.4153),   # Stade Louis II
    "Montpellier":     (43.6222,   3.8128),   # Stade de la Mosson
    "Nancy":           (48.6833,   6.2000),   # Stade Marcel Picot
    "Nantes":          (47.2558,  -1.5247),   # Stade de la Beaujoire
    "Nice":            (43.7050,   7.2603),   # Allianz Riviera
    "Nimes":           (43.8381,   4.3806),   # Stade des Costières
    "Paris FC":        (48.8167,   2.3000),   # Stade Charléty
    "Paris SG":        (48.8414,   2.2530),   # Parc des Princes
    "Pau FC":          (43.3000,  -0.3667),   # Stade du Hameau
    "Reims":           (49.2389,   4.0531),   # Stade Auguste Delaune
    "Rennes":          (48.1078,  -1.7144),   # Roazhon Park
    "Rodez":           (44.3500,   2.5667),   # Stade Paul Lignon
    "Red Star":        (48.9136,   2.3628),   # Stade Bauer
    "Rodez":           (44.3500,   2.5667),
    "Sochaux":         (47.5000,   6.8167),   # Stade Auguste Bonal
    "St Etienne":      (45.4608,   4.3906),   # Stade Geoffroy-Guichard
    "Strasbourg":      (48.5628,   7.7519),   # Stade de la Meinau
    "Toulouse":        (43.5833,   1.4333),   # Stadium de Toulouse
    "Troyes":          (48.2956,   4.0667),   # Stade de l'Aube
    "Valenciennes":    (50.3667,   3.5167),   # Stade du Hainaut
}

# City-level fallbacks for any missing teams
CITY_FALLBACK = {
    "Beziers":         (43.3440,   3.2150),
    "Niort":           (46.3278,  -0.4603),
    "Orleans":         (47.9029,   1.9039),
    "Tours":           (47.3941,   0.6848),
    "Caen":            (49.1667,  -0.3667),
    "Bari":            (41.0806,  16.8683),
    "Novara":          (45.4500,   8.6167),
    "Reggina":         (38.1136,  15.6476),
    "Triestina":       (45.6495,  13.7768),
    "Craiova":         (44.3333,  23.8167),
}


def get_coords(team):
    """Get lat/lon for a team, with city fallback."""
    if team in STADIUMS:
        return STADIUMS[team]
    if team in CITY_FALLBACK:
        return CITY_FALLBACK[team]
    # Last resort: try stripping common suffixes
    for suffix in [" FC", " CF", " SC", " AC", " SS", " AS"]:
        t2 = team.replace(suffix, "").strip()
        if t2 in STADIUMS:
            return STADIUMS[t2]
    return None


def fetch_weather_day(lat, lon, date_str, session, cache):
    """Fetch weather for one location and date, using cache."""
    key = f"{lat:.3f},{lon:.3f},{date_str}"
    if key in cache:
        return cache[key]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":           lat,
        "longitude":          lon,
        "start_date":         date_str,
        "end_date":           date_str,
        "daily":              "precipitation_sum,wind_speed_10m_max,temperature_2m_mean",
        "timezone":           "auto",
        "wind_speed_unit":    "kmh",
    }
    try:
        r = session.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})
        result = {
            "precipitation":  daily.get("precipitation_sum",  [None])[0],
            "wind_speed":     daily.get("wind_speed_10m_max", [None])[0],
            "temperature":    daily.get("temperature_2m_mean",[None])[0],
        }
        cache[key] = result
        return result
    except Exception as e:
        return {"precipitation": None, "wind_speed": None, "temperature": None}


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def main(divs=None, merge=False):
    print("Loading match data...", flush=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if divs:
        df = df[df["Div"].isin(divs)]
        print(f"  Filtered to: {divs}")

    # Load existing weather data
    if os.path.exists(WEATHER_PATH):
        weather_df = pd.read_csv(WEATHER_PATH)
        done_keys  = set(weather_df["key"])
    else:
        weather_df = pd.DataFrame()
        done_keys  = set()

    # Load cache
    cache = load_cache()

    # Find games needing weather
    df["date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["key"]      = df["HomeTeam"] + "|" + df["date_str"]
    missing        = df[~df["key"].isin(done_keys)].copy()

    # Deduplicate by (HomeTeam, date_str) — same stadium same day
    missing_unique = missing.drop_duplicates(subset=["HomeTeam","date_str"])
    print(f"  {len(done_keys)} games already fetched")
    print(f"  {len(missing_unique)} unique (team, date) combos to fetch")

    session  = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research)"})
    new_rows = []
    skipped  = 0

    for i, (_, row) in enumerate(missing_unique.iterrows()):
        coords = get_coords(row["HomeTeam"])
        if coords is None:
            skipped += 1
            if skipped <= 5 or skipped % 20 == 0:
                print(f"  WARNING: no coords for '{row['HomeTeam']}'")
            continue

        lat, lon = coords
        result   = fetch_weather_day(lat, lon, row["date_str"], session, cache)

        new_rows.append({
            "key":           row["key"],
            "HomeTeam":      row["HomeTeam"],
            "date_str":      row["date_str"],
            "lat":           lat,
            "lon":           lon,
            "precipitation": result["precipitation"],
            "wind_speed":    result["wind_speed"],
            "temperature":   result["temperature"],
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(missing_unique)} fetched...", flush=True)
            save_cache(cache)

        time.sleep(SLEEP)

    save_cache(cache)

    # Save weather data
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        weather_df = pd.concat([weather_df, new_df], ignore_index=True)
        weather_df = weather_df.drop_duplicates(subset=["key"])
        weather_df.to_csv(WEATHER_PATH, index=False)
        print(f"\nSaved {len(weather_df)} weather records → {WEATHER_PATH}")
        if skipped:
            print(f"Skipped {skipped} games (no stadium coords)")

    # ── Merge into big5_with_probs.csv ────────────────────────────────────────
    if merge:
        print("\nMerging weather into big5_with_probs.csv...")
        df_main   = pd.read_csv(DATA_PATH, low_memory=False)
        df_main["Date"] = pd.to_datetime(df_main["Date"], errors="coerce")
        df_main["date_str"] = df_main["Date"].dt.strftime("%Y-%m-%d")
        df_main["key"]      = df_main["HomeTeam"] + "|" + df_main["date_str"]

        w = pd.read_csv(WEATHER_PATH)[["key","precipitation","wind_speed","temperature"]]
        merged = df_main.merge(w, on="key", how="left")
        merged = merged.drop(columns=["date_str","key"])
        merged.to_csv(DATA_PATH, index=False)
        coverage = merged["wind_speed"].notna().mean()
        print(f"Done. Weather coverage: {coverage:.1%}")
        print(f"Saved → {DATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--div",   nargs="*", default=None)
    parser.add_argument("--merge", action="store_true",
                        help="Merge weather into big5_with_probs.csv after fetching")
    args = parser.parse_args()
    main(args.div, args.merge)