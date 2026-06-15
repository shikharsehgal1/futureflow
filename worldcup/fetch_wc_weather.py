#!/usr/bin/env python3
"""
Build data/wc_weather.csv for all 71 World Cup 2026 group-stage matches.

Mirrors the club model's weather feature in
preliminarymodel/fetch_fixtures.py:
  - REUSES the exact weather_goal_adjustment(temp, wind, precip) formula.
  - REUSES the Open-Meteo forecast API (https://api.open-meteo.com/v1/forecast,
    free, no key), daily=precipitation_sum,wind_speed_10m_max,temperature_2m_mean,
    wind_speed_unit=kmh, timezone=auto.

The Odds API feed (wc_match_summary.csv) has no venue, so we map each match to
its host city/stadium from the official 2026 FIFA World Cup schedule
(team-pairing -> venue), then look up that venue's hard-coded lat/lon.

Raw Open-Meteo JSON responses are cached under data/.weather_cache/ keyed by
lat,lon,date so re-runs do not refetch.
"""

import json
import os
import time
import unicodedata

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SUMMARY_CSV = os.path.join(DATA, "wc_match_summary.csv")
OUT_CSV = os.path.join(DATA, "wc_weather.csv")
CACHE_DIR = os.path.join(DATA, ".weather_cache")


# ----------------------------------------------------------------------------
# Weather adjustment — copied verbatim from preliminarymodel/fetch_fixtures.py
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# 16 host venues -> (stadium, city, lat, lon)
# ----------------------------------------------------------------------------
VENUES = {
    "Mexico City":       ("Estadio Azteca",          "Mexico City",          19.3030, -99.1505),
    "Guadalajara":       ("Estadio Akron",           "Guadalajara",          20.6817, -103.4628),
    "Monterrey":         ("Estadio BBVA",            "Monterrey",            25.6694, -100.2444),
    "Toronto":           ("BMO Field",               "Toronto",              43.6332, -79.4185),
    "Vancouver":         ("BC Place",                "Vancouver",            49.2768, -123.1119),
    "Atlanta":           ("Mercedes-Benz Stadium",   "Atlanta",              33.7554, -84.4008),
    "Boston":            ("Gillette Stadium",        "Boston (Foxborough)",  42.0909, -71.2643),
    "Dallas":            ("AT&T Stadium",            "Dallas (Arlington)",   32.7473, -97.0945),
    "Houston":           ("NRG Stadium",             "Houston",              29.6847, -95.4107),
    "Kansas City":       ("Arrowhead Stadium",       "Kansas City",          39.0489, -94.4839),
    "Los Angeles":       ("SoFi Stadium",            "Los Angeles (Inglewood)", 33.9535, -118.3392),
    "Miami":             ("Hard Rock Stadium",       "Miami (Miami Gardens)", 25.9580, -80.2389),
    "New York/New Jersey": ("MetLife Stadium",       "New York/New Jersey",  40.8136, -74.0744),
    "Philadelphia":      ("Lincoln Financial Field", "Philadelphia",         39.9008, -75.1675),
    "SF Bay Area":       ("Levi's Stadium",          "San Francisco Bay Area", 37.4030, -121.9700),
    "Seattle":           ("Lumen Field",             "Seattle",              47.5952, -122.3316),
}


# ----------------------------------------------------------------------------
# Match (home, away) -> host city. Pairings from official 2026 FIFA WC schedule.
# Teams normalized to match wc_match_summary.csv naming.
# ----------------------------------------------------------------------------
MATCH_CITY = {
    ("South Korea", "Czech Republic"):           "Guadalajara",
    ("Canada", "Bosnia & Herzegovina"):          "Toronto",
    ("USA", "Paraguay"):                          "Los Angeles",
    ("Qatar", "Switzerland"):                     "SF Bay Area",
    ("Brazil", "Morocco"):                        "New York/New Jersey",
    ("Haiti", "Scotland"):                        "Boston",
    ("Australia", "Turkey"):                      "Vancouver",
    ("Germany", "Curaçao"):                       "Houston",
    ("Netherlands", "Japan"):                     "Dallas",
    ("Ivory Coast", "Ecuador"):                   "Philadelphia",
    ("Sweden", "Tunisia"):                        "Monterrey",
    ("Spain", "Cape Verde"):                      "Atlanta",
    ("Belgium", "Egypt"):                         "Seattle",
    ("Saudi Arabia", "Uruguay"):                  "Miami",
    ("Iran", "New Zealand"):                      "Los Angeles",
    ("France", "Senegal"):                        "New York/New Jersey",
    ("Iraq", "Norway"):                           "Boston",
    ("Argentina", "Algeria"):                     "Kansas City",
    ("Austria", "Jordan"):                        "SF Bay Area",
    ("Portugal", "DR Congo"):                     "Houston",
    ("England", "Croatia"):                       "Dallas",
    ("Ghana", "Panama"):                          "Toronto",
    ("Uzbekistan", "Colombia"):                   "Mexico City",
    ("Czech Republic", "South Africa"):           "Atlanta",
    ("Switzerland", "Bosnia & Herzegovina"):      "Los Angeles",
    ("Canada", "Qatar"):                          "Vancouver",
    ("Mexico", "South Korea"):                    "Guadalajara",
    ("USA", "Australia"):                         "Seattle",
    ("Scotland", "Morocco"):                      "Boston",
    ("Brazil", "Haiti"):                          "Philadelphia",
    ("Turkey", "Paraguay"):                       "SF Bay Area",
    ("Netherlands", "Sweden"):                    "Houston",
    ("Germany", "Ivory Coast"):                   "Toronto",
    ("Ecuador", "Curaçao"):                       "Kansas City",
    ("Tunisia", "Japan"):                         "Monterrey",
    ("Spain", "Saudi Arabia"):                    "Atlanta",
    ("Belgium", "Iran"):                          "Los Angeles",
    ("Uruguay", "Cape Verde"):                    "Miami",
    ("New Zealand", "Egypt"):                     "Vancouver",
    ("Argentina", "Austria"):                     "Dallas",
    ("France", "Iraq"):                           "Philadelphia",
    ("Norway", "Senegal"):                        "New York/New Jersey",
    ("Jordan", "Algeria"):                        "SF Bay Area",
    ("Portugal", "Uzbekistan"):                   "Houston",
    ("England", "Ghana"):                         "Boston",
    ("Panama", "Croatia"):                        "Toronto",
    ("Colombia", "DR Congo"):                     "Guadalajara",
    ("Bosnia & Herzegovina", "Qatar"):            "Seattle",
    ("Switzerland", "Canada"):                    "Vancouver",
    ("Scotland", "Brazil"):                       "Miami",
    ("Morocco", "Haiti"):                         "Atlanta",
    ("Czech Republic", "Mexico"):                 "Mexico City",
    ("South Africa", "South Korea"):              "Monterrey",
    ("Curaçao", "Ivory Coast"):                   "Philadelphia",
    ("Ecuador", "Germany"):                       "New York/New Jersey",
    ("Japan", "Sweden"):                          "Dallas",
    ("Tunisia", "Netherlands"):                   "Kansas City",
    ("Paraguay", "Australia"):                    "SF Bay Area",
    ("Turkey", "USA"):                            "Los Angeles",
    ("Norway", "France"):                         "Boston",
    ("Senegal", "Iraq"):                          "Toronto",
    ("Egypt", "Iran"):                            "Seattle",
    ("New Zealand", "Belgium"):                   "Vancouver",
    ("Cape Verde", "Saudi Arabia"):               "Houston",
    ("Uruguay", "Spain"):                         "Guadalajara",
    ("Panama", "England"):                        "New York/New Jersey",
    ("Croatia", "Ghana"):                         "Philadelphia",
    ("Algeria", "Austria"):                       "Kansas City",
    ("Jordan", "Argentina"):                      "Dallas",
    ("Colombia", "Portugal"):                     "Miami",
    ("DR Congo", "Uzbekistan"):                   "Atlanta",
}


def split_match(name):
    """'A vs B' -> ('A', 'B')."""
    h, a = name.split(" vs ")
    return h.strip(), a.strip()


def local_date_for(lat, lon, commence_iso):
    """
    Convert match UTC commence time to the venue's local calendar date.
    We don't have a tz library guaranteed, so approximate with a longitude
    offset (15deg per hour). Open-Meteo uses timezone=auto for the forecast
    itself; this date-bucketing just needs to land on the right local day,
    and a longitude estimate is accurate to within an hour everywhere here.
    """
    dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    offset_hours = round(lon / 15.0)
    local_dt = dt + timedelta(hours=offset_hours)
    return local_dt.date().isoformat()


def cached_fetch(session, lat, lon, date_str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{lat:.4f}_{lon:.4f}_{date_str}.json"
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f), True

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,wind_speed_10m_max,temperature_2m_mean",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    with open(path, "w") as f:
        json.dump(data, f)
    time.sleep(0.25)
    return data, False


def main():
    df = pd.read_csv(SUMMARY_CSV)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    rows = []
    n_forecast = 0
    n_fallback = 0
    unmapped = []

    for _, m in df.iterrows():
        home, away = split_match(m["match"])
        city_key = MATCH_CITY.get((home, away))
        if city_key is None:
            unmapped.append(m["match"])
            stadium, city, lat, lon = ("UNKNOWN", "UNKNOWN", None, None)
        else:
            stadium, city, lat, lon = VENUES[city_key]

        if lat is None:
            rows.append({
                "match_id": m["match_id"], "match": m["match"],
                "venue": "UNKNOWN", "city": "UNKNOWN",
                "date": str(m["commence"])[:10],
                "lat": "", "lon": "",
                "temp_c": "", "wind_kmh": "", "precip_mm": "",
                "weather_adj": 1.0,
            })
            n_fallback += 1
            continue

        date_str = local_date_for(lat, lon, m["commence"])

        temp = wind = prec = None
        adj = 1.0
        try:
            data, _ = cached_fetch(session, lat, lon, date_str)
            daily = data.get("daily", {})
            prec = (daily.get("precipitation_sum") or [None])[0]
            wind = (daily.get("wind_speed_10m_max") or [None])[0]
            temp = (daily.get("temperature_2m_mean") or [None])[0]
            if temp is None and wind is None and prec is None:
                # date outside forecast window -> leave adj at 1.0
                n_fallback += 1
            else:
                adj = weather_goal_adjustment(temp, wind, prec)
                n_forecast += 1
        except Exception as e:
            print(f"  fetch failed for {m['match']} ({city}): {e}")
            n_fallback += 1

        rows.append({
            "match_id": m["match_id"],
            "match": m["match"],
            "venue": stadium,
            "city": city,
            "date": date_str,
            "lat": lat,
            "lon": lon,
            "temp_c": "" if temp is None else round(float(temp), 1),
            "wind_kmh": "" if wind is None else round(float(wind), 1),
            "precip_mm": "" if prec is None else round(float(prec), 2),
            "weather_adj": adj,
        })

    out = pd.DataFrame(rows, columns=[
        "match_id", "match", "venue", "city", "date",
        "lat", "lon", "temp_c", "wind_kmh", "precip_mm", "weather_adj",
    ])
    out.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {OUT_CSV}")
    print(f"Total matches:        {len(out)}")
    print(f"Real forecasts:       {n_forecast}")
    print(f"Fallback (1.0):       {n_fallback}")
    if unmapped:
        print(f"UNMAPPED matches ({len(unmapped)}): {unmapped}")

    print("\nMost goal-suppressing (lowest weather_adj):")
    show = out[out["weather_adj"] < 1.0].sort_values("weather_adj").head(15)
    for _, r in show.iterrows():
        print(f"  {r['weather_adj']:.4f}  {r['match']:<40} "
              f"{r['city']:<22} t={r['temp_c']}C w={r['wind_kmh']}kmh p={r['precip_mm']}mm")


if __name__ == "__main__":
    main()
