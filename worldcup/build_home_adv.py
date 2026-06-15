#!/usr/bin/env python3
"""
build_home_adv.py
=================

Per-match "effective home advantage" index for the 2026 World Cup, driven by
diaspora / crowd composition.

The 2026 WC is hosted across the USA, Canada and Mexico. Venues are nominally
"neutral", but host-city diaspora communities (and host-nation home matches)
create real home-crowd atmospheres. This script quantifies that as a small,
conservative goal-supremacy adjustment for the FUNDAMENTALS / MODEL layer.

IMPORTANT (model usage note)
----------------------------
`net_home_adv_goals` is intended to be ADDED to the home team's expected goal
supremacy in a fundamentals (Poisson/Dixon-Coles style) model ONLY.

It must NOT be added on top of de-vigged market odds: the betting market
already prices crowd / diaspora effects, so doing so would double-count.

Method (transparent + conservative)
------------------------------------
1. For each team in a match we estimate `home_support` in [0, 1]:

   support = base_proximity + diaspora_boost   (clamped to [0, 1])

   - base_proximity: a host nation playing at home in its own country gets a
     strong proximity term. (Mexico in Mexico, USA in the USA, Canada in Canada.)
   - diaspora_boost: a non-host (or travelling) team gets a partial boost when
     the venue city hosts a large, football-passionate diaspora for that nation.

   Support tiers used for diaspora_boost (city-specific, see CITY_DIASPORA):
       0.55  dominant local diaspora (e.g. Mexico in LA/Houston/Dallas)
       0.40  very strong diaspora
       0.30  strong diaspora
       0.20  notable diaspora
       0.10  modest diaspora
       0.00  negligible

2. Convert the support DIFFERENTIAL into goals:

       net = SUPPORT_TO_GOALS * (home_support - away_support)

   with SUPPORT_TO_GOALS chosen so that:
       - a true home crowd (support diff ~1.0) -> ~+0.35 goals
       - a strong diaspora edge (diff ~0.4-0.55) -> ~+0.10 to +0.20 goals
       - balanced / neutral -> ~0

   The result is clamped to [-0.35, +0.35]. `net` can be negative when the
   nominal away team carries the bigger crowd.

The mapping of "home"/"away" follows the fixture string "Home vs Away" exactly,
so `net_home_adv_goals` is always expressed relative to the listed home team.
"""

import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHER_CSV = os.path.join(HERE, "data", "wc_weather.csv")
OUT_CSV = os.path.join(HERE, "data", "home_advantage.csv")

# Goals per unit of support differential. diff of ~1.0 (true home crowd vs an
# away side with no support) -> 0.35 goals.
SUPPORT_TO_GOALS = 0.35
CAP = 0.35

# Host nations -> proximity support when playing at home in their own country.
# Mexico's three host cities are a genuine home advantage; the USA hosts the
# bulk of matches; Canada hosts in Toronto/Vancouver.
HOST_NATION_HOME_SUPPORT = {
    "Mexico": 1.00,   # playing in Mexico = true home
    "USA": 0.95,      # USA hosting at home
    "Canada": 0.90,   # Canada hosting at home
}

# City -> which host country it is in. Drives the host-nation proximity term and
# a small generic "host familiarity" edge for the host nation even in away venues.
US_CITIES = {
    "Los Angeles (Inglewood)", "San Francisco Bay Area", "New York/New Jersey",
    "Boston (Foxborough)", "Houston", "Dallas (Arlington)", "Philadelphia",
    "Atlanta", "Seattle", "Miami (Miami Gardens)", "Kansas City",
}
CANADA_CITIES = {"Toronto", "Vancouver"}
MEXICO_CITIES = {"Guadalajara", "Monterrey", "Mexico City"}


def city_country(city: str) -> str:
    if city in MEXICO_CITIES:
        return "Mexico"
    if city in CANADA_CITIES:
        return "Canada"
    if city in US_CITIES:
        return "USA"
    return "USA"  # all listed venues are US/CA/MX; default safe


# ---------------------------------------------------------------------------
# City-specific diaspora support for travelling (non-host) nations.
# Values are diaspora_boost in [0, 1]. Grounded in metro demographics (see
# sources in the accompanying report). Conservative by design.
#
# Tiers: 0.55 dominant | 0.40 very strong | 0.30 strong | 0.20 notable |
#        0.10 modest
# ---------------------------------------------------------------------------
CITY_DIASPORA = {
    # --- USA: heavy Mexican/Hispanic metros (Mexico already handled as host) ---
    "Los Angeles (Inglewood)": {
        # Tehrangeles: largest Iranian community outside Iran (~230k metro)
        "Iran": 0.40,
        # Large Central/South American + general Latino crowd
        "Paraguay": 0.10,
    },
    "Houston": {
        # Large Central American + Colombian/Salvadoran etc.; football-mad Latino city
        "DR Congo": 0.05,
        "Colombia": 0.20,
        "Uzbekistan": 0.05,
    },
    "Dallas (Arlington)": {
        # Heavy Hispanic metro; some Croatian/European but small
        "Croatia": 0.10,
    },
    "Miami (Miami Gardens)": {
        # The diaspora capital: Cuban 1.1M+, Colombian, Venezuelan, Haitian 335k metro,
        # Brazilian, broad South American.
        "Colombia": 0.55,
        "Brazil": 0.40,
        "Haiti": 0.40,
        "Portugal": 0.15,   # Lusophone overlap + Brazilian-leaning neutral crowd
        "Uruguay": 0.25,
        "Cape Verde": 0.10,
        "Spain": 0.15,      # large Spanish-speaking, often neutral-Latin
        "Saudi Arabia": 0.05,
    },
    "New York/New Jersey": {
        # Huge, broad immigrant metro. Colombian 500k+, Senegalese, Brazilian,
        # large European communities.
        "Senegal": 0.30,
        "Brazil": 0.30,
        "Colombia": 0.30,
        "Morocco": 0.20,
        "France": 0.10,
        "Ecuador": 0.30,    # very large Ecuadorian community in NYC (Queens)
        "Norway": 0.05,
        "England": 0.10,
        "Panama": 0.15,
        "Germany": 0.10,
    },
    "Boston (Foxborough)": {
        # Largest US Brazilian + Haitian communities (state-leading); Irish/Scottish roots.
        "Brazil": 0.40,
        "Haiti": 0.35,
        "Scotland": 0.15,
        "Morocco": 0.10,
        "Ghana": 0.10,
        "England": 0.10,
        "Norway": 0.05,
    },
    "Philadelphia": {
        # Broad metro; notable Brazilian, Haitian, Ivorian/African, Italian/European
        "Brazil": 0.20,
        "Haiti": 0.20,
        "Ivory Coast": 0.10,
        "Ghana": 0.10,
        "Croatia": 0.10,
        "France": 0.05,
    },
    "Atlanta": {
        # Korean ~48k, Vietnamese ~52k metro; large African (incl. Ghanaian/Nigerian)
        # and Latino growth.
        "South Korea": 0.30,
        "Ghana": 0.20,
        "South Africa": 0.10,
        "Spain": 0.10,
        "Morocco": 0.10,
        "Haiti": 0.15,
        "DR Congo": 0.10,
    },
    "Seattle": {
        # Diverse Pacific NW; notable East African (Egyptian/Somali), Iranian, Asian
        "Egypt": 0.15,
        "Iran": 0.15,
        "Japan": 0.10,
    },
    "Kansas City": {
        # Heavy Mexican/Hispanic metro; otherwise limited specific diasporas
        "Argentina": 0.10,
        "Austria": 0.05,
    },
    "San Francisco Bay Area": {
        # Very diverse; large Asian + tech-driven international communities
        "Switzerland": 0.10,
        "Australia": 0.10,
        "Turkey": 0.10,
        "Algeria": 0.05,
    },

    # --- CANADA ---
    "Toronto": {
        # Among the most multicultural cities on earth. Strong Portuguese, Croatian,
        # Bosnian, Iranian, Ghanaian, Italian, Caribbean communities.
        "Croatia": 0.30,
        "Bosnia & Herzegovina": 0.25,
        "Ghana": 0.20,
        "Panama": 0.10,
        "Germany": 0.10,
        "Ivory Coast": 0.10,
        "Senegal": 0.15,
        "Iraq": 0.15,
    },
    "Vancouver": {
        # Highly diverse Pacific gateway; large Iranian, East/South Asian communities.
        "Iran": 0.25,
        "Egypt": 0.10,
        "Qatar": 0.05,
        "Switzerland": 0.10,
        "Australia": 0.15,
        "Belgium": 0.05,
        "Turkey": 0.10,
        "New Zealand": 0.10,
    },

    # --- MEXICO (Mexico itself handled as host; travelling teams get little) ---
    "Guadalajara": {
        "Colombia": 0.10,
        "Spain": 0.15,   # Spanish cultural/linguistic affinity
        "Uruguay": 0.10,
        "South Korea": 0.05,
    },
    "Monterrey": {
        "Spain": 0.10,
        "Japan": 0.05,
    },
    "Mexico City": {
        "Colombia": 0.15,
        "Spain": 0.10,
    },
}

# A tiny generic edge for a host nation playing in a SIBLING host country
# (e.g. USA fans travelling to Canada/Mexico, which is easy and common).
# Kept very small. Only applied when the team is a host nation but NOT playing
# in its own country.
SIBLING_HOST_EDGE = {
    ("USA", "Canada"): 0.20,
    ("USA", "Mexico"): 0.15,
    ("Canada", "USA"): 0.15,
    ("Mexico", "USA"): 0.30,   # huge Mexican diaspora across the US
    ("Mexico", "Canada"): 0.10,
    ("Canada", "Mexico"): 0.05,
}


def support_for(team: str, city: str, opponent: str) -> float:
    """Estimate home_support in [0,1] for `team` at `city`."""
    country_of_city = city_country(city)
    boost = 0.0

    # 1) Host-nation playing at home in its own country.
    if team in HOST_NATION_HOME_SUPPORT and team == country_of_city:
        return min(1.0, HOST_NATION_HOME_SUPPORT[team])

    # 2) Host nation playing in a sibling host country (light travel edge).
    if team in HOST_NATION_HOME_SUPPORT and country_of_city in HOST_NATION_HOME_SUPPORT:
        boost = max(boost, SIBLING_HOST_EDGE.get((team, country_of_city), 0.10))

    # 3) Specific diaspora boost from the venue city.
    city_map = CITY_DIASPORA.get(city, {})
    boost = max(boost, city_map.get(team, 0.0))

    return min(1.0, boost)


def build_rationale(home, away, city, hs, as_, net):
    cc = city_country(city)
    parts = []
    if home == cc and home in HOST_NATION_HOME_SUPPORT:
        parts.append(f"{home} plays at home in {cc} (true home crowd)")
    elif hs >= 0.30:
        parts.append(f"{home} has a strong local diaspora in {city}")
    elif hs >= 0.10:
        parts.append(f"{home} has a modest local following in {city}")

    if away == cc and away in HOST_NATION_HOME_SUPPORT:
        parts.append(f"{away} plays at home in {cc} (true home crowd)")
    elif as_ >= 0.30:
        parts.append(f"{away} has a strong local diaspora in {city}")
    elif as_ >= 0.10:
        parts.append(f"{away} has a modest local following in {city}")

    if not parts:
        parts.append(f"{city}: no strong diaspora edge for either side (near-neutral)")

    direction = (
        f"net favours {home} (+{net:.2f})" if net > 0.01
        else f"net favours {away} ({net:.2f} to {home})" if net < -0.01
        else "balanced (~0)"
    )
    return "; ".join(parts) + f" -> {direction}."


def main():
    df = pd.read_csv(WEATHER_CSV)

    rows = []
    for _, r in df.iterrows():
        match = r["match"]
        city = r["city"]
        m = re.match(r"^(.*?)\s+vs\s+(.*)$", match)
        if not m:
            raise ValueError(f"Could not parse match string: {match!r}")
        home, away = m.group(1).strip(), m.group(2).strip()

        hs = round(support_for(home, city, away), 3)
        as_ = round(support_for(away, city, home), 3)

        net = SUPPORT_TO_GOALS * (hs - as_)
        net = max(-CAP, min(CAP, net))
        net = round(net, 3)

        rows.append({
            "match_id": r["match_id"],
            "match": match,
            "city": city,
            "home_team": home,
            "away_team": away,
            "home_support": hs,
            "away_support": as_,
            "net_home_adv_goals": net,
            "rationale": build_rationale(home, away, city, hs, as_, net),
        })

    out = pd.DataFrame(rows, columns=[
        "match_id", "match", "city", "home_team", "away_team",
        "home_support", "away_support", "net_home_adv_goals", "rationale",
    ])
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")

    top = out.reindex(
        out["net_home_adv_goals"].abs().sort_values(ascending=False).index
    ).head(8)
    print("\nTop 8 matches by |net_home_adv_goals|:")
    for _, t in top.iterrows():
        print(f"  {t['net_home_adv_goals']:+.2f}  {t['match']:<35} @ {t['city']}")


if __name__ == "__main__":
    main()
