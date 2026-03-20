# shikharsoccer — Football Prediction Model

A full-stack football prediction system for the Big 5 European leagues plus CL/Europa/Conference League. Built from scratch using weighted least squares power ratings, market-implied goal differentials, attack/defence decomposition, weather controls, Transfermarkt injury data, and cross-league ratings.

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [Quick Start](#quick-start)
3. [Weekly Workflow](#weekly-workflow)
4. [Project Structure](#project-structure)
5. [Data Sources](#data-sources)
6. [Model Architecture](#model-architecture)
7. [Feature Implementation Log](#feature-implementation-log)
8. [Findings & Insights](#findings--insights)
9. [Configuration & Hyperparameters](#configuration--hyperparameters)
10. [Requirements](#requirements)

---

## What This Does

Given a set of upcoming football fixtures with closing odds, this system:

1. Fits **goal-differential power ratings** for all teams across 10 leagues (Big 5 top + second divisions) using a combination of actual results and market-implied goal differentials
2. Decomposes ratings into **attack and defence components** via Langville-Meyer iterative OD decomposition on an xG matrix
3. Applies **preseason priors** based on previous season performance and Transfermarkt squad valuations
4. Adjusts predictions for **weather conditions** (temperature, wind, precipitation) based on peer-reviewed sports science research
5. Flags **late-breaking injuries** from Transfermarkt that the market may not have fully priced in
6. Generates **H/D/A probabilities**, expected goals, over/under 2.5, fair odds, EV%, and Asian handicap spreads
7. Computes **closing line value** on settled bets to track whether the model has genuine edge
8. Runs a **cross-league model** for CL/Europa/Conference League using 3,854 European games with closing odds

---

## Quick Start

### 1. Install everything

```bash
pip3.10 install \
  pandas \
  numpy \
  scipy \
  requests \
  flask \
  cloudscraper \
  beautifulsoup4 \
  lxml \
  selenium \
  webdriver-manager
```

Full list with versions and purpose:

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, CSV I/O |
| `numpy` | Matrix operations, WLS solver |
| `scipy` | Poisson distribution, factorial, probit |
| `requests` | HTTP requests for odds API, Club Elo, Open-Meteo |
| `flask` | Frontend web server (`app.py`) |
| `cloudscraper` | Cloudflare bypass for Transfermarkt scraping |
| `beautifulsoup4` | HTML parsing for TM injury/valuation pages |
| `lxml` | Faster HTML parser backend for BeautifulSoup |
| `selenium` | Browser automation (OddsPortal scraper only) |
| `webdriver-manager` | Auto-installs ChromeDriver for Selenium |

> **Note:** `lxml` may require a system library on some machines. If install fails: `brew install libxml2` (Mac) or `apt-get install libxml2-dev` (Linux).

---

### 2. Set your API key

Open `preliminarymodel/fetch_fixtures.py` and replace:
```python
API_KEY = "YOUR_API_KEY_HERE"
```
with your key from [the-odds-api.com](https://the-odds-api.com) (free tier: 500 requests/month).

---

### 3. One-time setup

```bash
# Fetch historical weather for all 47k games (~45 mins, runs once)
python3.10 fetch_weather.py
python3.10 fetch_weather.py --merge

# Fetch squad valuations (run each August at season start)
python3.10 tm_valuations.py
python3.10 mainmodel/part7_ratings.py
```

---

### 4. Launch the dashboard

```bash
python3.10 app.py
```

Then open **http://localhost:5000** in your browser. Everything — predictions, ratings, injuries, CL, historical performance, and the full pipeline — is accessible from the UI. You never need to touch the terminal again after this point.

The dashboard lets you:
- View all current predictions with alerts highlighted
- Run the full Friday pipeline with one click (live log output)
- Refresh injuries from Transfermarkt
- View league ratings tables and charts
- Track historical CLV and alert hit rate
- Rebuild cross-league ratings and predict CL fixtures

I strongly recommend you do this. Saves you the effort of running everything.

---

## Weekly Workflow

### Monday (after weekend results)
```bash
python3.10 update_data.py
python3.10 preliminarymodel/fetch_fixtures.py --results
```

### Friday (before weekend games)
```bash
rm data/fixtures.csv
python3.10 preliminarymodel/fetch_fixtures.py     # fetch odds + weather forecast
python3.10 tm_injuries.py                          # scrape TM injuries
python3.10 mainmodel/part7_ratings.py              # refit ratings
python3.10 mainmodel/part7_predict.py              # generate predictions
```

### European weeks (CL/Europa)
```bash
python3.10 crossleague/build_cross_ratings.py
python3.10 crossleague/predict_cl.py
```

### Season start — August only
```bash
python3.10 tm_valuations.py
python3.10 mainmodel/part7_ratings.py
```

### One-time setup (after first install)
```bash
python3.10 fetch_weather.py          # fetch all historical weather (~45 mins)
python3.10 fetch_weather.py --merge  # merge into big5_with_probs.csv
```

---

## Project Structure

```
shikharsoccer/
│
├── data/
│   ├── big5_with_probs.csv       # Master dataset: 47k games, 10 leagues, 10 seasons
│   │                              # Includes mu_mkt, weather_adj per game
│   ├── fixtures.csv               # Upcoming fixtures with odds + weather forecasts
│   ├── predictions.csv            # Current week predictions
│   ├── predictions_log.csv        # All-time predictions with results + CLV
│   ├── tm_injuries.csv            # Current injury adjustments per fixture
│   ├── tm_valuations.csv          # Squad market values + carry weights
│   ├── weather.csv                # Historical weather per (team, date)
│   ├── weather_cache.json         # Open-Meteo cache (don't delete)
│   ├── tm_injuries_cache.json     # TM scraper cache (24hr TTL)
│   └── ratings.json               # Latest ratings for frontend
│
├── mainmodel/
│   ├── part7_ratings.py           # Core ratings model — fits all 10 leagues
│   └── part7_predict.py           # Prediction engine — H/D/A, xG, EV, alerts
│
├── preliminarymodel/
│   └── fetch_fixtures.py          # The Odds API + Open-Meteo forecast fetcher
│
├── crossleague/
│   ├── build_cross_ratings.py     # Unified cross-league ratings (CL/Europa/Conf)
│   ├── predict_cl.py              # CL/Europa match predictions
│   ├── data/
│   │   ├── european_combined.csv  # 3,854 CL/Europa/Conference games 2015-26
│   │   └── cross_ratings.csv     # Cross-league team ratings
│   └── scrapers/
│       ├── fetch_club_elo.py      # Club Elo API scraper
│       └── fetch_cl_odds.py       # European competition odds
│
├── fetch_weather.py               # Open-Meteo historical weather scraper
├── tm_injuries.py                 # Transfermarkt injury scraper
├── tm_valuations.py               # Transfermarkt squad valuation scraper
├── update_data.py                 # Monday results updater
├── tune_halflifes.py              # Grid search hyperparameter tuner
└── app.py                         # Frontend Flask app (localhost:5000)
```

---

## Data Sources

| Source | Data | Cost | Coverage |
|--------|------|------|----------|
| football-data.co.uk | Results + closing odds (Big 5) | Free | 2015-26, 10 leagues |
| The Odds API | Live upcoming fixture odds | Free tier (500 req/mo) | All upcoming fixtures |
| Open-Meteo | Historical + forecast weather | Free, no key | Any location, 1940-present |
| Transfermarkt | Squad values, injuries, suspensions | Free (scraped) | All Big 5 teams |
| Internal DB exports | CL/Europa/Conference results + odds | Internal | 2015-26 |
| Club Elo (clubelo.com) | Cross-league Elo ratings | Free API | 1960-present |

**Master dataset** (`data/big5_with_probs.csv`): 47,000+ rows covering Premier League, Championship, Bundesliga, 2. Bundesliga, La Liga, Segunda División, Serie A, Serie B, Ligue 1, Ligue 2 from 2015-16 through 2025-26. Each row includes result, closing odds, market-implied goal differential (`mu_mkt`), and weather conditions.

---

## Model Architecture

### Step 1 — Market-Implied Goal Differential (`mu_mkt`)

Closing H/D/A odds are converted to an implied goal differential via probit regression:

```
mu_mkt = probit(p_home / (p_home + p_away)) * scale
```

This is calibrated using historical results to convert the three-way market into a single goal-space scalar. This is the core "market signal" that anchors the model — at `MARKET_WEIGHT=0.95`, the model trusts the market heavily and only deviates when it has systematic evidence to do so.

### Step 2 — Weighted Least Squares Ratings (`fit_ratings`)

For each league-season, a design matrix is built with one indicator per team and one home advantage term:

```
X_home - X_away + HFA = Goal Differential
```

Two rows are created per game:
- **Actual row**: weighted by `decay_w * (1 - MARKET_WEIGHT) * weather_adj`
- **Market row**: weighted by `decay_w_mkt * MARKET_WEIGHT`

Weights include:
- **Time decay**: `0.5^(days/HALF_LIFE)` — recent games matter more
- **Market decay**: steeper at `0.5^(days/MARKET_HALF_LIFE)` — recent odds more informative
- **Weather control**: extreme conditions (high wind, heavy rain, heat) downweight actual goal diff rows — the scoreline is less informative about true team quality when conditions were abnormal

**Decoupled priors** prevent shrinkage from contaminating HFA:
- Team prior: `[team=+1, HFA=0] = 0` — shrinks team toward league average
- HFA prior: `[HFA=+1] = 0.3` — anchors home advantage independently

### Step 3 — Form/Momentum Blend

A short-window (21-day) rating is computed separately and blended 15/85 with the base rating. The form signal has essentially zero RMSE impact (within 0.0001 across all half-lives tested) but is retained for its directional value in capturing hot/cold streaks.

### Step 4 — Langville-Meyer Attack/Defence Decomposition

Iterative OD (Offensive/Defensive) decomposition on a time-decayed xG matrix (shots on target × conversion rate). This separates each team's rating into:
- **Attack**: tendency to create and convert chances
- **Defence**: tendency to suppress opponent chances

The attack/defence split is used for asymmetric lambda estimation in the Poisson model, which drives xGH, xGA, and O25% predictions. It is damped by `SPLIT_DAMP=0.15` to prevent overfitting on small samples.

### Step 5 — Preseason Priors

Before each season, `build_preseason_priors()` is called to initialise team ratings:

| Team type | Prior | Source |
|-----------|-------|--------|
| Returning teams | `carry_weight × last_season_rating` | Previous season WLS |
| Promoted teams | `-0.453` (goals) | Historical average of promoted teams |
| Relegated teams | `+0.236` (goals) | Historical average of relegated teams |

**Carry weight** is either flat (0.50 tier 1, 0.10 tier 2) or adjusted by Transfermarkt squad value change year-on-year:
- Squad value up 30% → carry weight 0.58 (invested heavily, should carry more)
- Squad value down 20% → carry weight 0.42 (sold key players, regress more)

Prior weight fades naturally via `PRESEASON_WEIGHT * adaptive_prior` — strong at week 1, near-zero by week 10.

### Step 6 — Prediction Engine

For each fixture:

```python
mu = h_rat - a_rat + home_adv         # goal diff prediction
mu += fresh_injury_adj                 # only if injury < 48h old
lH = clip((total + mu) / 2)           # home expected goals
lA = clip((total - mu) / 2)           # away expected goals
lH *= weather_adj                     # scale down in bad conditions
lA *= weather_adj
```

Symmetric Poisson on `mu` → H/D/A probabilities
Asymmetric LM lambdas → xGH, xGA, O25%

Outputs per fixture:
- H%, D%, A% — win probabilities
- xGH, xGA — expected goals
- O25% — over 2.5 goals probability
- Fair H/D/A — 1/p for each outcome
- EV H/D/A — `(p × odds - 1) × 100`
- AH spread — nearest 0.25 goal handicap
- ΔH, ΔD, ΔA — difference vs market probabilities
- **ALERT** — when any delta exceeds 6pp

---

## Feature Implementation Log

This section maps every feature from the original requirements to its implementation.

### Original Requirements (from spec doc)

**✅ Part 1 — Indicators-based power ratings**
`mainmodel/part7_ratings.py` — `fit_ratings()` builds exactly this design matrix. Decay, priors, and home advantage all implemented. See Steps 1-2 above.

**✅ Part 2 — Box score database**
`data/big5_with_probs.csv` — 47k games across 10 leagues from football-data.co.uk. Includes results, shots, closing odds, and derived columns.

**✅ Part 3 — Closing lines data**
H/D/A closing odds from football-data.co.uk (historical) and The Odds API (current). Bet365 closing line used as primary reference for CLV tracking.

**✅ Part 4 — Merged dataset**
`data/big5_with_probs.csv` contains results + odds merged. `mu_mkt` column is the probit-transformed market signal used in ratings.

**✅ Part 5 — Probability to goal space conversion**
`mu_mkt` computed via probit regression. Calibrated to convert (p_home, p_draw, p_away) to a single goal-differential scalar. Handles three-way market correctly.

**✅ Part 6 — Goal differential model**
Full WLS model in `fit_ratings()`. Two hyperparameters (HALF_LIFE, MARKET_WEIGHT) plus MARKET_HALF_LIFE tuned via grid search over 90 league-seasons.

**✅ Part 7 — Operational pipeline**
Full weekly workflow operational. `fetch_fixtures.py` → `part7_ratings.py` → `part7_predict.py`. Predictions saved to CSV with CLV tracked on settlement.

---

### Additional Features

**✅ Attack and Defence ratings (Langville-Meyer)**
`fit_ratings()` Step 3. Iterative OD decomposition on xG matrix. SPLIT_DAMP=0.15 prevents overfitting. Used for asymmetric lambda estimation → better xG and O25% predictions than symmetric model. Note: xG is shown in predictions output but is not fed back into the ratings — it's a downstream output of the LM decomposition, not a direct input.

**✅ Tunable half-life**
`tune_halflifes.py` — parallelised grid search over 49 combinations of (MARKET_HALF_LIFE, FORM_HALF_LIFE). Pre-builds design matrices once per league-season. Results: MARKET_HALF_LIFE=30d is optimal (RMSE=1.5619). Form half-life essentially irrelevant — all values within 0.0001 RMSE.

**✅ Form/momentum**
21-day short-window rating blended 15/85 with base rating. Implemented but empirically near-zero impact on RMSE, as expected — the market already prices in form.

**✅ Preseason priors based on past season and transfers**
`build_preseason_priors()` in `part7_ratings.py`. Handles returning/promoted/relegated teams differently. ATK/DFC priors set separately for promoted (-0.18/-0.16) and relegated (+0.04/-0.14) teams. Enhanced with Transfermarkt valuation-adjusted carry weights when `data/tm_valuations.csv` is present.

**✅ Total goals model (O25%)**
Asymmetric Poisson model using LM lambdas. `p_over25` computed from full joint Poisson distribution (not approximation). Over/under odds and EV tracked when available from The Odds API.

**✅ Closing line value tracking**
`fetch_fixtures.py --results` settles predictions. For each settled game, computes CLV = model probability vs closing odds. Tracked in `data/predictions_log.csv` with log-likelihood scoring. CLV chart in frontend historical tab.

**✅ Cross-league ratings / Cup data / League difficulty**
`crossleague/` — unified ratings fitted on 3,854 CL/Europa/Conference games (2015-26) with closing odds. CL games weighted 1.5×, league games 0.3× (tier 1) or 0.05× (tier 2). Re-centres so mean of Big 5 = 0. `predict_cl.py` uses cross-league ratings for CL/Europa quarter-finals and beyond.

**✅ Betting spreads (Asian Handicap)**
AH spread = `round(mu × 4) / 4` — nearest 0.25 goal handicap. Displayed in predictions output alongside fair odds and EV%.

**✅ Weather**
`fetch_weather.py` — fetches historical daily weather for all 47k games via Open-Meteo. 294-team stadium coordinate lookup. Adjustments derived from peer-reviewed research:
- *Pavlinovic et al. 2024* (Sports): temperature ≥21°C reduces high-intensity running 15% (large ES)
- *Zhong et al. 2024* (Biology of Sport): wind, humidity affect technical performance in UCL
- *Bray et al. 2008*: wind ≥20km/h significantly affects passing accuracy

Applied as `weather_adj` multiplier:
- ≥21°C: ×0.92 | ≥28°C: ×0.85 | <5°C: ×0.97
- Wind ≥20km/h: ×0.96 | ≥30km/h: ×0.92 | ≥40km/h: ×0.88
- Rain ≥2mm: ×0.97 | ≥5mm: ×0.95 | ≥15mm: ×0.90

Historical controls: actual goal diff rows downweighted by `weather_adj` in WLS — extreme weather games are treated as less informative about true team quality.
Forecast: `fetch_fixtures.py` automatically fetches Open-Meteo forecast for each upcoming fixture at time of odds fetch.

**✅ Transfermarkt player and team valuations + injuries**
`tm_injuries.py` — weekly Friday scraper. Fetches current injuries/suspensions for all fixture teams via `/{slug}/sperrenundverletzungen/verein/{id}/plus/1`. Position-weighted impact (ST=1.0 down to GK=0.15). Value from `/plus/1` endpoint. Adjustment applied only when `since_date` < 48 hours — market is assumed to have priced in older injuries.

`tm_valuations.py` — August scraper. Squad market values → adjusted carry weights for preseason priors. Year-on-year value change modulates carry weight between [0.32, 0.68] for tier 1.

**✅ Frontend**
Flask app (`app.py`) at `localhost:5000`. Tabs: Predictions, Ratings, Injuries, CL/European, Historical Performance, Pipeline Control. All pipeline scripts runnable from UI with live SSE log streaming.

**⚠️ Live updating**
Partial. Friday pipeline runs on-demand from frontend. True live in-play updating (during matches) not implemented — would require a separate in-play model with time-remaining and current score as state variables.

---

## Findings & Insights

### Market efficiency
At `MARKET_WEIGHT=0.95`, the model essentially treats closing odds as truth and only deviates when systematic evidence accumulates. This is the correct approach — Pinnacle/Bet365 closing lines are highly efficient, and trying to beat them purely on model sophistication has very low expected value. The model's edge comes from:
1. **Systematic disagreements** the market makes (e.g. draw probability miscalibration)
2. **Late injury news** the market hasn't fully reacted to
3. **Weather conditions** in less-liquid leagues

### Weather effect
63.5% of historical games have weather data. The effect is real but subtle — most European football is played in mild conditions. The biggest adjustments appear in:
- Southern Spain/Canary Islands in summer (heat)
- Northern France/England in winter storms
- Coastal stadiums (wind — Monaco, Brest, Brighton)

Marseille vs Lille on 22/03/26 had the largest forecast adjustment (0.864 — 27km/h wind + 17.8mm rain). This moves O25% meaningfully but barely touches H/D/A.

### Form signal
Form half-life tuning showed the form blend is essentially irrelevant — RMSE difference across all tested half-lives was <0.0001. The market already prices in form. The form feature is retained for edge cases (e.g. a team that's fired their manager, where recent results may be temporarily uncorrelated with underlying quality).

### Cross-league ratings
Top 5 by cross-league rating (2025-26):
1. Arsenal +1.589 (34 CL games)
2. Bayern Munich +1.547 (36 CL games)
3. Man City +1.368 (30 CL games)
4. Barcelona +1.304 (34 CL games)
5. Paris SG +1.279 (41 CL games)

Real Madrid at +0.913 (9th) reflects a difficult current CL season. The cross-league model correctly penalises recent poor European performance relative to domestic dominance.

### Preseason priors
Promoted teams average -0.453 goals vs league average in their first top-flight season. Relegated teams average +0.236 (they're still good teams by second division standards). These priors are critical in weeks 1-5 before enough data accumulates — without them, newly promoted teams get wild early ratings.

### Injury adjustment
The market almost always prices in known injuries at 0.95 weight. The 48-hour recency gate ensures the injury scraper only adds signal for genuinely fresh news (training ground injuries announced day-of). In practice this fires rarely — roughly 1-2 games per week have a fresh injury adjustment that moves the model.

---

## Configuration & Hyperparameters

All in `mainmodel/part7_ratings.py`:

```python
HALF_LIFE         = 90    # days — goal diff decay (optimal from grid search)
MARKET_HALF_LIFE  = 30    # days — odds decay (tuned, optimal)
MARKET_WEIGHT     = 0.95  # weight on market-implied vs actual goal diff
FORM_WEIGHT       = 0.15  # blend weight for short-window form rating
FORM_HALF_LIFE    = 21    # days — form window (irrelevant empirically)
SPLIT_DAMP        = 0.15  # LM attack/defence split damping
PRESEASON_WEIGHT  = 5.0   # prior weight (fades with games_per_team)
CARRY_WEIGHT_TIER1 = 0.50 # regression to mean for returning top-div teams
CARRY_WEIGHT_TIER2 = 0.10 # regression to mean for returning second-div teams
```

Alert threshold (in `part7_predict.py`):
```python
ALERT_THRESHOLD = 0.06    # 6pp disagreement with market triggers alert
```

Injury gate (in `part7_predict.py`):
```python
# Only applies injury adjustment if since_date < 48 hours ago
```

---

## Requirements

```bash
pip3.10 install pandas numpy scipy requests flask cloudscraper beautifulsoup4 lxml selenium webdriver-manager
```

**Python version**: 3.10+

**API keys needed**:
- The Odds API: set `API_KEY` in `preliminarymodel/fetch_fixtures.py` — [the-odds-api.com](https://the-odds-api.com) (free tier: 500 req/month)
- Open-Meteo: no key required
- Transfermarkt: no key required (scraped via cloudscraper)

**External dependencies**:
- Chrome browser (for Selenium — only needed if using OddsPortal scraper)
- ChromeDriver auto-installed by `webdriver-manager`

---

## Leagues Covered

| Code | League | Tier |
|------|--------|------|
| E0 | Premier League | 1 |
| E1 | Championship | 2 |
| D1 | Bundesliga | 1 |
| D2 | 2. Bundesliga | 2 |
| SP1 | La Liga | 1 |
| SP2 | Segunda División | 2 |
| I1 | Serie A | 1 |
| I2 | Serie B | 2 |
| F1 | Ligue 1 | 1 |
| F2 | Ligue 2 | 2 |

Plus CL, Europa League, and Conference League via `crossleague/`.# futureflow
# futureflow
