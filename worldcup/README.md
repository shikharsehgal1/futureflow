# World Cup Probability Cup Model

A prediction system for the **SportsPredict.com "Probability Cup"** (powered by Jump
Trading) — the 2026 FIFA World Cup forecasting contest scored by **relative Brier
score**. It produces a well-calibrated probability for *every* question the platform
asks, for *every* one of the 71 group-stage matches, and serves them in a searchable
dashboard for fast entry before each kickoff.

## Why this design (the scoring math)

The contest scores each binary YES/NO question with **Brier = (prediction − outcome)²**
and ranks you by **cumulative (field-average Brier − your Brier)** across all questions.
Two facts dictate the strategy:

1. **You only need to beat the average competitor, not be perfect.** The casual field
   is soft, especially on secondary markets. De-vigged *sharp* (Pinnacle) probabilities
   beat that field on almost every question — so we **anchor to Pinnacle** and report the
   de-vigged number, *not* a hand-tuned model that drifts from the market.
2. **Rank is cumulative, not average.** More questions answered = more points. So we
   answer **everything** — moneyline, draw, double chance, totals, odd/even, BTTS, team
   totals, handicaps, correct score, first-half/second-half, corners, cards, shots,
   anytime scorer, player cards — for all matches.

Calibration guardrail: probabilities are **clipped to [3%, 97%]**. A confident-wrong
call costs ~1.0 in Brier (catastrophic), while clipping costs almost nothing when right.
We do **not** shrink toward 50% — the sharp line is already best-calibrated, and shrinking
would surrender edge versus the field. Set your **final** value just before kickoff using
the freshest line (the dashboard's "Re-pull odds" button).

## Pipeline

```
ODDS         fetch_wc_odds.py   The Odds API (soccer_fifa_world_cup, eu = Pinnacle)
               ├─ featured (h2h,totals,spreads) → data/wc_featured_raw.json [3 cr, all events]
               └─ per-event (btts,dc,dnb,1H,corners,cards,players) → data/events/<id>.json [~12 cr]
TEAM STATS   fetch_statsbomb.py  StatsBomb WC18/22 events → data/team_stats.csv (40 teams)
               build_extra_stats.py  results history → data/team_stats_extra.csv (18 gap teams)
PLAYERS      analyze_players.py  StatsBomb events → data/player_stats.csv, player_rankings.csv
WEATHER      fetch_wc_weather.py Open-Meteo forecast per venue → data/wc_weather.csv
FUNDAMENTALS build_elo_v2.py     goal-Elo on 49k results → data/intl_ratings.csv, wc_elo_questions.csv
                                    │
                                    ▼
             predict_wc.py   de-vig + double-Poisson (devig.py) + niche (niche.py)
                             + team-stat model (team_model.py) + weather scaling + value tiers
                             → data/wc_questions.csv   (one row per match × question, ~6,200)
                             → data/wc_match_summary.csv
                                    ▼
             app_wc.py       searchable "Ask"-bar dashboard → http://localhost:5001
```

## Probability hierarchy (per question — best source wins via de-dup)

1. **`sharp:pinnacle`** — direct de-vig of a Pinnacle market for this exact question (best).
2. **`consensus:<book>`** — median de-vig across books when Pinnacle is missing.
3. **`book:<name>`** — single-book, margin-stripped (Yes-only player props).
4. **`poisson`** — double-Poisson (Dixon-Coles) fitted to de-vigged h2h+totals; covers
   correct score, odd/even, half markets, arbitrary handicaps.
5. **`teamstat`** — per-team rate model (team_model.py) for the UNPRICED markets
   (offsides, fouls) and team-vs-team comparisons (shots/corners/cards). Opponent-adjusted
   (dampened), weather-scaled for shot/corner events, SB foul scale corrected.
6. **`elo:model`** — fundamentals cross-check only (separate file `wc_elo_questions.csv`).
   NEVER entered: it disagrees with the market by up to 38pp (it can't see squad talent).

## Value tiers (which to prioritise / skip)

Each question is tagged **HIGH / MEDIUM / SKIP**. Expected relative-Brier points ≈
`(your_p − field_p)²`, biggest where you're confident AND the field is timid. Score =
`reliability × |p−0.5|×2 × market_obscurity`. **SKIP** = odd/even & half-compare (no skill),
longshot correct scores, and low-value model coin-flips — under cumulative scoring these add
variance without expected gain. Filter by tier and "Sort: value" in the dashboard.

## Data coverage & honest limits
- **Team stats:** 40 teams from StatsBomb WC18/22 (exact events); 18 gap teams (Czechia,
  Curaçao, etc.) have real **goals** from results history but **priors** for the other stats
  (FBref, the only free recent source, is Cloudflare-blocked).
- **Players:** WC18/22 squads only — partially stale; per-90 noisy on small minutes. The sharp
  anytime-scorer market is the better attacker signal.
- **Weather:** applied only to team-stat shot/corner events (market already prices weather into
  goal markets — re-applying would double-count); shown as a 🌡️ badge when goal-suppressing.
- **No market exists** for offsides or team fouls → those are pure model (use with modest confidence).

## Live ops & extra signals

```
refresh.py            one refresh cycle: pull featured odds (3cr) + recompute + settle results.
                      --no-niche keeps it ~5 credits; niche markets stay manual (credit-frugal).
track_brier.py        fetch finished-match scores → score the model's goal-market questions
                      (Brier + skill vs 0.5-baseline + calibration buckets + per-market/source).
lbenz_model.py        FUNDAMENTALS base = Luke Benz's published Bayesian bivariate-Poisson
                      ratings (alpha/delta per team, github.com/lbenz730/world_cup_2026).
                      Far better-calibrated than home-grown Elo; covers all 219 teams + knockouts.
apply_adjustments.py  ADJUSTED FUNDAMENTALS = lbenz bvp + diaspora home-edge + in-tournament
                      momentum → data/wc_fundamentals.csv (cross-check; NEVER over a sharp price).
fetch_apifootball.py  optional: set API_FOOTBALL_KEY to replace the 18 gap teams' prior stats
                      with real shots/corners/fouls/offsides/cards (no-key path prints setup).
```

**Soft factors (display + fundamentals layer only — the market already prices them, so they
never override a de-vigged sharp number):**
- 🏟️ **Home/diaspora crowd** (`home_advantage.csv`): host nations +0.30–0.35 goals, big diasporas
  +0.10–0.20 (Iran in LA, Colombia in Miami…). Shown as a per-match badge.
- 🌡️ **Weather** (`wc_weather.csv`): scales team-stat shot/corner events only; badge when ≤0.92.
- 🚑 **Injuries / late news** (`wc_injuries.csv`): web-sourced, dated; value is FRESH breaks the
  market may lag (e.g. Neymar out of Brazil's opener). Shown per match.
- 📈 **In-tournament momentum** (`momentum_weights.json`): +0.107 goals per goal of in-cup
  over-performance (cap ±0.3); nudges the fundamentals rating as group results land.

Dashboard endpoints: `/api/questions /api/summary /api/players /api/elo /api/injuries /api/ask`.

## Usage

```bash
# 1. Pull odds (featured = cheap; add niche markets for games closing within 30h)
python3 worldcup/fetch_wc_odds.py --events --hours 30

# 2. Compute every question
python3 worldcup/predict_wc.py

# 3. Open the dashboard (no Flask needed — pure stdlib)
python3 worldcup/app_wc.py      # → http://localhost:5001
```

In the dashboard: search any question/team/market, filter by market, click a **%** to
copy it, and read the kickoff countdown. The source badge shows whether a number is sharp
or modelled.

## Markets the platform asks that have NO market anywhere
- **Offsides** — no book prices it; the platform's catalog doesn't include it either.
- **Fouls (team total)** — only thin player-foul markets exist.
These are intentionally not emitted; everything else is covered.

## API budget
Free tier = 500 credits/month. Featured pull = 3 credits (all 71 events). Each per-event
niche pull ≈ 12 credits. Pull niche markets only for matches you're about to enter, via
`--hours`. Remaining quota is printed on every fetch.
