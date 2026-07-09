# World Cup Probability Cup Model

A prediction system for the **SportsPredict.com "Probability Cup"** (powered by Jump
Trading) — the 2026 FIFA World Cup forecasting contest scored by **relative Brier
score**. It produces a calibrated probability for every question the platform asks,
for every match, and can submit them directly to the live SportsPredict API.

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

Calibration guardrail: probabilities are **clipped to [3%, 97%]** and shrinked toward
50% for live submission. A confident-wrong call costs ~1.0 in Brier (catastrophic),
while clipping costs almost nothing when right.

## Pipeline

```
ODDS         fetch_wc_odds.py    The Odds API (soccer_fifa_world_cup, eu = Pinnacle)
               ├─ featured (h2h,totals,spreads) → data/wc_featured_raw.json [3 cr, all events]
               └─ per-event (btts,dc,dnb,1H,corners,cards,players) → data/events/<id>.json [~12 cr]
TEAM STATS   fetch_statsbomb.py   StatsBomb WC18/22 events → data/team_stats.csv (40 teams)
               build_extra_stats.py  results history → data/team_stats_extra.csv (18 gap teams)
PLAYERS      analyze_players.py   StatsBomb events → data/player_stats.csv, player_rankings.csv
WEATHER      fetch_wc_weather.py  Open-Meteo forecast per venue → data/wc_weather.csv
FUNDAMENTALS build_elo_v2.py      goal-Elo on 49k results → data/intl_ratings.csv
               lbenz_model.py       Luke Benz bivariate-Poisson ratings (all 219 teams)
               apply_adjustments.py + diaspora home edge + in-tournament momentum
                                    → data/wc_fundamentals.csv
                                    │
                                    ▼
             predict_wc.py   de-vig + double-Poisson (devig.py) + niche (niche.py)
                             + team-stat model (team_model.py) + weather scaling + value tiers
                             → data/wc_questions.csv   (one row per match × question)
                             → data/wc_match_summary.csv
                                    │
             tilt_matches.py  optional: rebaseline + apply directional tilts to lambdas
             sp_solver.py     map live SP questions → model probabilities
                             → data/sp_entries.csv
             sp_submit.py     POST/PATCH predictions to SportsPredict API
                             → live submissions (open markets only)
                                    │
             app_wc.py       searchable "Ask"-bar dashboard → http://localhost:5001
```

## Live SportsPredict API pipeline

The model is wired directly into the live contest:

| Script | Purpose |
|--------|---------|
| `fetch_sp.py` | Pulls today's live question list from SportsPredict → `data/sp_questions.csv`. Also reports how many predictions we already have on record. |
| `sp_solver.py` | Parses each free-text question and maps it to a model probability → `data/sp_entries.csv`. Tags each row HIGH / MED / LOW confidence. |
| `sp_submit.py` | Submits HIGH+MED confidence rows to the API. PATCHes existing predictions if unchanged, skips locked markets, backs off on 429s. |
| `sp_pipeline.py` | One command runs the full cycle: fetch → solve → submit. Pass `--refresh-model` to also re-run `predict_wc.py`, `apply_adjustments.py`, and `tilt_matches.py --rebaseline`. |

Run the full live cycle:

```bash
export SP_API_KEY="sp_live_..."
python3 worldcup/sp_pipeline.py --refresh-model
```

## Probability hierarchy (per question — best source wins via de-dup)

1. **`sharp:pinnacle`** — direct de-vig of a Pinnacle market for this exact question (best).
2. **`consensus:<book>`** — median de-vig across books when Pinnacle is missing.
3. **`book:<name>`** — single-book, margin-stripped (Yes-only player props).
4. **`poisson`** — double-Poisson (Dixon-Coles) fitted to de-vigged h2h+totals; covers
   correct score, odd/even, half markets, arbitrary handicaps.
5. **`teamstat`** — per-team rate model (team_model.py) for the UNPRICED markets
   (offsides, fouls, shots on target) and team-vs-team comparisons (shots/corners/cards).
   Opponent-adjusted (dampened), weather-scaled for shot/corner events, SB foul scale corrected.
   Reliability raised to **0.55** after strong team-stat category performance.
6. **`elo:model`** — fundamentals cross-check only (separate file `wc_elo_questions.csv`).
   NEVER entered: it disagrees with the market by up to 38pp (it can't see squad talent).

## Group stage results & model updates

Internal Brier score on 48 settled group-stage matches, 341 goal-market questions:

- **Model Brier 0.1712** vs 0.50-baseline **0.2500** → **+31.5% skill**
- External contest performance: **+625 RBP** across 559 forecasts

What worked (+RBP):

| Category | RBP gap | Notes |
|----------|---------|-------|
| Team shots on target | +6.5 | Strongest edge; team-stat SoT model beat crowd. |
| Offsides | +3.7 | Unpriced market; model signal was good. |
| Total goals (under) | +3.6 | Directionally right — crowd worse at totals. |
| Team to score anytime | +2.2 | Solid internal Brier (+18%). |
| Match winners | +1.7 | Moneyline/DC/DNB good. |

What leaked:

| Category | RBP gap | Fix applied |
|----------|---------|-------------|
| Total cards | -7.5 | Card deflation **0.65 → 0.70**. |
| Both teams SoT | -5.5 | Correlation fudge **0.92** on joint probability. |
| Team corners | -2.8 | Monitored; no strong directional fix. |
| Team card comparisons | -1.6 | Covered by higher card deflation. |

Calibration: the model was overconfident above 80%. Live submissions now apply a
**global 0.88 shrink toward 50%** and cap at **88%**.

Full recap: `data/group_stage_recap_analysis.md`

## Knockout stage adjustments

Single-elimination World Cup matches behave differently from group games:

- **-14% goals** (2.69 group → 2.31 knockout)
- **+23% 90-minute draws** relative to group stage
- **Under 2.5 hits ~59%** in knockouts vs 49-53% implied
- **Extra time** occurs in ~30-49% of knockout matches
- **Favorite -1.5 handicap** covers only ~31% vs implied ~45%
- **Underdog +0.5** covers ~60% vs implied 50%

Code changes in `sp_solver.py`:

- **Knockout goal adjustment**: matches from `2026-06-28` onward scale Poisson lambdas by **0.88** (12% fewer goals). This lowers totals/BTTS and raises unders/draws.
- **Card deflation**: raised to **0.70** after the group-stage card leak.
- **Both-teams SoT correlation fudge**: **0.92** to fix independence overstatement.
- **Parser coverage**: added "regulation", "end in a tie", "score first", "ahead at halftime", "win in regulation", BTTS phrasing, extra-time/to-qualify, and penalty-kick order handling.

Full strategy note: `data/knockouts_strategy.md`

## Value tiers (which to prioritise / skip)

Each question is tagged **HIGH / MEDIUM / LOW**. Expected relative-Brier points ≈
`(your_p − field_p)²`, biggest where you're confident AND the field is timid. Score =
`reliability × |p−0.5|×2 × market_obscurity`. **LOW / SKIP** = odd/even & half-compare
(no skill), longshot correct scores, and low-value model coin-flips — under cumulative
scoring these add variance without expected gain. Filter by tier and "Sort: value" in
the dashboard.

## Data coverage & honest limits
- **Team stats:** 40 teams from StatsBomb WC18/22 (exact events); 18 gap teams (Czechia,
  Curaçao, etc.) have real **goals** from results history but **priors** for the other stats.
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
                      Now handles predictions_log snapshots without a `tier` column.
lbenz_model.py        FUNDAMENTALS base = Luke Benz's published Bayesian bivariate-Poisson
                      ratings (alpha/delta per team, github.com/lbenz730/world_cup_2026).
apply_adjustments.py  ADJUSTED FUNDAMENTALS = lbenz bvp + diaspora home-edge + in-tournament
                      momentum → data/wc_fundamentals.csv (cross-check; NEVER over a sharp price).
fetch_apifootball.py  optional: set API_FOOTBALL_KEY to replace the 18 gap teams' prior stats
                      with real shots/corners/fouls/offsides/cards.
tilt_matches.py       optional: apply manual tilts (confidence-weighted) to selected match lambdas
                      while keeping a clean `.bak` baseline. `--rebaseline` resets from fresh summary.
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
# Full local pipeline (odds → model → dashboard)
python3 worldcup/fetch_wc_odds.py --events --hours 30
python3 worldcup/predict_wc.py
python3 worldcup/app_wc.py      # → http://localhost:5001

# Live contest submission (one command)
export SP_API_KEY="sp_live_..."
python3 worldcup/sp_pipeline.py --refresh-model

# Dry-run a few submissions to see payloads
python3 worldcup/sp_submit.py --dry-run --max 5

# Submit only one match
python3 worldcup/sp_submit.py --match "Spain vs Austria"
```

In the dashboard: search any question/team/market, filter by market, click a **%** to
copy it, and read the kickoff countdown. The source badge shows whether a number is sharp
or modelled.

## Markets the platform asks that have NO market anywhere
- **Offsides** — no book prices it; the platform's catalog doesn't include it either.
- **Fouls (team total)** — only thin player-foul markets exist.
These are intentionally not emitted; everything else is covered.

## API budget
- **The Odds API** free tier = 500 credits/month. Featured pull = 3 credits (all events). Each per-event
  niche pull ≈ 12 credits. Pull niche markets only for matches you're about to enter, via
  `--hours`. Remaining quota is printed on every fetch.
- **SportsPredict API** has no documented rate limit; `sp_submit.py` sleeps 1.3–1.6s between calls
  and backs off on 429s.
