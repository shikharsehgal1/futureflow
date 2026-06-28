# Knockouts Strategy — 2026 World Cup Probability Cup

Group stage: +625 RBP on 559 forecasts. The model has clear edges and clear leaks.
This is the plan for the single-elimination rounds (Round of 32 onward), incorporating
research on WC knockout markets and the code changes made to act on it.

## 1. What worked in the group stage

| Category | RBP gap | Internal signal |
|----------|---------|-----------------|
| Team shots on target | +6.5 | Strongest edge. Team-stat SoT model beat crowd. |
| Offsides | +3.7 | Unpriced market; model signal was good. |
| Total goals (under) | +3.6 | Directionally right but miscalibrated — the crowd was just worse. |
| BTTS + totals | +3.0 | Mixed. BTTS internal Brier was only +4% vs baseline. |
| Team to score anytime | +2.2 | Internal Brier +18%, solid. |
| Match winners | +1.7 | Moneyline/DC/DNB Brier +24%, good. |
| Team SoT comparisons | +1.5 | Keep leaning on model here. |
| Team foul comparisons | +1.5 | Keep. |

## 2. What leaked

| Category | RBP gap | Diagnosis |
|----------|---------|-----------|
| Total cards | -7.5 | Worst category. Card model over-fired. Deflation raised. |
| Both teams SoT | -5.5 | Independence assumption overstates joint probability. Correlation fudge added. |
| Team corners | -2.8 | Possible over/under-confidence; no strong directional fix. |
| Team card comparisons | -1.6 | Same card overestimation issue. |

Internal Brier (48 matches, 341 score-settleable questions): **0.1712** vs 0.2500
baseline → **+31.5% skill**. Only **Total Goals** was worse than a coin flip
(Brier 0.2856), confirming the model was contrarian on unders but not well calibrated.

## 3. Knockouts are a different sport

Research on single-game WC elimination matches vs group stage:

- **Goals**: -14% (2.69 group → 2.31 knockout).
- **90-min draws**: +23% relative (22% → 27%).
- **Under 2.5 hit rate**: ~59% in knockouts vs 49-53% implied.
- **Extra time**: ~30-49% of knockout matches historically.
- **Favorite -1.5 handicap**: covers only ~31% vs implied ~45%.
- **Underdog +0.5**: covers ~60% vs implied 50%.
- **First penalty kick**: team shooting first wins ~58% of shootouts.
- **Cards**: rise in later rounds as stakes increase, but early knockouts remain
cleaner than domestic football.

The market systematically under-prices caution, draws, and extra time in
single-elimination games.

## 4. Code changes made

### `worldcup/sp_solver.py`
- **Knockout goal adjustment**: matches with `commence >= 2026-06-28` scale the
  Poisson lambdas by `KO_GOAL_ADJ = 0.88` (12% reduction). This reduces total-
  goals-over probabilities, increases unders and draws, and lowers BTTS.
- **Card deflation**: `WC_CARD_DEFLATION` raised from 0.65 to **0.70** after the
  -7.5 RBP card leak.
- **Both-teams SoT correlation fudge**: `BOTH_SOT_CORR = 0.92` applied to the joint
  probability to fix the independence overstatement.
- **Parser coverage**: added handlers for SP's common "regulation" phrasing:
  - plain "Will both teams score ...?"
  - "Will regulation ... end in a tie?"
  - "Will [team] score the first goal of the match?"
  - "Will [team] be ahead at halftime?"
  - "Will [team] win the match in regulation?"

### `worldcup/predict_wc.py`
- `_RELIABILITY["teamstat"]` raised from 0.50 to **0.55** because team SoT,
  offsides, and team totals were the best recap categories.

### `worldcup/track_brier.py`
- Fixed crash when the snapshot `predictions_log.csv` lacks a `tier` column.

## 5. Betting / prediction strategy for knockouts

### Core priors
1. **Anchor to the sharp closing line**, but apply a knockout caution overlay.
2. **Prefer unders and draws** in 90-minute markets. Under 2.5 is the most
   documented edge across knockout tournaments.
3. **Fade heavy favorite handicaps** (-1.5, -2). Quality compression in knockouts
   makes large margins rare.
4. **Back underdog +0.5 / +1.5** when the crowd is heavy on the favorite — the
   public overweights brand names in elimination games.
5. **Use "to qualify / advance"** for favorites instead of 90-minute moneyline to
   avoid the draw/ET drag. For underdogs, 90-minute markets can be better value.
6. **Extra time is underpriced** historically. If a prop offers ET yes above 30%
   implied, it has long-term value.

### Category-specific plan
- **Total goals**: lean under in every knockout round except the third-place playoff.
  The model now applies the 12% KO lambda reduction automatically.
- **Moneyline / DNB**: in pick'em matches, DNB 0 is better than 1X2. In tight
  tactical matchups, the draw probability is higher than the market implies.
- **BTTS**: expect lower in knockouts due to caution; the model will reflect this.
- **Cards**: keep the deflation for early knockouts. Monitor late-QF/SF/Final fixtures
  where rising stakes can push card counts up.
- **SoT**: team-stat SoT remains the model's strongest edge. Continue entering these
  confidently, but the both-teams joint probability is now deflated.
- **Player props**: prefer anytime-scorer market over model estimates. Player SoT
  props are still noisy; use modest confidence.
- **Correct score / longshots**: skip low-probability correct scores under the
  cumulative scoring rule — they add variance without expected RBP gain.

### Stage-specific notes
- **Round of 32 (new 2026 format)**: no historical pricing baseline. Bookmakers are
  calibrating blind. Target quality-gap underdogs and Under 2.5 aggressively.
- **Round of 16**: quality gaps still exist; DNB on favorites and underdog +0.5 are
  the cleanest plays.
- **Quarter-Finals / Semi-Finals**: most defensive rounds. Under 2.5 and draw are
  the historical edges. Cards can rise here.
- **Final**: tiny sample, high variance. Avoid systematic goal bets; focus on draw
  and card props if the question set includes them.
- **Third-Place Playoff**: the exception — both teams are disappointed but free to
  attack. Over 2.5 and BTTS Yes are historically better.

## 6. Operational checklist before each knockout matchday

1. **Pull fresh odds** for the upcoming knockout fixtures:
   ```bash
   python3 worldcup/fetch_wc_odds.py --events --hours 24
   ```
   (Note: API quota was exhausted at the time of this update; wait for the quota
   reset or use a fresh key.)
2. **Re-run the prediction pipeline**:
   ```bash
   python3 worldcup/predict_wc.py        # rebuilds wc_questions.csv + wc_match_summary.csv
   python3 worldcup/sp_solver.py         # solves the live SP questions
   ```
3. **Review `data/sp_entries.csv`**: filter `confidence in (high, med)` and sanity-
   check the source badges.
4. **Check the dashboard** (`python3 worldcup/app_wc.py`) for any last-minute line
   moves and weather badges.
5. **Avoid overwriting** with stale group-stage match data — the knockout matches will
   appear in `wc_match_summary.csv` only after the odds pull.

## 7. Risk management

- The 88% probability cap / 0.88 global shrink in `sp_solver.py` stays in place.
  The calibration buckets showed 90-100% predictions only hit 78% actual.
- For futures / trading bots: keep the existing `KO_SHRINK = 0.75` and
  `KO_ELO_REGRESS_ALPHA = 0.20` in `simulate_tournament.py`.
- Don't let one bad match (e.g., Panama vs England -151 RBP) dominate bankroll.
  Cumulative scoring rewards volume, so keep answering every question with positive
  edge rather than chasing a single big call.

## 8. What to watch for next

- **Card category**: the deflation bump is a hypothesis; re-score after the first KO
  round to see if the RBP leak closes.
- **Total goals**: if the KO under adjustment is too strong (market may already
  price some caution), dial `KO_GOAL_ADJ` back toward 1.0.
- **Both-teams SoT**: the 0.92 correlation fudge is a starting point; refine if the
  KO sample is large enough to estimate the true correlation.
- **Parser gaps**: remaining low-confidence SP questions are mostly hydration-break,
  substitute, and player-card props. Add handlers only if they appear frequently in
  the KO question set.

## Reference files

- `worldcup/data/group_stage_recap_analysis.md` — internal Brier breakdown + recap
  reconciliation.
- `worldcup/data/knockouts_strategy.md` — this file.
- `worldcup/sp_solver.py` — live SP solver with KO adjustments.
- `worldcup/predict_wc.py` — dashboard with updated teamstat reliability.
- `worldcup/track_brier.py` — fixed Brier scoring tool.
