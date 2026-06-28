# Probability Cup Group Stage Recap — Internal Analysis

External recap (SportsPredict): +625 RBP across 559 settled forecasts.
This note reconciles that field performance with the model's own Brier scoring on the
48 group-stage matches that have finished and been cached in `wc_scores.json`.

## Internal Brier summary

- Settled: 48 matches, 341 goal-market questions (score-settleable only).
- Model Brier: **0.1712** vs 0.50-baseline **0.2500** → **+31.5% skill**.
- Calibration: well judged in the 40-80% bucket, overconfident above 80%.

| Predicted bucket | n   | Actual hit rate | Implied calibration |
|------------------|-----|-----------------|---------------------|
| 0-10%            | 80  | 4%              | underconfident      |
| 10-20%           | 56  | 18%             | roughly fair        |
| 20-30%           | 29  | 31%             | fair                |
| 30-40%           | 9   | 22%             | underconfident      |
| 40-50%           | 44  | 45%             | good                |
| 50-60%           | 44  | 50%             | good                |
| 60-70%           | 16  | 62%             | good                |
| 70-80%           | 25  | 68%             | slightly overconf.  |
| 80-90%           | 20  | 75%             | overconfident       |
| 90-100%          | 18  | 78%             | overconfident       |

The high-end overconfidence is why `sp_solver.py` already caps submissions at 88% and
applies a global 0.88 shrink toward 50%. That cap should remain in place.

## By market

| Market            | n   | Brier  | Skill vs 0.50 |
|-------------------|-----|--------|---------------|
| Correct Score     | 86  | 0.0688 | +72%          |
| Handicap          | 50  | 0.1433 | +43%          |
| First To Score    | 11  | 0.1752 | +30%          |
| Double Chance     | 35  | 0.1829 | +27%          |
| Moneyline         | 30  | 0.1893 | +24%          |
| Team Total        | 45  | 0.2054 | +18%          |
| BTTS              | 24  | 0.2400 | +4%           |
| Total Goals O/E   | 24  | 0.2503 | ~0%           |
| Total Goals       | 36  | 0.2856 | -14%          |

Key takeaway: **Total Goals was the only internal market worse than a coin flip.**
Yet the external recap shows **+3.6 RBP on "Total goals (under)"**. That means the
model was directionally contrarian on unders and the crowd was *even worse* at pricing
totals — the model's under calls were miscalibrated but still beat the field.

## By source

| Source   | n   | Brier  | Skill |
|----------|-----|--------|-------|
| poisson  | 274 | 0.1667 | +33%  |
| sharp    | 67  | 0.1897 | +24%  |

Surprising: the direct de-vigged Pinnacle "sharp" rows underperformed the Poisson-derived
rows in this sample. Likely drivers:

1. Sharp source is concentrated on Moneyline / Total Goals / BTTS, and Total Goals was
catastrophically miscalibrated this tournament.
2. Poisson-derived rows are dominated by Correct Score, Handicap and Team Total, which
had small samples and low Brier.
3. Small sample on sharp (67 questions) vs Poisson (274).

Do not over-interpret this as "Poisson > Pinnacle" in general. It is a tournament-specific
sample effect. The right lesson is that the sharp total-goal line should be sanity-checked
against the Poisson / Elo cross-check, especially in low-scoring editions.

## Recap categories vs internal data

| Recap category (RBP gap) | Internal market(s) | Internal Brier | Notes |
|--------------------------|--------------------|----------------|-------|
| Team shots on target (+6.5) | Shots / Shots on Target | not score-settleable | Model source is `teamstat:sot`. Strong → keep. |
| Offsides (+3.7) | Offsides | not score-settleable | `teamstat:offsides` is unpriced; model beat crowd. |
| Total goals (under) (+3.6) | Total Goals | **0.2856** (poor) | Directionally right but miscalibrated; crowd was worse. |
| BTTS + totals (+3.0) | BTTS + Total Goals | 0.2674 | Mixed. |
| Team to score in 2nd half (+2.9) | 2nd Half | not score-settleable | Poisson 2H model. |
| Team corner comparisons (+2.2) | Corners | not score-settleable | `teamstat:corners`. |
| Team to score anytime (+2.2) | Team Total | 0.2054 | Good internal. |
| Total goals (over) (+2.1) | Total Goals | 0.2856 | Same as under bucket; model overs too. |
| Match winners (+1.7) | Moneyline / Double Chance / DNB | 0.1858 | Solid, driven by Poisson moneyline. |
| Team SoT comparisons (+1.5) | Shots on Target | not score-settleable | Keep. |
| Team foul comparisons (+1.5) | Fouls | not score-settleable | Keep. |
| Player shots on target (+0.8) | Shots on Target | not score-settleable | Player prop margins are noisy. |
| Total shots on target (-0.1) | Shots on Target | not score-settleable | Near-neutral. |
| Team card comparisons (-1.6) | Cards | not score-settleable | Already applying `WC_CARD_DEFLATION = 0.65`. |
| Team corners (-2.8) | Corners | not score-settleable | Possible over/under-confidence. |
| Both teams shots on target (-5.5) | Shots on Target | not score-settleable | Joint independence assumption likely wrong. |
| Total cards (-7.5) | Cards | not score-settleable | Weakest recap category; deflation may need more. |

## Actionable adjustments for knockouts

1. **Total Goals / unders**: apply a knockout-specific goal reduction (~10%) to the
   Poisson lambdas, because knockout scoring drops 14% and Under 2.5 hits ~59%.
2. **Draw probability**: knockouts draw ~27% at 90 min vs ~22% in groups; lift draw
   probabilities modestly for moneyline/DC/DNB questions.
3. **Cards**: group-stage card deflation already at 0.65; keep it for early knockouts
   (games are still cleaner than league football) but monitor late-KO fixtures where
   stakes can raise card counts.
4. **Both teams SoT**: the product of two independent Poisson tail probabilities
   overstates the joint event; add a negative correlation fudge (~5-8%) in knockouts.
5. **High-confidence clipping**: maintain the 88% cap / 0.88 shrink — the calibration
   buckets confirm it is necessary.
6. **Futures / bracket sim**: keep the existing `KO_SHRINK = 0.75` and
   `KO_ELO_REGRESS_ALPHA = 0.20` in `simulate_tournament.py`.

## Files

- `worldcup/track_brier.py` — fixed to handle the `predictions_log.csv` snapshot that
  lacks a `tier` column.
- `worldcup/data/brier_log.csv` — generated by `track_brier.py`.
- `worldcup/data/group_stage_recap_analysis.md` — this file.
