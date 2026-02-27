---
name: WDSC Phase1a Ensembles
overview: Build league table, derive team power rankings, and predict home-team win probabilities for 16 matchups using multiple ensemble-based modeling pipelines; standardize evaluation with proper scoring rules and temporal cross-validation.
todos:
  - id: inventory_schema
    content: Parse `Docs/whl_2025.csv`, validate columns, and build game-level + team-season aggregate tables plus league standings.
    status: pending
  - id: define_metrics_cv
    content: Implement time-based cross-validation splits and metric computation (log loss, Brier, AUC, calibration).
    status: pending
  - id: pipelineA_elo_blend
    content: Implement xG/MOV-adjusted Elo rankings + calibrated logistic blend for matchup probabilities.
    status: pending
  - id: pipelineB_stacking
    content: Implement ML base models (logit + GBM + Elo feature) and stacking + calibration; derive power rankings from expected wins.
    status: pending
  - id: pipelineC_bayes
    content: Implement Bayesian attack/defense strength model and ensemble with Elo; produce rankings + matchup probs.
    status: pending
  - id: pipelineD_baseline
    content: Implement transparent four-factors score and ensemble with Elo as a fast benchmark.
    status: pending
  - id: final_ensemble
    content: Combine pipeline outputs into a model-of-models weighted ensemble optimized for validation log loss; emit final CSVs for SurveyMonkey.
    status: pending
isProject: false
---

# Phase 1a: Hockey power rankings + win-probability ensembles

## What we know from your repo

- **Primary data**: `[Docs/whl_2025.csv](Docs/whl_2025.csv)` (row-level segments with `toi`, line/goalie IDs, shots, xG, goals, penalties).
- **Modeling advice/winner patterns** (ensemble, Elo, calibration):
  - `[Docs/2025 Winning Teams/advice.txt](Docs/2025 Winning Teams/advice.txt)`
  - `[Docs/2025 Winning Teams/winner.txt](Docs/2025 Winning Teams/winner.txt)`
  - `[Docs/2025 Winning Teams/finalist.txt](Docs/2025 Winning Teams/finalist.txt)`
- **Tournament matchups**: `WHSDSC_Rnd1_matchups.xlsx` (16 first-round games and their home/away teams; assumed to live alongside your main data, e.g., `[Docs/WHSDSC_Rnd1_matchups.xlsx](Docs/WHSDSC_Rnd1_matchups.xlsx)` if stored in `Docs/`).

## Deliverables for Phase 1a

- **League table** (standings) for the season (not submitted, but required foundation).
- **Part 1 submission**: power ranking of all 32 teams.
- **Part 2 submission**: win probability for the **home team** in **16 matchups**.

## Key assumptions (explicit, easy to change later)

- **Game outcome** is computed by summing `home_goals` / `away_goals` across `record_id` within each `game_id`.
- **OT handling**: `went_ot==1` implies overtime; standings points will default to standard hockey points (**2 win, 1 OT loss, 0 regulation loss**). If your module specifies a different points system, we’ll swap it.
- **Chronology for validation**: if no dates exist, we’ll order games by numeric part of `game_id` (e.g., `game_100` → 100) for time-based splits.
- **Matchups input**: we will read the official first-round pairings directly from `WHSDSC_Rnd1_matchups.xlsx` (16 rows, with at least `home_team` and `away_team`, and optionally `game_id`/region columns).

## Common data engineering (shared by every approach)

- **Program 0 (shared library module)**: `src/wdsc_phase1a/` utilities for:
  - Reading `whl_2025.csv` and validating schema.
  - Aggregating to:
    - **Game-level table**: one row per game (`game_id`, teams, OT flag, final score, xG/shot/penalty summaries).
    - **Team-season table**: per-team rates and context splits (per 60 / per minute of TOI).
  - Feature blocks that reflect hockey nuance:
    - **xG & finishing**: xGF/60, xGA/60, xG share, goals-xG (finishing/goalie effect proxy).
    - **Shot generation/suppression**: shots for/against per 60, shot share.
    - **Special teams**: power-play vs penalty-kill segments using `PP_up`/`PP_kill_dwn` labels; create PP xG/60, PK xGA/60.
    - **Discipline**: penalties committed, PIM rates.
    - **Goalie effects**: goalie IDs aggregated to team-by-goalie minutes; optional shrinkage to avoid overfitting.
    - **Line matchup context**: proportions of TOI by `first_off/second_off` × opponent defensive pairings; summarized as team-level exposure features (avoid high-cardinality one-hot blowups).
  - Outputs:
    - `outputs/league_table.csv`
    - `outputs/games.parquet` (or `.csv`)
    - `outputs/team_season_stats.parquet`

## Multiple two-program pipelines (choose 2–4 to run in parallel with separate agents)

Each pipeline is deliberately **two programs**: one produces rankings, one produces matchup probabilities.

### Pipeline A: xG-weighted Elo + calibrated logistic blend (interpretable, organizer-aligned)

- **Program A1 — `rankings_elo.py`**
  - Build an Elo system updated per game.
  - Use **xG-differential or goal-differential MOV multiplier** (advice: Elo+MOV performs well).
  - Add **home-ice** term; treat OT as reduced-information outcome (e.g., smaller K or MOV cap).
  - Strength-of-schedule: track opponent Elo exposure.
  - Output: `outputs/power_rankings_elo.csv` (team, elo, rank).
- **Program A2 — `predict_matchups_elo_blend.py`**
  - Convert Elo difference to base win probability.
  - Blend with a lightweight model (logistic regression) using team-season context (xG share, special teams, discipline) as controls.
  - Calibrate final probabilities (Platt or isotonic) on validation folds.
  - Output: `outputs/matchup_predictions_pipelineA.csv`.

### Pipeline B: Game-level ML (GBM) + Elo feature + stacking (high accuracy, careful leakage control)

- **Program B1 — `rankings_ml_expected_wins.py`**
  - Train a model to predict home win (probability) from **pre-game** features (team rolling form, Elo, season-to-date rates).
  - Derive a **power ranking** from “expected win rate vs average team” computed by simulating each team against a fixed reference opponent.
  - Output: `outputs/power_rankings_pipelineB.csv`.
- **Program B2 — `predict_matchups_stacked.py`**
  - Base learners (diverse inductive biases):
    - Regularized logistic regression (strong baseline)
    - Gradient boosting (XGBoost/LightGBM)
    - CatBoost (if you keep some categorical IDs)
    - Elo-only model
  - Meta-learner: logistic regression on out-of-fold predictions (stacking).
  - Calibration step on held-out time folds.
  - Output: `outputs/matchup_predictions_pipelineB.csv`.

### Pipeline C: Bayesian team-strength model (attack/defense) + Elo ensemble (robust with small samples)

- **Program C1 — `rankings_bayes_attack_defense.py`**
  - Fit a hierarchical model for team offense/defense latent strengths using **xG** (preferred) and/or goals.
  - Partial pooling stabilizes teams with fewer minutes/odd schedules.
  - Output: `outputs/power_rankings_pipelineC.csv` (posterior mean + uncertainty intervals).
- **Program C2 — `predict_matchups_bayes_ensemble.py`**
  - Produce win probabilities by combining:
    - Bayesian implied win probability (from latent strengths)
    - Elo implied win probability
    - Optional simple logistic correction with special teams/disciplines
  - Combine via weighted average where weights are chosen by minimizing validation log loss.
  - Output: `outputs/matchup_predictions_pipelineC.csv`.

### Pipeline D: “Four factors” hockey model + simple ensemble (fast, transparent benchmark)

- **Program D1 — `rankings_four_factors.py`**
  - Create a composite score from standardized components:
    - xG share
    - special teams differential
    - penalty discipline
    - goalie/finishing residual (goals−xG)
  - Output: `outputs/power_rankings_pipelineD.csv`.
- **Program D2 — `predict_matchups_four_factors_ensemble.py`**
  - Turn ranking-score difference into probabilities via a logistic link.
  - Ensemble with Elo-only probability for stability.
  - Output: `outputs/matchup_predictions_pipelineD.csv`.

## Evaluation and model-comparison metrics (what to optimize)

Because Part 2 requires **probabilities**, use **proper scoring rules** as primary metrics.

- **Primary (probabilistic accuracy)**:
  - **Log loss (cross-entropy)**: best single metric for comparing probability models; heavily penalizes overconfident wrong picks.
  - **Brier score**: complements log loss; more interpretable and closely tied to calibration.
- **Secondary (ranking/discrimination)**:
  - **ROC AUC**: tells whether the model ranks winners above losers (not a calibration metric).
- **Calibration diagnostics (don’t optimize alone, but must check)**:
  - Reliability curve + summary like **ECE** (expected calibration error).
- **For power rankings (Part 1)**:
  - Evaluate rankings by their ability to predict **held-out game outcomes**:
    - Treat rating difference as a predictor and compute **log loss/Brier** on held-out games.
    - Also report **Spearman correlation** between ranking score and future-period win%/xG share as a sanity check.

## Validation design (prevents “too good to be true” results)

- Use **temporal cross-validation** (rolling/forward-chaining): train on early games, validate on later games.
- Ensure features are **season-to-date / pre-game only** (no post-game leakage).

## Final ensembling across pipelines (the “model of models” step)

- After you have Pipelines A–D producing matchup probabilities, create a final combiner:
  - Weighted average of pipeline probabilities.
  - Choose weights to minimize validation log loss (constrained to sum to 1).
  - Output: `outputs/matchup_predictions_ensemble.csv`.

## What each agent can run in parallel later

- Agent 1: Pipeline A (Elo + calibrated blend)
- Agent 2: Pipeline B (GBM + stacking)
- Agent 3: Pipeline C (Bayesian + Elo ensemble)
- Agent 4: Pipeline D (four-factors transparent baseline)
All agents share the same `outputs/games.parquet` and time-split definitions so metrics are comparable.

