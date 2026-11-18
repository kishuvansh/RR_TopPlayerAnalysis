# Combined Contextual Pressure Index (CCPI) Specification

## 1. Overview & Objective
The Combined Contextual Pressure Index (CCPI) is a secondary pressure framework that assigns an intuitive, domain-grounded pressure score between 0 and 100 to every delivery, split by role:
- **Batting Pressure Index (BPI)**: Reflects the situational difficulty, resource scarcity, and scoreboard urgency faced by the batter on strike.
- **Bowling Pressure Index (BoPI)**: Reflects the opponent's scoring threat, defence margin, and damage risk faced by the bowler delivering the ball.

Scores are mapped to initial qualitative bands:
- `0–19`: Low
- `20–39`: Moderate (Ordinary: `< 40`)
- `40–59`: Elevated
- `60–79`: High (High: `>= 60` and live)
- `80–100`: Extreme

---

## 2. Innings-Specific Component Formulations

### Innings 1
In the first innings, there is no chase target; pressure arises from par-score trajectory deficit/threat, wickets lost, and remaining innings stage:

#### Innings 1 Batting Pressure Index (BPI_1)
- **Scoring Deficit vs Expected Trajectory (Weight: 40%)**:
  Calculated by comparing current team total to the historical baseline par total for the competition group at the current legal ball mark.
- **Wickets Lost (Weight: 35%)**:
  Normalized proportion of wickets lost ($10 - \text{wickets\_left}$).
- **Innings Stage (Weight: 25%)**:
  Progress of the innings based on legal balls elapsed ($120 - \text{balls\_left}$).

$$\text{BPI}_1 = 0.40 \times \text{Deficit}_{0-100} + 0.35 \times \text{WicketsLost}_{0-100} + 0.25 \times \text{Stage}_{0-100}$$

#### Innings 1 Bowling Pressure Index (BoPI_1)
- **Scoring Threat (Ahead of Expected Trajectory) (Weight: 40%)**:
  Higher when the batting side is scoring well above expected trajectory.
- **Batting Resources Remaining (Weight: 35%)**:
  Higher when the batting side has many wickets in hand late in the innings.
- **Innings Stage (Weight: 25%)**:
  Death overs and powerplay transition stage weight.

$$\text{BoPI}_1 = 0.40 \times \text{Threat}_{0-100} + 0.35 \times \text{BattingResources}_{0-100} + 0.25 \times \text{Stage}_{0-100}$$

---

### Innings 2
In the second innings, pressure is governed by required run rate relative to achievable baseline, wickets remaining, and ball countdown:

#### Innings 2 Batting Pressure Index (BPI_2)
- **Required Run Rate Difficulty vs Expected Achievable Rate (Weight: 45%)**:
  $\text{RRR} - \text{AchievableRate}$, normalized.
- **Wicket Pressure (Weight: 30%)**:
  $\text{WicketsLost} / 10$, amplified in the final 5 overs.
- **Time Pressure (Weight: 25%)**:
  Proportion of chase elapsed ($120 - \text{balls\_left}$).

$$\text{BPI}_2 = 0.45 \times \text{RRRDifficulty}_{0-100} + 0.30 \times \text{WicketPressure}_{0-100} + 0.25 \times \text{TimePressure}_{0-100}$$

#### Innings 2 Bowling Pressure Index (BoPI_2)
- **Run-Defence Difficulty (Weight: 45%)**:
  Proximity of the batting team's current run rate to the required rate and low runs to defend.
- **Wicket Threat (Weight: 30%)**:
  Opponent wickets remaining in hand.
- **Time Pressure (Weight: 25%)**:
  Death overs countdown with a small margin.

$$\text{BoPI}_2 = 0.45 \times \text{DefenceDifficulty}_{0-100} + 0.30 \times \text{WicketThreat}_{0-100} + 0.25 \times \text{TimePressure}_{0-100}$$

---

## 3. Mandatory Amendments

1. **Liveness Gate**:
   A ball counts as true competitive pressure **only if the win probability is between 5% and 95%**.
   Deliveries where the game is already decided (e.g. 80 runs needed off 15 balls with 8 wickets down) have mathematically high formula scores but zero leverage. These "lost cause" deliveries are gated out of HIGH pressure.
2. **Bowling Index Disambiguation**:
   The bowling index is explicitly defined from the bowler's perspective (high score = high danger of conceding victory / defending small margins). We report the correlation between BPI and BoPI to verify they are not trivial mirror inversions.
3. **Weight Calibration & Testing**:
   Starting weights are evaluated alongside:
   - Equal component weights (33.3% / 33.3% / 33.3%)
   - Non-Negative Least Squares (NNLS) weights fitted against the empirical Leverage Index (LI).
   $R^2$ is reported before selecting the final weight set.
4. **CCPI Tiers**:
   - `HIGH`: Score $\ge 60$ AND within the Live win-probability window (5%–95%).
   - `ORDINARY`: Score $< 40$.
   - Percentile-based cutoffs are tested in Phase 4 sensitivity.
