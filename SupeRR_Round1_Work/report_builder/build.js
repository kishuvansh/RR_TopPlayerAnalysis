const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, Footer, Header, PageNumber,
  BorderStyle, PageBreak, TableOfContents,
} = require("docx");
const H = require("./helpers");
const { readCSV, P, H1, H2, H3, Bullet, Caption, Formula, Callout, Img, Tbl, Spacer, FONT, INK, MUTED, ACCENT, WORK } = H;

const OUT = process.argv[2];

// ---------- data ----------
const bat = readCSV("handoff/TOP10_BATTERS.csv");
const bowl = readCSV("handoff/TOP10_BOWLERS.csv");
const boot = readCSV("outputs/tables/final_bootstrap_rank_intervals.csv");
const batBox = readCSV("outputs/tables/final_top10_batters_boxscore.csv");
const bowlBox = readCSV("outputs/tables/final_top10_bowlers_boxscore.csv");
const tiers = readCSV("outputs/tables/final_tier_profile.csv");
const rel = readCSV("outputs/tables/final_split_half_reliability.csv");
const vars_ = readCSV("outputs/tables/final_variants_comparison.csv");
const wts = readCSV("outputs/tables/final_weight_sensitivity.csv");
const pool = readCSV("outputs/tables/final_eligible_pool.csv");
const popRef = readCSV("outputs/tables/final_population_reference.csv")[0];
const calib = readCSV("outputs/tables/phase1_calibration_bins.csv");
const tierCmp = readCSV("outputs/tables/competition_tier_comparability.csv");

const COUNTRY_FIX = { "4219": "India" };
const country = r => r.country || COUNTRY_FIX[r.player_id] || "";
const f = (x, d = 2) => Number(x).toFixed(d);
const k = x => (Number(x) * 1000).toFixed(2); // rating per ball x1000
const bootOf = (role, pid) => boot.find(b => b.role === role && b.player_id === String(pid));
const wOf = (role, rank) => wts.find(w => w.role === role && w.rank === String(rank) + ".0");
const relRow = s => rel.find(r => r.Check.startsWith(s));
const nBat = pool.find(p => p.role === "Batting").eligible_players;
const nBowl = pool.find(p => p.role === "Bowling").eligible_players;
const batOverlap = wts.find(w => w.role === "bat" && w.name.startsWith("MEAN")).median_rank;
const bowlOverlap = wts.find(w => w.role === "bowl" && w.name.startsWith("MEAN")).median_rank;
const tHigh = tiers[0], tMid = tiers[1], tOrd = tiers[2], tDead = tiers[3];
const maxCalErr = Math.max(...calib.map(c => Number(c.Abs_Error ?? c.Absolute_Error ?? c.abs_error ?? 0))) || 0.0385;

const children = [];
const add = (...xs) => children.push(...xs);

// ---------- title block ----------
add(
  new Paragraph({ children: [new TextRun({ text: "SupeRR Selector Season 2  ·  Round 1", size: 20, color: ACCENT, bold: true, font: FONT })], spacing: { after: 80 } }),
  new Paragraph({ children: [new TextRun({ text: "Who Delivers When It Matters", size: 52, bold: true, color: INK, font: FONT })], spacing: { after: 60 } }),
  new Paragraph({ children: [new TextRun({ text: "Methodology report: a win-probability approach to rating T20 batters and bowlers under pressure", size: 26, color: MUTED, font: FONT })], spacing: { after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 8 } } }),
  P("**Data:** IPL and men's T20I ball-by-ball data, January 2022 – November 2026 (780 clean matches, 186,246 deliveries).   **Code:** Python pipeline in `SupeRR_Round1_Work/src/` (scripts s05–s11, seed 42)."),
);

// ---------- executive summary ----------
add(H1("Summary"));
add(P("We define pressure as **how much a single delivery can change who wins the match**. For every ball we estimate the batting side's win probability, simulate the seven possible outcomes of the next ball (dot, 1, 2, 3, 4, 6, wicket), and measure the expected swing in win probability. This number, the **Leverage Index (LI)**, is scaled so the average ball scores 1.0. A ball counts as **high pressure** when its LI is in the top 20% of *live* match situations (win probability 5–95%, after the first two overs). It counts as **ordinary** when its LI is in the bottom 50% of the same live situations."));
add(P("We measure performance as **win probability added per ball**: the change in the team's chance of winning caused by that delivery. Each player's rating comes from a ridge regression that adjusts for the opponent faced, innings, competition tier, batting position, pace or spin, and venue. Fitting the same model separately on high-pressure and ordinary balls gives every player a **pressure rating**, an **ordinary baseline**, and the gap between them (**resilience**). The final score combines pressure rating (50%), match-by-match consistency (25%) and resilience (25%). Only players with at least 60 high-pressure balls in at least 15 matches are ranked."));

const sumCols = [{ h: "#", w: 4, align: "center" }, { h: "Batter", w: 22 }, { h: "Bowler", w: 22 }];
add(Tbl(sumCols, bat.map((b, i) => [String(i + 1), `**${b.name}**  (${country(b)})`, `**${bowl[i].name}**  (${country(bowl[i])})`]), { size: 19 }));
add(Caption("Table 1. Final submitted rankings (identical to the Unstop dropdown selections). The full evidence is in Section 5."));
add(Callout(`**How sure are we?** The bowling list is the more robust of the two. The top four bowlers make the top 10 in 59–84% of match-level bootstrap resamples, and on average ${f(bowlOverlap, 1)} of the 10 bowlers survive any reasonable change to the score weights. The batting list is less certain: pressure batting is a noisier signal, split-half reliability is close to zero, and no batter makes the top 10 in more than 61% of resamples. Section 6 sets out these checks plainly instead of overstating confidence.`));

// ---------- 1. data ----------
add(new Paragraph({ children: [new PageBreak()] }));
add(H1("1. Data and scope"));
add(P("We used every file in the data pack (ball-by-ball, matches, competitions, teams, players, playing XIs and the competition–team–player mapping). The scope is all IPL seasons and all men's T20 Internationals from 2022 onwards. We applied the following exclusions so that every remaining delivery comes from a full, regulation 20-over contest:"));
add(Tbl([{ h: "Exclusion", w: 60 }, { h: "Matches", w: 14, align: "right" }, { h: "Reason", w: 60 }], [
  ["A-team / development events (Emerging Teams Asia Cup 2024, Rising Stars 2025)", "30", "Not senior international standard"],
  ["Rain-affected / DLS / revised-target matches", "26", "Artificial targets distort the required rate"],
  ["No result / abandoned", "25", "No outcome to model"],
  ["Ball-by-ball records missing", "16", "Cannot reconstruct match state"],
  ["Tied and decided by Super Over", "6", "Regulation play ended level"],
  ["**Total excluded (of 883)**", "**103**", "**780 matches kept: 355 IPL, 425 T20I**"],
]));
add(Caption("Table 2. Match exclusions. Super-over deliveries are also dropped; only innings 1 and 2 are used."));
add(H3("Data reconstruction decisions that affect the ratings"));
add(Bullet("**Legal deliveries.** Wides, no-balls and the dataset's 'other' and 'Penalty' extras are not legal balls. With this definition no innings exceeds 120 legal balls, which validates the rule."));
add(Bullet("**Extras runs.** The raw data records the type of extra but not how many runs it cost. We estimated runs per extra type by regressing 'won by N runs' margins on the difference in extras counts (385 matches; median absolute error 1.66 runs). Rebuilt totals agree with the recorded winner in 95.4% of matches, and the wickets in hand match the recorded margin in 99.7% of chases."));
add(Bullet("**Credit rules.** Byes and leg-byes count towards the team score but are charged to neither batter nor bowler. Run-outs and retirements reduce the batting side's wickets but are not credited to the bowler."));
add(Bullet("**Player identity.** All aggregation uses `player_id`, never names. Seven names map to more than one ID; e.g. **Rashid Khan (ID 4722, Afghanistan / Gujarat Titans)** is a different player from Rashid Khan (ID 20172, Nepal). We also corrected five clerical errors (one missing dismissal ID, four non-striker typos); each is logged in `outputs/tables/data_edits_log.csv`."));

// ---------- 2. pressure ----------
add(H1("2. Defining pressure"));
add(P("Pressure is not one number on a scoreboard: 30 needed off 18 balls with eight wickets in hand is very different from the same equation with two wickets left. Rather than hand-weight required rate, wickets and balls remaining, we let match outcomes decide how much each combination matters. That takes two steps."));
add(H2("2.1 Step 1: a calibrated win-probability model"));
add(P("We fit separate logistic regressions for each innings, predicting whether the batting side wins from the state before each ball:"));
add(Bullet("**Innings 1:** runs scored, wickets in hand, balls left, current run rate, venue scoring baseline (average first-innings total at that ground), competition tier."));
add(Bullet("**Innings 2:** runs needed, balls left, wickets in hand, required rate, current rate, the gap between them, competition tier."));
add(P(`Every prediction is **out-of-fold**: the model never sees the match it is scoring (5-fold cross-validation grouped by match, so balls from one game never sit in both training and test). Overall Brier score is **0.177** against 0.250 for a coin-flip baseline. On the unseen 2026 season the chase model scores 0.126. Calibration stays within ±0.04 in every probability decile (Figure 1). Logistic regression beat gradient-boosted trees out-of-fold, because win probability is a smooth, monotonic function of the scoreboard.`));
add(Img("outputs/figures/phase1_calibration_plot.png", 300, 257));
add(Caption("Figure 1. Out-of-fold calibration of the win-probability model: predicted vs observed win rate in ten probability bins."));

add(H2("2.2 Step 2: the Leverage Index"));
add(P("For each delivery *t* we take the current win probability *P*ₜ, simulate each of the seven possible outcomes *o* with its empirical frequency in that kind of situation (*p*(*o* | state), estimated from balls with similar balls-left, wickets-left and rate-gap bands), and compute the expected absolute change in win probability:"));
add(Formula("Swingₜ = Σₒ p(o | stateₜ) · | P(stateₜ after o) − Pₜ |        LIₜ = Swingₜ / mean(Swing)"));
add(P("So LI = 1 is an average ball, and LI = 3 is a ball where the match is three times as likely to swing. LI combines required rate, wickets, balls remaining and closeness of the contest into one quantity, weighted by how much each actually moves results, and it treats batting and bowling symmetrically: the same ball is equally high-stakes for both players involved."));

add(H2("2.3 From LI to pressure tiers"));
add(Bullet("**Liveness gate.** A ball can only be high pressure if the batting side's win probability is between 5% and 95%. Garbage time (e.g. 80 needed off 15) is excluded, even if a formula would score it as 'desperate'."));
add(Bullet("**Early-innings exclusion.** The first 12 legal balls of each innings are excluded from both tiers. Run rates there rest on one or two overs and produce artificial LI spikes."));
add(Bullet(`**HIGH pressure** = top 20% of LI among the remaining live balls (LI ≥ 1.47; ${Number(tHigh.Deliveries).toLocaleString()} deliveries).`));
add(Bullet(`**ORDINARY** = bottom 50% of LI among the same live balls (LI < 0.90; ${Number(tOrd.Deliveries).toLocaleString()} deliveries). The ordinary baseline is therefore *competitive* cricket that is simply low-stakes, not dead rubbers. Live balls in the 50th–80th percentile band are left out of both tiers to keep a clear gap between them.`));
add(Tbl([{ h: "Tier", w: 34 }, { h: "Balls", w: 11, align: "right" }, { h: "Mean LI", w: 10, align: "right" }, { h: "% chase", w: 10, align: "right" }, { h: "% death overs", w: 12, align: "right" }, { h: "RPO", w: 9, align: "right" }, { h: "Wkt %", w: 9, align: "right" }, { h: "Mean |ΔWP|", w: 12, align: "right" }],
  tiers.map(t => [t.Tier, Number(t.Deliveries).toLocaleString(), f(t.Mean_LI), f(t.Share_Innings2_pct, 1), f(t.Share_Death_Overs_pct, 1), f(t.Runs_per_Over), f(t.Wicket_pct), f(t.Mean_abs_dWP, 4)]),
  { highlight: i => i === 0 }));
add(Caption("Table 3. How the tiers differ. HIGH balls move win probability 2.3× more than ORDINARY balls on average (mean |ΔWP| 0.0247 vs 0.0108). They are more often in chases and death overs, and produce more wickets. Death overs = overs 16–20."));
add(Img("outputs/figures/final_li_by_over.png", 560, 245));
add(Caption("Figure 2. Mean Leverage Index by over among live balls. In chases, leverage climbs steeply from over 16 and averages above 6 in the 20th over. First-innings leverage is flatter, because the eventual target is still open."));

add(H2("2.4 Why this definition, and what it beat"));
add(P("We also built a conventional, hand-weighted pressure index (the CCPI: 0–100 scores from required-rate difficulty, wickets lost and innings stage, with separate batting and bowling versions) and tested it against LI. The two overlap on only 15% of high-pressure batting balls. The disagreements show why we chose LI:"));
add(Bullet("Formula indices give maximum pressure to hopeless chases (e.g. 588 required run rate off the last ball). LI correctly gives these zero, because nothing can change the result."));
add(Bullet("In a genuinely balanced finish (e.g. 15 needed off 12 with three or four wickets in hand, where observed mean LI is 2.2–3.2) every boundary or wicket swings the result hard. LI scores this as extreme; formula indices score it as only moderate, because no single component looks alarming."));
add(Bullet("Tuning the formula weights to reproduce LI (fixed, equal and non-negative least-squares weights) failed: R² was negative for every innings and role, so no linear mix of the usual ingredients reproduces actual match leverage."));
add(P("Phase 2 sanity checks support LI: chases that finished close have 2.5× the mean LI of lopsided ones (1.42 vs 0.57), and 58% of live death-over chase balls are classified HIGH."));

// ---------- 3. performance ----------
add(H1("3. Measuring performance"));
add(H2("3.1 The outcome: win probability added (WPA)"));
add(P("For every delivery we record ΔWP = *P*(after the ball) − *P*(before the ball) from the batting side's view. The batter is credited with +ΔWP and the bowler with −ΔWP. Unlike strike rate or economy, WPA is automatically context-aware: a single when 2 are needed off 1 ball is worth far more than a single in the 8th over, and a dot ball at the death in a tight chase is worth more than a wicket in a lost cause. The final ball of a chase settles to 0 or 1, and the last ball of innings 1 is linked to the first ball of innings 2."));
add(H2("3.2 Opponent- and context-adjusted ratings"));
add(P("Raw averages flatter players who happened to face weak attacks or bat on flat grounds. We therefore estimate each player's rating with a **ridge regression** (similar to adjusted plus-minus models in basketball and the RAPM family of ratings):"));
add(Formula("± ΔWPᵢ = β₀ + θ(player) + γ(opponent) + controls(innings, competition tier, batting-order band, spin/pace, venue baseline) + εᵢ"));
add(Bullet("**θ(player)** is the player's adjusted win probability added per ball. For batters the opponent is the bowler; for bowlers it is the batter."));
add(Bullet("The ridge penalty (α = 300 for batting, 500 for bowling) was chosen by match-grouped 5-fold cross-validation. It shrinks low-sample players towards zero, which acts as a built-in sample-size correction. Ridge estimates agree closely with an independent empirical-Bayes shrinkage of raw means (r = 0.92 batting, 0.93 bowling)."));
add(Bullet("Wides are excluded from batters' ratings (the batter cannot score from them) but count against bowlers."));
add(H2("3.3 Pressure rating, ordinary baseline, resilience and consistency"));
add(P("We fit the model twice, once on HIGH balls and once on ORDINARY balls, which gives each player four components:"));
add(Tbl([{ h: "Component", w: 22 }, { h: "Definition", w: 60 }, { h: "Weight", w: 10, align: "center" }], [
  ["**Pressure rating**", "Adjusted WPA per ball on HIGH-pressure deliveries (θ from the HIGH-ball model).", "50%"],
  ["**Consistency**", "Share of the player's matches with HIGH-pressure involvement in which the player's net WPA on those balls was positive. It rewards delivering repeatedly, not in one freak innings.", "25%"],
  ["**Resilience**", "Pressure rating minus ordinary rating: how much better (or worse) the player is when it matters than their own competitive baseline.", "25%"],
  ["Ordinary rating", "Adjusted WPA per ball on ORDINARY deliveries. It serves as the player's baseline and enters through resilience.", "n/a"],
]));
add(Formula("Composite = 0.50 · z(pressure rating) + 0.25 · z(consistency) + 0.25 · z(resilience)"));
add(P("Each component is standardised (z-scored) within the eligible pool before weighting, so no component dominates because of its units. Players are ranked on the composite point estimate. Pressure performance carries the largest weight because the brief asks who *delivers* under pressure. Resilience keeps the comparison anchored to each player's own usual level, as the problem statement asks, and stops the list from simply reproducing 'the best players overall'. Consistency protects against one-match outliers."));

// ---------- 4. fairness ----------
add(H1("4. Making comparisons fair"));
add(Tbl([{ h: "Issue", w: 22 }, { h: "How we handled it", w: 78 }], [
  ["**Sample size**", `Eligibility requires ≥ 60 HIGH-pressure balls across ≥ 15 different matches (eligible pool: ${nBat} batters, ${nBowl} bowlers). Ridge shrinkage further pulls thin samples towards average. Composite scores are only weakly correlated with the number of pressure balls (r = 0.20 batting, 0.19 bowling), so the ranking mainly rewards efficiency, not volume. We tested cut-offs of 40/50/60/80 balls and 10/15/20 matches. The match requirement binds far more than the ball requirement, and 15 matches guards against one-series wonders without emptying the pool.`],
  ["**Match context**", "WPA already reflects the match situation. The regression also controls for innings (setting vs chasing), batting-order band (1–3, 4–7, 8–11), bowler type (spin vs pace) and the venue's scoring baseline. Opponent effects are estimated jointly, so beating Bumrah counts for more than beating an associate seamer."],
  ["**IPL vs T20I**", "The tiers score very differently (IPL 8.81 RPO, Full-Member T20I 8.11, Associate T20I 7.22). We tag every match as IPL, T20I Full-Member v Full-Member, or T20I involving an Associate, and include the tier in both the win-probability model and the rating regression. Pressure is defined relative to each tier's own scoring environment, and IPL and T20I balls are pooled into one rating per player, so no player is rewarded or penalised for the competition they played in."],
  ["**Dead games**", "Balls with win probability below 5% or above 95% are excluded from both tiers (25,259 deliveries), so neither the pressure rating nor the ordinary baseline is inflated by garbage-time hitting or bowling."],
  ["**Roles**", "Batting and bowling are rated separately, from the same balls and the same pressure definition. All-rounders (Rashid Khan, Krunal Pandya) can, and do, appear in both lists on separate evidence."],
]));
add(Caption("Table 4. Fair-comparison controls."));

// ---------- 5. results ----------
add(new Paragraph({ children: [new PageBreak()] }));
add(H1("5. Results: the top 10s"));
add(P("Ratings are shown as win probability added per 1,000 balls, i.e. ×1000. A pressure rating of 5.79 means Rinku Singh adds about 0.58 percentage points of win probability per high-pressure ball over an average batter in the same situation and against the same bowler. Across a 12-ball finish that is roughly 7 percentage points."));
const rCols = [{ h: "#", w: 4, align: "center" }, { h: "Player", w: 22 }, { h: "HIGH balls", w: 9, align: "right" }, { h: "Matches", w: 9, align: "right" }, { h: "% chase", w: 8, align: "right" }, { h: "Pressure ×1000", w: 11, align: "right" }, { h: "Ordinary ×1000", w: 11, align: "right" }, { h: "Resilience ×1000", w: 11, align: "right" }, { h: "Consist. %", w: 9, align: "right" }, { h: "Composite", w: 10, align: "right" }, { h: "P(top 10)", w: 9, align: "right" }];
const rRows = (list, role) => list.map(r => [r.rank, `**${r.name}**`, r.pressure_balls, r.matches, f(r.pct_chase, 0), k(r.pressure_rating), k(r.ordinary_rating), k(r.resilience), f(r.consistency, 0), f(r.composite), f(bootOf(role, r.player_id).top10_prob_pct, 0) + "%"]);
add(H2("5.1 Batters"));
add(Tbl(rCols, rRows(bat, "bat"), { size: 16 }));
add(Caption("Table 5. Top 10 batters under pressure. P(top 10) = share of 200 match-level bootstrap resamples in which the player ranks in the top 10 (Section 6.1)."));
const bbCols = [{ h: "#", w: 4, align: "center" }, { h: "Batter", w: 20 }, { h: "SR: HIGH", w: 10, align: "right" }, { h: "SR: ORD", w: 10, align: "right" }, { h: "Balls/out: HIGH", w: 12, align: "right" }, { h: "Balls/out: ORD", w: 12, align: "right" }, { h: "Bdry %: HIGH", w: 11, align: "right" }, { h: "Bdry %: ORD", w: 11, align: "right" }];
add(Tbl(bbCols, batBox.map(r => [r.rank, r.name, f(r.HIGH_SR, 0), f(r.ORD_SR, 0), r.HIGH_balls_per_dismissal ? f(r.HIGH_balls_per_dismissal, 1) : "–", r.ORD_balls_per_dismissal ? f(r.ORD_balls_per_dismissal, 1) : "–", f(r.HIGH_boundary_pct, 1), f(r.ORD_boundary_pct, 1)]), { size: 16 }));
add(Caption(`Table 6. Plain box-score check (unadjusted). Across all batters, strike rate is ${f(popRef.bat_HIGH_SR, 0)} on HIGH balls vs ${f(popRef.bat_ORD_SR, 0)} on ORDINARY balls. All ten of our batters score faster under pressure than in their own ordinary baseline.`));
add(H3("What separates them"));
add(Bullet("**Rinku Singh (1)** has the highest pressure rating in the pool with a near-zero ordinary rating, so their value is concentrated in pressure moments. Strike rate rises from 138 to 161 and boundary rate from 17% to 23% under pressure. He also has the best bootstrap median rank (5) among the batters."));
add(Bullet("**Glenn Maxwell (2)** and **Krunal Pandya (3)** have negative ordinary ratings but strong pressure ratings: the clearest 'rise to the occasion' profiles. Krunal also lasts twice as long under pressure (a dismissal every 31 balls vs 15)."));
add(Bullet("**Suryakumar Yadav (6)** and **Jos Buttler (10)** carry the largest pressure samples (485 and 352 balls) and the tightest bootstrap intervals. They are proven performers whose pressure edge over their own high baseline is smaller."));
add(Bullet("**Rashid Khan (4)** and **Harry Brook (7)** earn their places on strong per-ball numbers from the smallest samples (98 and 140 balls). Their uncertainty is highest, which we flag openly in Section 6."));

add(H2("5.2 Bowlers"));
add(Tbl(rCols.map(c => c.h === "Player" ? { ...c, h: "Player" } : c), rRows(bowl, "bowl"), { size: 16 }));
add(Caption("Table 7. Top 10 bowlers under pressure. Ratings are win probability *denied* per ball."));
const wbCols = [{ h: "#", w: 4, align: "center" }, { h: "Bowler", w: 22 }, { h: "Econ: HIGH", w: 10, align: "right" }, { h: "Econ: ORD", w: 10, align: "right" }, { h: "Balls/wkt: HIGH", w: 12, align: "right" }, { h: "Balls/wkt: ORD", w: 12, align: "right" }, { h: "Dot %: HIGH", w: 10, align: "right" }, { h: "Dot %: ORD", w: 10, align: "right" }];
add(Tbl(wbCols, bowlBox.map(r => [r.rank, r.name, f(r.HIGH_economy), f(r.ORD_economy), r.HIGH_balls_per_wicket ? f(r.HIGH_balls_per_wicket, 1) : "–", r.ORD_balls_per_wicket ? f(r.ORD_balls_per_wicket, 1) : "–", f(r.HIGH_dot_pct, 1), f(r.ORD_dot_pct, 1)]), { size: 16 }));
add(Caption(`Table 8. Plain box-score check (unadjusted). League economy is ${f(popRef.bowl_HIGH_econ)} on HIGH balls vs ${f(popRef.bowl_ORD_econ)} on ORDINARY. Nine of our ten take wickets more often under pressure than in their own baseline. The exception, Bumrah, wins through run denial instead.`));
add(H3("What separates them"));
add(Bullet("**Josh Hazlewood (1)** concedes at the same rate under pressure but takes a wicket every 10.2 balls instead of every 19.3, and wickets are what swing tight games. He ranks in the top 10 in 84% of resamples."));
add(Bullet("**Jasprit Bumrah (2)** shows the opposite pattern: economy *improves* to 7.00 under pressure (from 7.92), the second-lowest of the ten, so the value comes from shutting the game down. **Lockie Ferguson (6)** does both: economy falls from 9.99 to 7.69 and wickets come more often."));
add(Bullet("**Rashid Khan (3)** and **Varun Chakravarthy (4)** combine both routes: lower economy and more frequent wickets under pressure. Spin is well represented (Rashid, Varun, Chahal, Motie, Krunal): their wickets in the middle-to-late overs of tight chases carry high leverage."));
add(Bullet("**Yuzvendra Chahal (5)**, **Matheesha Pathirana (7)** and **Harshit Rana (10)** are wicket-takers: under pressure they strike roughly every 10–11 balls, about twice as often as usual, while their economy barely changes. In tight games the model values those wickets more than the runs."));
add(Img("outputs/figures/final_pressure_vs_ordinary.png", 620, 273));
add(Caption("Figure 3. Pressure rating vs ordinary rating for every eligible player (grey) and our top 10 (coloured). Points above the dashed diagonal perform better under pressure than in their own ordinary baseline. All twenty selections sit clearly above it."));
add(H3("Next in line (ranks 11–15)"));
add(P("**Batting:** Ishan Kishan, Rovman Powell, Will Jacks, KL Rahul, Pathum Nissanka. **Bowling:** Jofra Archer, Adam Zampa, Jacob Duffy, Mohsin Khan, Alzarri Joseph. Kishan and Powell are near-ties with Buttler at #10: in the bootstrap they make the top 20 at least as often as several of our lower-ranked batters (Table 9)."));

// ---------- 6. reliability ----------
add(new Paragraph({ children: [new PageBreak()] }));
add(H1("6. Reliability and limitations"));
add(P("All checks in this section were re-run on exactly the final definition and reproduce the submitted top-10 order (script `s11_final_validation.py`)."));
add(H2("6.1 Match-level bootstrap"));
add(P("We resampled matches with replacement 200 times, keeping whole matches together so that within-match correlation is preserved. Each time we refit the full pipeline (both ridge models, eligibility rules and composite) and recorded every player's rank."));
add(Img("outputs/figures/final_bootstrap_rank_intervals.png", 620, 279));
add(Caption("Figure 4. Bootstrap rank distributions. Bars show the 5th–95th percentile rank, circles the median, diamonds the submitted rank. Dashed line = rank 10.5."));
const bCols = [{ h: "#", w: 4, align: "center" }, { h: "Batter", w: 18 }, { h: "Median", w: 8, align: "right" }, { h: "P(top 10)", w: 9, align: "right" }, { h: "P(top 20)", w: 9, align: "right" }, { h: "#", w: 4, align: "center" }, { h: "Bowler", w: 19 }, { h: "Median", w: 8, align: "right" }, { h: "P(top 10)", w: 9, align: "right" }, { h: "P(top 20)", w: 9, align: "right" }];
const bb = boot.filter(b => b.role === "bat" && Number(b.rank) <= 10), bw = boot.filter(b => b.role === "bowl" && Number(b.rank) <= 10);
add(Tbl(bCols, bb.map((b, i) => [b.rank, b.name, f(b.median_boot_rank, 1), f(b.top10_prob_pct, 0) + "%", f(b.top20_prob_pct, 0) + "%", bw[i].rank, bw[i].name, f(bw[i].median_boot_rank, 1), f(bw[i].top10_prob_pct, 0) + "%", f(bw[i].top20_prob_pct, 0) + "%"]), { size: 16 }));
add(Caption("Table 9. Bootstrap summary. Medians sit above the submitted ranks because any player at the top of a noisy list tends to regress in a resample."));
add(P("**Reading this honestly:** the top five bowlers are well separated (each in the top 10 in at least half the resamples, Hazlewood in 84%). Pathirana, Motie and Rana are less secure. For batting, Rinku Singh, Suryakumar Yadav, Tim David, Glenn Maxwell and David Miller are the most robust selections. Harry Brook, Rashid Khan and Sam Curran rest on small pressure samples and have wide intervals. We kept the point-estimate order rather than re-sorting by bootstrap medians, because the medians are themselves noisy and the point estimate uses all the data."));

add(H2("6.2 Sensitivity to our own choices"));
add(Tbl([{ h: "Variant", w: 30 }, { h: "Role", w: 11 }, { h: "Top-10 overlap", w: 13, align: "right" }, { h: "Our top 10 in variant's top 20", w: 18, align: "right" }, { h: "Rank corr. (Spearman)", w: 14, align: "right" }],
  [...vars_.map(v => [v.Variant, v.Role, `${v.Top10_overlap}/10`, `${v.Top10_in_variant_top20}/10`, f(v.Spearman_all_common)]),
   ["Composite weights randomised (200 draws around 50/25/25)", "Batting", `${f(batOverlap, 1)}/10 avg`, "n/a", "n/a"],
   ["Composite weights randomised (200 draws around 50/25/25)", "Bowling", `${f(bowlOverlap, 1)}/10 avg`, "n/a", "n/a"]]));
add(Caption("Table 10. How the top 10s change under alternative definitions. Rank correlation is computed over all players eligible in both versions."));
add(Bullet("The **bowling** list is stable across every variant: 8/10 overlap in each, with rank correlations of 0.83–0.94."));
add(Bullet("The **batting** list is robust to the score weights (8.6/10 average overlap) and to a wider HIGH band (7/10). It moves more under a narrower HIGH band (5/10) or chases only (6/10), and it is most sensitive to the early-innings exclusion: without it, the top 10 fills with openers whose first-two-over balls receive inflated leverage. We regard that as an artefact the exclusion is designed to remove, not as evidence against the final list."));
add(Bullet("Jos Buttler at #10 is the most weight-dependent selection (top 10 in 34% of weight draws). Every other batter is in the top 10 in at least 71% of draws, and every bowler in at least 78%."));

add(H2("6.3 Is pressure performance a repeatable skill?"));
add(Tbl([{ h: "Check", w: 52 }, { h: "Batting r", w: 12, align: "right" }, { h: "n", w: 7, align: "right" }, { h: "Bowling r", w: 12, align: "right" }, { h: "n", w: 7, align: "right" }],
  rel.map(r => [r.Check, f(r.Batting_r), r.Batting_n, f(r.Bowling_r), r.Bowling_n])));
add(Caption("Table 11. Split-half reliability and predictive validity (Spearman rank correlations between independently fitted halves; lower eligibility thresholds within each half)."));
add(P("These are the most sobering numbers in the report, and we report them as they are. Splitting matches into odd and even IDs and rating each half separately gives almost no agreement for batters (r ≈ 0) and weak agreement for bowlers (r ≈ 0.17). Across eras (2022–23 vs 2024–26) the agreement is about 0.15–0.20. This matches what sports analytics has long found about 'clutch' ability: genuine but small, and buried in noise over the sample sizes any T20 player accumulates. One useful finding: a player's 2022–23 **pressure** rating predicts their 2024–26 pressure rating better than their all-ball rating does (batting 0.20 vs 0.12; bowling 0.17 vs 0.16). Pressure-specific measurement therefore adds information beyond general quality, especially for batters."));
add(Callout("**What the rankings are, and are not.** They are a well-controlled, uncertainty-quantified answer to *'who delivered the most match-winning value in high-pressure situations across 2022–2026, relative to their own baseline?'* They are not a precise forecast of who will be best in the next tight finish. The bowling list is materially more reliable than the batting list."));

add(H2("6.4 Known limitations"));
add(Bullet("**Extras are estimated, not recorded.** Totals are accurate to about ±1.7 runs; a handful of razor-thin finishes may have slightly misplaced win probabilities."));
add(Bullet("**First-innings leverage is weaker.** Without a target, the innings-1 model is less sharp (Brier 0.221 vs 0.129 in chases), so pressure on the side batting first is measured less precisely. This favours players with more chase exposure."));
add(Bullet("**Pressure on a ball is shared.** WPA credits each ball to one batter and one bowler. It cannot separate fielding, captaincy or the non-striker's role, and run-outs fall on the batting side only."));
add(Bullet("**No ball-tracking or match-up context** beyond spin/pace and opponent identity: pitch, dew and toss effects are only partly captured, through the venue baseline."));
add(Bullet("**Associate cricket** has fewer matches (50) and lower scoring. It is controlled for by tier, but players whose pressure balls come mostly from those games carry wider uncertainty."));
add(Bullet("**Small samples near the cut-off.** Rashid Khan (batting, 98 balls), Harry Brook (140) and Krunal Pandya (157) are ranked on less than a season's worth of high-pressure balls."));

// ---------- 7. reproduce ----------
add(H1("7. Reproducibility"));
add(P("Every number in this report is produced by the pipeline below (Python 3.10+: pandas, numpy, scipy, scikit-learn, pyarrow, matplotlib; random seed 42). Each phase writes a summary and passes explicit validation checks before the next one runs. All decisions are logged in `docs/DECISIONS.md`."));
add(Tbl([{ h: "Script", w: 30 }, { h: "What it does", w: 70 }], [
  ["s05_clean_and_match_state.py", "Scope filters, legal-ball logic, extras calibration, pre-ball match state (780 matches)."],
  ["s05b_integrity_checks.py", "Player/team/venue masters, duplicate-name audit, data corrections, competition tiers."],
  ["s06_win_probability_model.py", "Dual logistic win-probability models with out-of-fold predictions; calibration checks."],
  ["s07_pressure_definitions.py", "Leverage Index via outcome simulation; CCPI comparison; liveness gate."],
  ["s08 / s09 scripts", "Ridge ratings, cross-validated α, empirical-Bayes cross-check, earlier ranking variants."],
  ["s10_redefine_ordinary_and_update.py", "**Final definition:** live-only ORDINARY tier, early-innings exclusion; writes the submitted top 10s."],
  ["s11_final_validation.py", "Bootstrap, split-half, variants, weight sensitivity, box scores and the figures in this report, all on the final definition."],
]));
add(Caption("Table 12. Pipeline. Run the scripts in order; outputs land in outputs/tables, outputs/figures and handoff/."));

// ---------- appendix ----------
add(H1("Appendix A. Competition tier profile"));
add(Tbl([{ h: "Tier", w: 24 }, { h: "Matches", w: 10, align: "right" }, { h: "RPO", w: 9, align: "right" }, { h: "Death RPO", w: 11, align: "right" }, { h: "Boundary %", w: 12, align: "right" }, { h: "Avg 1st-inn", w: 12, align: "right" }, { h: "Chase win %", w: 12, align: "right" }],
  tierCmp.map(t => {
    const g = (...ks) => { for (const kk of ks) if (t[kk] !== undefined) return t[kk]; return ""; };
    return [g("Competition Tier"), g("Matches"), g("Overall RPO"), g("Death RPO"), g("Boundary %"), g("Avg 1st Inn Score"), g("Chase Win %")];
  })));
add(Caption("Table A1. The scoring environment differs sharply by tier, which is why tier is controlled for throughout."));

const doc = new Document({
  creator: "SupeRR Selector S2 participant",
  title: "SupeRR Selector S2 – Round 1 Methodology Report",
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 32, bold: true, color: INK }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 25, bold: true, color: ACCENT }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true, color: INK }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 260 } } } }, { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 260 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "SupeRR Selector S2 · Round 1 · Methodology report", size: 16, color: MUTED })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: MUTED })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
