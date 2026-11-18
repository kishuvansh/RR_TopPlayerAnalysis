# Known Limitations & Data Weaknesses

1. **Unrecorded Extras Run Amounts:** The raw dataset records category enums for extras rather than exact runs surrendered. Run amounts were reconstructed via match-margin linear regression on 385 defended matches (median absolute residual: 1.656 runs).
2. **Current Run Rate Boundary Volatility:** In the first 12 balls of an innings, sample size is insufficient to compute stable run rates, producing artificial leverage spikes in ball 1. This was resolved via the early-innings exclusion rule (`legal_balls_before >= 12`).
3. **Duplicate Player Names:** 7 duplicate name pairs exist across database IDs. In all cases, secondary IDs had 0 balls in clean regulation matches or represented different individuals across countries.
4. **Associate Competition Depth:** Associate-nation matches feature lower par totals and higher leverage concentration. While controlled via `comp_tier_pooled` fixed effects, rankings for players facing primarily associate bowling carry wider confidence bounds.
