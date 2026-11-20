# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pradeephere4u/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pradeephere4u/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on Exophase at 2026-11-26T00:00:00Z — five quantiles (5%, 25%, 50%, 75%, 95%)
per trophy. Built by an AI agent for make.fun's
[“Forecast how many GTA VI players earn each trophy in week one” bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01),
$250 in USDC.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` is a Node 22 program (standard library only, no network). It reads
every `<id>.in` case file in the input directory and writes one
`<id>.out` forecast per case: exactly
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}` — five quantiles
of trophy `n`'s earn rate in percent, non-decreasing, one row per trophy.
Runs are fully deterministic: two runs on the same case write byte-identical
output, well inside the 8-second per-case cap.

## The model

Per trophy, on the logit scale:

- **Distinct snapshots.** Exophase refreshes its counts on its own schedule,
  so consecutive identical reads are collapsed — a refresh shows up as a jump
  in players tracked, and repeats between refreshes carry no information.
- **Trend fit.** A weighted least-squares line of logit(rate) against
  ln(1 + hours since launch), weights = inverse binomial variance, so precise
  late reads count more than noisy launch-night ones. The slope is shrunk
  toward the trophy's tier mean (empirical-Bayes), and the move from the last
  read is capped at 3 logit units so a steep short-window slope cannot explode
  over a multi-day horizon. With fewer than 3 distinct snapshots, or no
  variation at all (a flat launch-night history), it falls back to
  persistence: the median stays at the last read and the spread carries the
  uncertainty.
- **Honest spread.** Sigma combines a random-walk term growing with the
  horizon, a refresh-jump term measured from the history's own logit jumps
  (0.8 prior when no refresh has been seen), the trophy's fit residual, and
  the slope's estimation error projected to the target — floored at 0.35 and
  capped at 3.0 logit units.

## Local validation

- Practice cases `p00`/`p01`: exact output shape, 55 trophies each, quantiles
  ordered, 2,637 bytes per file, two runs byte-identical, ~0.1 s per case.
  Their histories are completely flat (one distinct snapshot), so the
  forecast correctly falls back to persistence with a wide band — much wider
  than the house baseline's, which is what the known launch-night refresh
  demands.
- Synthetic rich-history case (119 reads over 2.5 days, player base growing
  638 → 125,000, refreshes every 6 h, scored against the simulated truth at
  the target with the contest's own pinball loss): loss **0.332** vs the house
  baseline's **0.582** — 43% lower — with typical miss 1.63 points vs 10.44.
- Edge cases (single read, empty history, null rates) all produce valid output
  without crashing.

## License

MIT — see [LICENSE](LICENSE).
