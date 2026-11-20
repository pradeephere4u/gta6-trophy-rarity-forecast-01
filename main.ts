// GTA VI trophy rarity forecast — gta6-trophy-rarity-forecast-01
//
// Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
// rate on Exophase at the case's target, as five quantiles (levels 0.05, 0.25,
// 0.5, 0.75, 0.95). Node 22, standard library only, no network, fully
// deterministic: two runs on the same case write byte-identical output.
//
// The model, per trophy, on the logit scale (logit(p) = ln(p / (1 - p))):
//   * Collapse the history to distinct snapshots — Exophase refreshes its
//     counts on its own schedule, so consecutive identical reads carry no new
//     information; a refresh shows up as a jump in players_tracked.
//   * Fit a weighted least-squares line of logit(rate) against
//     ln(1 + hours since launch), weights = inverse binomial variance
//     (N * p * (1 - p)), so precise late reads count more than the noisy
//     launch-night ones. The slope is shrunk toward the trophy's tier mean
//     (empirical-Bayes, tau = 0.5), because a single trophy's slope over a
//     short window is noisy and tiers behave alike.
//   * Extrapolate the fitted line to the target. The move from the last read
//     is capped at 3 logit units so a steep short-window slope cannot explode
//     over a multi-day horizon. With fewer than 3 distinct snapshots, or no
//     variation at all (a flat launch-night history), fall back to
//     persistence: the median stays at the last read and the spread carries
//     the uncertainty.
//   * Spread (sigma, logit units) combines four honest sources: a random-walk
//     term growing with the horizon (0.5 per sqrt day), a refresh-jump term
//     (the RMS of the logit jumps the history itself shows at refreshes, or
//     a 0.8 prior when none has been seen), the trophy's own fit residual,
//     and the slope's estimation error projected to the target. The band is
//     floored at 0.35 and capped at 3.0 logit units.
//   * q_tau = logistic(logit(median) + z_tau * sigma), rounded to 4 decimals.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Launch moment: rates are modelled against hours since this instant. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** Random-walk spread, logit units per square-root day. */
const SIGMA_RW = 0.5;
/** Prior refresh-jump spread when the history shows no refresh yet. */
const SIGMA_REFRESH_PRIOR = 0.8;
/** Prior spread for the slope of a trophy's trend (empirical-Bayes tau). */
const SLOPE_TAU = 0.5;
/** Widest move the fitted trend may add from the last read, logit units. */
const MAX_TREND_MOVE = 3.0;
/** Spread floor and cap, logit units. */
const SIGMA_MIN = 0.35;
const SIGMA_MAX = 3.0;

interface Trophy {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const logisticPct = (y: number): number => clampPct(100 / (1 + Math.exp(-y)));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from launch to an ISO timestamp (floored at -0.99 so ln stays defined). */
const hoursSinceLaunch = (iso: string): number =>
  Math.max(-0.99, (Date.parse(iso) - LAUNCH_MS) / 3600000);
/** x-axis value for a read: ln(1 + hours since launch). */
const xOf = (iso: string): number => Math.log(1 + hoursSinceLaunch(iso));

interface Snapshot {
  at: string;
  x: number;
  players: number;
  rates: Array<number | null>;
}

/**
 * Collapse consecutive reads that show the same page (same players_tracked
 * and same rates) into one snapshot, keeping the first occurrence. Between
 * Exophase refreshes the page is stale, so repeats add no information and
 * would only overweight the stale level in the fit.
 */
function distinctSnapshots(history: Read[]): Snapshot[] {
  const out: Snapshot[] = [];
  let prevKey = '';
  for (const r of history) {
    const key = `${r.players_tracked}|${r.rates.join(',')}`;
    if (key === prevKey) continue;
    prevKey = key;
    out.push({ at: r.at, x: xOf(r.at), players: r.players_tracked, rates: r.rates });
  }
  return out;
}

interface Fit {
  /** Intercept of the fitted line, logit units. */
  a: number;
  /** Slope of the fitted line, logit units per unit of x. */
  b: number;
  /** Standard error of the slope. */
  seB: number;
  /** Residual standard deviation, logit units. */
  residSd: number;
  /** Number of points the fit used. */
  n: number;
}

/**
 * Weighted least squares of y on x. Returns null when the fit is not
 * identified (fewer than 3 points, no spread in x, or no movement in y).
 */
function wls(xs: number[], ys: number[], ws: number[]): Fit | null {
  const n = xs.length;
  if (n < 3) return null;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw <= 0) return null;
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  const a = my - b * mx;
  let sse = 0;
  let wsse = 0;
  let ySpread = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    sse += r * r;
    wsse += ws[i] * r * r;
    ySpread = Math.max(ySpread, Math.abs(ys[i] - my));
  }
  // A fit whose y never moves is no fit at all: the line is unidentified.
  if (ySpread < 1e-9) return null;
  const dof = Math.max(1, n - 2);
  const residSd = Math.sqrt(sse / dof);
  // Scale of the weighted residuals; inflates the slope error when the
  // straight line misses the data (the usual WLS variance estimate).
  const sigma2 = wsse / dof;
  const seB = Math.sqrt(Math.max(0, sigma2) / sxx);
  return { a, b, seB, residSd, n };
}

/**
 * The refresh-jump spread: how far a trophy's logit rate jumps at an
 * Exophase refresh beyond what its smooth trend predicts. Measured from the
 * history itself as the RMS of the trend-adjusted successive differences
 * between distinct snapshots, across all trophies. Falls back to the prior
 * when the history shows no refresh yet (a flat launch-night page).
 */
function refreshSigma(
  snaps: Snapshot[],
  trophies: Trophy[],
  fits: Array<Fit | null>,
): number {
  if (snaps.length < 2) return SIGMA_REFRESH_PRIOR;
  let ss = 0;
  let count = 0;
  for (let i = 0; i < trophies.length; i++) {
    const b = fits[i]?.b ?? 0;
    for (let j = 1; j < snaps.length; j++) {
      const r0 = snaps[j - 1].rates[i];
      const r1 = snaps[j].rates[i];
      if (r0 === null || r1 === null) continue;
      const d = logit(r1) - logit(r0) - b * (snaps[j].x - snaps[j - 1].x);
      ss += d * d;
      count++;
    }
  }
  if (count === 0) return SIGMA_REFRESH_PRIOR;
  const sigma = Math.sqrt(ss / count);
  return Math.min(1.5, Math.max(0.2, sigma));
}

/**
 * Precision-weighted mean slope per tier, so each trophy's slope can be
 * shrunk toward what its own tier does. A trophy with a precisely measured
 * slope keeps it; a noisy one is pulled to the tier.
 */
function tierMeanSlopes(
  trophies: Trophy[],
  fits: Array<Fit | null>,
): Map<string, number> {
  const acc = new Map<string, { sw: number; sb: number }>();
  for (let i = 0; i < trophies.length; i++) {
    const f = fits[i];
    if (f === null || !Number.isFinite(f.b) || !Number.isFinite(f.seB)) continue;
    // Cap the precision weight so one trophy cannot dominate its tier.
    const w = 1 / (f.seB * f.seB + 0.0025);
    const e = acc.get(trophies[i].tier) ?? { sw: 0, sb: 0 };
    e.sw += w;
    e.sb += w * f.b;
    acc.set(trophies[i].tier, e);
  }
  const out = new Map<string, number>();
  for (const [tier, e] of acc) {
    if (e.sw > 0) out.set(tier, e.sb / e.sw);
  }
  return out;
}

interface Row {
  id: number;
  q: number[];
}

/** Forecast one case. Never throws on thin data: it always writes a row. */
function forecastCase(c: Case): Row[] {
  const snaps = distinctSnapshots(c.history);
  const xTarget = xOf(c.target);
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const rw = SIGMA_RW * Math.sqrt(horizonDays);

  // Last known rate per trophy (last non-null read). A trophy never seen in
  // any read falls back to its tier's median last rate, else a neutral 1%.
  const lastRateRaw = (i: number): number => {
    for (let j = snaps.length - 1; j >= 0; j--) {
      const r = snaps[j].rates[i];
      if (r !== null && Number.isFinite(r)) return r;
    }
    return 1;
  };
  const known: boolean[] = c.trophies.map((_, i) => {
    for (let j = snaps.length - 1; j >= 0; j--) {
      const r = snaps[j].rates[i];
      if (r !== null && Number.isFinite(r)) return true;
    }
    return false;
  });
  const tierFallback = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const vals = c.trophies
      .filter((t, i) => t.tier === tier && known[i])
      .map((_, i) => lastRateRaw(i))
      .sort((a, b) => a - b);
    if (vals.length > 0) tierFallback.set(tier, vals[Math.floor(vals.length / 2)]);
  }
  const lastRate: number[] = c.trophies.map((t, i) =>
    known[i] ? lastRateRaw(i) : (tierFallback.get(t.tier) ?? 1),
  );
  const xLast = snaps.length > 0 ? snaps[snaps.length - 1].x : 0;

  // First pass: one trend fit per trophy, on the distinct snapshots.
  const fits: Array<Fit | null> = c.trophies.map((_, i) => {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    for (const s of snaps) {
      const r = s.rates[i];
      if (r === null || !Number.isFinite(r)) continue;
      const p = clampPct(r) / 100;
      xs.push(s.x);
      ys.push(logit(r));
      // Inverse binomial variance of the logit: precise late reads weigh more.
      ws.push(Math.max(1, s.players) * p * (1 - p));
    }
    return wls(xs, ys, ws);
  });
  const tierMeans = tierMeanSlopes(c.trophies, fits);
  const sigmaRefresh = refreshSigma(snaps, c.trophies, fits);
  const dx = Math.max(0, xTarget - xLast);

  // Second pass: median and spread per trophy.
  return c.trophies.map((t, i) => {
    const f = fits[i];
    const base = logit(lastRate[i]);
    let medianLogit = base;
    let resid = 0.5;
    let slopeUnc = 0;
    if (f !== null) {
      const tierMean = tierMeans.get(t.tier);
      let b = f.b;
      if (tierMean !== undefined && Number.isFinite(tierMean)) {
        // Empirical-Bayes shrinkage of the slope toward the tier mean.
        const prec = 1 / (f.seB * f.seB + 1e-9);
        const precPrior = 1 / (SLOPE_TAU * SLOPE_TAU);
        b = (f.b * prec + tierMean * precPrior) / (prec + precPrior);
      }
      // The trend's move from the last read, capped so a steep short-window
      // slope cannot explode over a multi-day horizon.
      const move = Math.min(MAX_TREND_MOVE, Math.max(-MAX_TREND_MOVE, b * dx));
      medianLogit = base + move;
      resid = Math.min(1.5, Math.max(0.15, f.residSd));
      slopeUnc = Math.min(1.0, f.seB * dx);
    }
    const sigma = Math.min(
      SIGMA_MAX,
      Math.max(SIGMA_MIN, Math.sqrt(rw * rw + sigmaRefresh * sigmaRefresh + resid * resid + slopeUnc * slopeUnc)),
    );
    const q = Z.map((z) => round4(logisticPct(medianLogit + z * sigma)));
    // Rounding can only tie, never invert; enforce it explicitly anyway.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
