/**
 * The eval harness over {@link ./metrics} — turn a RANKED LIST plus a way to
 * judge it into a relevance verdict.
 *
 * `metrics.ts` shipped the formulas (nDCG@10, P@5, acc@5) and nothing in either
 * consumer repo ever called them: a metric contract with no harness is a
 * calculator, not an eval. This is the harness half.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY THIS ADDS `recallAtK` — the shipped metrics CANNOT SEE A MISS
 *
 * Every function in `metrics.ts` scores only the list it is handed. `ndcg`
 * normalises against the ideal ordering OF THOSE SAME GRADES, so a list that is
 * perfectly ordered scores **1.0 whether it contains all of the relevant
 * documents or one of them**. `precisionAtK` and `accessoryAtK` are likewise
 * within-list. That is correct for what they measure — ORDERING and POLLUTION —
 * and it makes the whole family structurally blind to RECALL.
 *
 * Which is the half of a search failure users actually report. "The search did
 * not find the conversation I remember" is a recall complaint, and pinning it
 * with nDCG alone yields a metric that stays green through exactly the
 * regression it was added to catch.
 *
 * So recall is measured against an EXTERNALLY KNOWN relevant set, and it is the
 * one number here that cannot be computed from the ranking alone.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY `coverage` IS A FIELD AND NOT A COMMENT
 *
 * Both failure directions of this harness produce a number that LOOKS like a
 * verdict:
 *
 *   - an empty ranking scores `ndcg → 0`, indistinguishable from "retrieval
 *     worked and everything it found was irrelevant";
 *   - a list where nothing is relevant ALSO scores 0 (`idcg === 0`);
 *   - and a recall of 0 computed against an EMPTY ground-truth set means "there
 *     was nothing to find", not "we found nothing" — the opposite verdict.
 *
 * A caller that reads only the scores cannot tell these apart, and each one
 * reads as failure when it may be vacuity. So `coverage` states which case this
 * is, `recallAtK` is `null` rather than `0` when no ground truth was supplied,
 * and {@link assertRelevance} refuses a non-`complete` evaluation instead of
 * comparing its scores to a floor. Same rule as any absence-vs-emptiness
 * report: a zero is only evidence once you know the measurement ran.
 *
 * Pure and domain-free — no corpus, no store, no query language. The caller
 * supplies the judgement, which is what lets a judgement be DERIVED (a term
 * overlap, a provenance check, a substring scan of the corpus) instead of
 * hand-authored qrels that rot as the corpus grows.
 */

import { accessoryAtK, ndcgAtK, precisionAtK } from './metrics';

/** Immutable inputs shared by a baseline and its treatment. Versions are host-defined. */
export interface SearchEvaluationPins {
  sourceRevision: string;
  corpus: string;
  querySet: string;
  modelConfiguration: string;
}

/** One independently judged population; budgets must never be pooled across cells. */
export interface SearchEvaluationCell {
  id: string;
  product: string;
  adapter: string;
  queryClass: string;
  temperature: 'cold' | 'warm';
  attemptIds: readonly string[];
  minExecuted: number;
  minJudged: number;
  /** Reference to the accepted target or the pre-treatment target decision. */
  budgetSource: string;
  maxP95LatencyMs: number;
  maxPerAttemptCostUsd: number;
  maxErrorRate: number;
  maxDegradedRate: number;
  quality: RelevanceFloor | 'no-match';
  /** Explicit predeclared exclusions; they remain in population accounting. */
  censoring?: { attemptIds: readonly string[]; reason: string };
}

export interface SearchEvaluationDeclaration {
  declaredAtMs: number;
  pins: SearchEvaluationPins;
  cells: readonly SearchEvaluationCell[];
}

export interface SearchEvaluationAttempt {
  id: string;
  cellId: string;
  outcome: 'success' | 'timeout' | 'error' | 'unknown';
  executed: boolean;
  degraded: boolean;
  latencyMs: number | null;
  costUsd: number | null;
  /** Required to judge a successful no-match case without treating it as vacuous relevance. */
  resultCount: number | null;
  quality?: RelevanceEvalResult;
}

export interface SearchEvaluationRun {
  startedAtMs: number;
  declaration: SearchEvaluationDeclaration;
  /** Actual inputs used by the runner, independently of its copied declaration. */
  pins: SearchEvaluationPins;
  attempts: readonly SearchEvaluationAttempt[];
}

export interface SearchEvaluationAssertion {
  status: 'pass' | 'fail' | 'unknown';
  failures: string[];
  unknowns: string[];
  cells: Array<{ id: string; attempted: number; eligible: number; executed: number; judged: number; censored: number }>;
}

/** Canonical JSON identity without a crypto/runtime dependency or object-key-order sensitivity. */
function evaluationIdentity(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(evaluationIdentity).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.entries(value).filter(([, v]) => v !== undefined).sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${JSON.stringify(k)}:${evaluationIdentity(v)}`).join(',')}}`;
  }
  return JSON.stringify(value) ?? 'undefined';
}

/**
 * Validate a frozen, pinned benchmark against its complete attempt ledger.
 * A good aggregate cannot cover a missing cell. Unknown/missing attempts and
 * undersized cells stay unknown; observed budget misses and rewritten inputs fail.
 * This extends the relevance harness rather than introducing another metric definition.
 */
export function assertSearchEvaluation(
  baseline: SearchEvaluationDeclaration,
  run: SearchEvaluationRun,
): SearchEvaluationAssertion {
  const failures: string[] = [];
  const unknowns: string[] = [];
  const cells: SearchEvaluationAssertion['cells'] = [];
  const fail = (id: string, message: string) => failures.push(`${id}: ${message}`);
  const unknown = (id: string, message: string) => unknowns.push(`${id}: ${message}`);
  const nonnegative = (n: number | null): n is number => n !== null && Number.isFinite(n) && n >= 0;
  const pinNames: Array<keyof SearchEvaluationPins> = ['sourceRevision', 'corpus', 'querySet', 'modelConfiguration'];
  for (const key of pinNames) {
    if (!baseline.pins[key]?.trim()) fail('manifest', `missing ${key} pin`);
    if (run.pins[key] !== baseline.pins[key]) fail('manifest', `${key} changed`);
  }
  if (!nonnegative(baseline.declaredAtMs) || !nonnegative(run.startedAtMs) || baseline.declaredAtMs >= run.startedAtMs) {
    fail('manifest', 'declaration must precede the run');
  }
  if (evaluationIdentity(baseline) !== evaluationIdentity(run.declaration)) fail('manifest', 'declaration changed');
  if (!baseline.cells.length) unknown('manifest', 'no declared cells');

  const expected = new Map<string, string>();
  const cellIds = new Set<string>();
  const populations = new Set<string>();
  for (const cell of baseline.cells) {
    if (!cell.id.trim() || cellIds.has(cell.id)) fail(cell.id, 'missing or duplicate cell id');
    cellIds.add(cell.id);
    const population = evaluationIdentity([cell.product, cell.adapter, cell.queryClass, cell.temperature]);
    if (![cell.product, cell.adapter, cell.queryClass].every(v => v.trim()) || !['cold', 'warm'].includes(cell.temperature)) {
      fail(cell.id, 'incomplete population identity');
    }
    if (populations.has(population)) fail(cell.id, 'duplicate population');
    populations.add(population);
    if (!cell.budgetSource.trim()) fail(cell.id, 'missing budget source');
    for (const key of ['maxP95LatencyMs', 'maxPerAttemptCostUsd', 'maxErrorRate', 'maxDegradedRate'] as const) {
      if (!nonnegative(cell[key]) || (key.endsWith('Rate') && cell[key] > 1)) fail(cell.id, `invalid ${key}`);
    }
    for (const id of cell.attemptIds) {
      if (!id.trim() || expected.has(id)) fail(cell.id, 'missing or duplicate attempt identity');
      expected.set(id, cell.id);
    }
  }
  const observed = new Map<string, SearchEvaluationAttempt>();
  for (const attempt of run.attempts) {
    if (observed.has(attempt.id)) fail(attempt.cellId, `duplicate receipt ${attempt.id}`);
    observed.set(attempt.id, attempt);
    if (expected.get(attempt.id) !== attempt.cellId) fail(attempt.cellId, `undeclared or wrong-cell receipt ${attempt.id}`);
  }

  for (const cell of baseline.cells) {
    const censored = new Set(cell.censoring?.attemptIds ?? []);
    if (cell.censoring && (!cell.censoring.reason.trim() || censored.size !== cell.censoring.attemptIds.length)) {
      fail(cell.id, 'invalid censoring declaration');
    }
    for (const id of censored) if (!cell.attemptIds.includes(id)) fail(cell.id, `undeclared censor ${id}`);
    const eligible = cell.attemptIds.filter(id => !censored.has(id));
    for (const key of ['minExecuted', 'minJudged'] as const) {
      if (!Number.isInteger(cell[key]) || cell[key] < 1) fail(cell.id, `invalid ${key}`);
      else if (cell[key] > eligible.length) unknown(cell.id, `undersized declared population for ${key}`);
    }
    if (cell.quality !== 'no-match') {
      const bounds = Object.entries(cell.quality);
      if (!bounds.length) fail(cell.id, 'missing quality target');
      for (const [key, value] of bounds) {
        if (!['minNdcgAtK', 'minPrecisionAtK', 'minRecallAtK', 'maxAccessoryAtK'].includes(key) ||
            !nonnegative(value) || (key !== 'maxAccessoryAtK' && value > 1)) fail(cell.id, `invalid quality target ${key}`);
      }
    }
    let executed = 0;
    let judged = 0;
    let errors = 0;
    let degraded = 0;
    const latencies: number[] = [];
    for (const id of cell.attemptIds) {
      const attempt = observed.get(id);
      if (!attempt || attempt.cellId !== cell.id) { unknown(cell.id, `missing receipt ${id}`); continue; }
      // Even a censored record must exist. Its omission cannot shrink the population.
      if (censored.has(id)) continue;
      if (!['success', 'timeout', 'error', 'unknown'].includes(attempt.outcome)) {
        unknown(cell.id, `unrecognized outcome ${id}`);
        continue;
      }
      if (attempt.executed) executed++;
      if (attempt.outcome === 'unknown') unknown(cell.id, `unknown outcome ${id}`);
      if (attempt.outcome === 'error' || attempt.outcome === 'timeout') errors++;
      if (attempt.outcome === 'timeout') fail(cell.id, `uncensored timeout ${id}`);
      if (attempt.degraded) degraded++;
      if (nonnegative(attempt.latencyMs)) latencies.push(attempt.latencyMs);
      else unknown(cell.id, `unmeasured latency ${id}`);
      if (!nonnegative(attempt.costUsd)) unknown(cell.id, `unmeasured cost ${id}`);
      else if (attempt.costUsd > cell.maxPerAttemptCostUsd) fail(cell.id, `cost exceeded ${id}`);
      if (attempt.outcome !== 'success') continue;
      if (!attempt.executed) { fail(cell.id, `success without execution ${id}`); continue; }
      if (cell.quality === 'no-match') {
        if (attempt.resultCount === null) unknown(cell.id, `unmeasured no-match ${id}`);
        else if (attempt.resultCount !== 0) fail(cell.id, `unexpected matches ${id}`);
        else judged++;
      } else if (!attempt.quality) unknown(cell.id, `unjudged ${id}`);
      else {
        const fractions = [attempt.quality.ndcgAtK, attempt.quality.precisionAtK];
        if (attempt.quality.recallAtK !== null) fractions.push(attempt.quality.recallAtK);
        if (!fractions.every(n => nonnegative(n) && n <= 1) ||
            !Number.isInteger(attempt.quality.k) || attempt.quality.k < 1 ||
            !Number.isInteger(attempt.quality.accessoryAtK) || attempt.quality.accessoryAtK < 0 ||
            attempt.quality.accessoryAtK > attempt.quality.k ||
            !attempt.quality.grades.every(n => nonnegative(n) && n <= 3)) {
          unknown(cell.id, `invalid quality measurement ${id}`);
          continue;
        }
        const quality = assertRelevance(attempt.quality, cell.quality);
        if (!quality.ok) fail(cell.id, `quality failed ${id}: ${quality.failures.join('; ')}`);
        judged++;
      }
    }
    if (executed < cell.minExecuted || judged < cell.minJudged) unknown(cell.id, 'undersized executed/judged population');
    if (eligible.length && errors / eligible.length > cell.maxErrorRate) fail(cell.id, 'error rate exceeded');
    if (eligible.length && degraded / eligible.length > cell.maxDegradedRate) fail(cell.id, 'degraded rate exceeded');
    latencies.sort((a, b) => a - b);
    if (latencies.length && latencies[Math.ceil(latencies.length * 0.95) - 1] > cell.maxP95LatencyMs) fail(cell.id, 'p95 latency exceeded');
    cells.push({ id: cell.id, attempted: cell.attemptIds.length, eligible: eligible.length, executed, judged, censored: censored.size });
  }
  return { status: failures.length ? 'fail' : unknowns.length ? 'unknown' : 'pass', failures, unknowns, cells };
}

/**
 * Whether this evaluation MEASURED anything.
 *
 * - `complete` — the ranking was scored; the numbers mean what they say.
 * - `empty-ranking` — retrieval returned nothing. Every score is 0 by vacuity.
 * - `no-ground-truth` — a relevant set was declared but is EMPTY, so recall has
 *   no referent. Ordering/pollution scores are still meaningful; recall is not.
 *
 * `empty-ranking` wins when both apply: nothing was returned, so nothing about
 * this run is informative.
 */
export type RelevanceEvalCoverage = 'complete' | 'empty-ranking' | 'no-ground-truth';

/** The externally-known relevant set recall is measured against. */
export interface RelevanceGroundTruth<T> {
  /** Ids that SHOULD appear. Derive these where you can; hand-written qrels rot. */
  ids: readonly string[];
  /** How to read the comparable id off a ranked item. */
  idOf: (item: T) => string;
}

export interface RelevanceEvalInput<T> {
  ranked: readonly T[];
  /** Judge one result, 0..3 (the metric contract's scale). */
  grade: (item: T) => number;
  /** Cutoff for every @k metric. Default 10, matching the harness's nDCG@10. */
  k?: number;
  /** Omit when the query has no known relevant set — recall then reports null. */
  groundTruth?: RelevanceGroundTruth<T>;
  /** grade >= this counts as relevant for precision. Default 2 (the harness's P@5). */
  precisionThreshold?: number;
  /** grade <= this counts as pollution. Default 1 (the harness's acc@5). */
  accessoryThreshold?: number;
}

export interface RelevanceEvalResult {
  coverage: RelevanceEvalCoverage;
  k: number;
  /** Derived grades in RANK ORDER — the input to every metric below. */
  grades: number[];
  ndcgAtK: number;
  precisionAtK: number;
  accessoryAtK: number;
  /**
   * Fraction of the ground-truth ids present in the top `k`.
   *
   * ⚠ `null` — never 0 — when no ground truth was supplied or it was empty. A
   * 0 here would read as "found none of them", the strongest possible failure
   * claim, from a run that had nothing to find.
   */
  recallAtK: number | null;
  /** Ground-truth ids found in the top `k`. */
  foundIds: string[];
  /** Ground-truth ids ABSENT from the top `k` — the actionable half of recall. */
  missingIds: string[];
}

/**
 * Fraction of `relevantIds` that appear in the first `k` of `retrievedIds`.
 *
 * The metric `metrics.ts` deliberately has no analogue of, because unlike the
 * others it cannot be computed from the ranking alone. Returns `null` for an
 * empty relevant set — see the `recallAtK` note above.
 */
export function recallAtK(
  retrievedIds: readonly string[],
  relevantIds: readonly string[],
  k = 10,
): number | null {
  const relevant = new Set(relevantIds);
  if (relevant.size === 0) return null;
  const top = new Set(retrievedIds.slice(0, k));
  let found = 0;
  for (const id of relevant) if (top.has(id)) found++;
  return found / relevant.size;
}

/** Score a ranked list. Pure; the caller owns the judgement. */
export function evaluateRelevance<T>(input: RelevanceEvalInput<T>): RelevanceEvalResult {
  const {
    ranked,
    grade,
    k = 10,
    groundTruth,
    precisionThreshold = 2,
    accessoryThreshold = 1,
  } = input;

  const grades = ranked.map(grade);
  const topIds = groundTruth ? ranked.slice(0, k).map(groundTruth.idOf) : [];
  const truthIds = groundTruth ? [...new Set(groundTruth.ids)] : [];
  const inTop = new Set(topIds);

  const coverage: RelevanceEvalCoverage =
    ranked.length === 0
      ? 'empty-ranking'
      : groundTruth && truthIds.length === 0
        ? 'no-ground-truth'
        : 'complete';

  return {
    coverage,
    k,
    grades,
    ndcgAtK: ndcgAtK(grades, k),
    precisionAtK: precisionAtK(grades, k, precisionThreshold),
    accessoryAtK: accessoryAtK(grades, k, accessoryThreshold),
    recallAtK: groundTruth ? recallAtK(topIds, truthIds, k) : null,
    foundIds: truthIds.filter((id) => inTop.has(id)),
    missingIds: truthIds.filter((id) => !inTop.has(id)),
  };
}

/** A relevance floor: every stated bound must hold, or the eval FAILS. */
export interface RelevanceFloor {
  minNdcgAtK?: number;
  minPrecisionAtK?: number;
  minRecallAtK?: number;
  /** Upper bound — pollution is the one metric where MORE is worse. */
  maxAccessoryAtK?: number;
}

export interface RelevanceAssertion {
  ok: boolean;
  /** Human-readable reasons the floor was not met; empty when `ok`. */
  failures: string[];
  result: RelevanceEvalResult;
}

/**
 * Compare an evaluation to a floor, REFUSING a run that measured nothing.
 *
 * A vacuous run is reported as a failure rather than compared to the floor, and
 * this is the whole point of the function: `coverage: 'empty-ranking'` scores
 * zero on every metric, so a floor check would "correctly" fail it with a
 * message about relevance — sending the reader to tune ranking when retrieval
 * returned nothing at all. `no-ground-truth` is the mirror image: it would
 * silently PASS any `minRecallAtK` (there is nothing to miss), which is the
 * dangerous direction — a regression test that cannot fail.
 */
export function assertRelevance(
  result: RelevanceEvalResult,
  floor: RelevanceFloor,
): RelevanceAssertion {
  const failures: string[] = [];

  if (result.coverage === 'empty-ranking') {
    failures.push(
      'VACUOUS: retrieval returned no results, so every score is 0 by vacuity — ' +
        'this is a retrieval failure, not a relevance one. Scores not compared to the floor.',
    );
  } else if (result.coverage === 'no-ground-truth' && floor.minRecallAtK !== undefined) {
    failures.push(
      'VACUOUS: minRecallAtK was requested but the ground-truth set is EMPTY, so ' +
        'recall has no referent and this check could never fail. Fix the ground-truth ' +
        'derivation before trusting this eval.',
    );
  } else {
    const { minNdcgAtK, minPrecisionAtK, minRecallAtK, maxAccessoryAtK } = floor;
    if (minNdcgAtK !== undefined && result.ndcgAtK < minNdcgAtK) {
      failures.push(`nDCG@${result.k} ${result.ndcgAtK.toFixed(4)} < floor ${minNdcgAtK}`);
    }
    if (minPrecisionAtK !== undefined && result.precisionAtK < minPrecisionAtK) {
      failures.push(`P@${result.k} ${result.precisionAtK.toFixed(4)} < floor ${minPrecisionAtK}`);
    }
    if (maxAccessoryAtK !== undefined && result.accessoryAtK > maxAccessoryAtK) {
      failures.push(`accessory@${result.k} ${result.accessoryAtK} > ceiling ${maxAccessoryAtK}`);
    }
    if (minRecallAtK !== undefined) {
      // Unreachable via the branch above, but a null here would otherwise
      // compare as false and read as a relevance failure.
      if (result.recallAtK === null) {
        failures.push('VACUOUS: minRecallAtK requested but no ground truth was supplied.');
      } else if (result.recallAtK < minRecallAtK) {
        failures.push(
          `recall@${result.k} ${result.recallAtK.toFixed(4)} < floor ${minRecallAtK} — ` +
            `missing ${result.missingIds.length}: ${result.missingIds.slice(0, 5).join(', ')}`,
        );
      }
    }
  }

  return { ok: failures.length === 0, failures, result };
}
