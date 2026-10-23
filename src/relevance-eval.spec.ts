import { describe, it, expect } from 'vitest';
import { ndcgAtK } from './metrics';
import {
  assertRelevance, assertSearchEvaluation, assertSearchFieldFreshness, evaluateRelevance, recallAtK,
  type SearchFieldFreshnessPolicy, type SearchFieldFreshnessInput,
  type SearchEvaluationDeclaration, type SearchEvaluationRun,
} from './relevance-eval';

/**
 * A ranked item: an id plus the grade a derived judgement would give it.
 *
 * `grade` and `idOf` are both annotated with the FULL `Doc` so the generic `T`
 * infers the same type at every call site — annotating each with only the field
 * it reads makes TS infer `T` from whichever is narrower and reject the other.
 */
interface Doc {
  id: string;
  g: number;
}
const doc = (id: string, g: number): Doc => ({ id, g });
const grade = (d: Doc) => d.g;
const idOf = (d: Doc) => d.id;

describe('recallAtK — the metric metrics.ts cannot express', () => {
  it('is the fraction of the RELEVANT set present in the top k', () => {
    expect(recallAtK(['a', 'b', 'c'], ['a', 'b', 'c', 'd'], 10)).toBe(0.75);
    expect(recallAtK(['a'], ['a'], 10)).toBe(1);
    expect(recallAtK(['x', 'y'], ['a', 'b'], 10)).toBe(0);
  });

  it('honours the k cutoff — a relevant doc BELOW k is missed, not found', () => {
    const retrieved = ['x', 'x', 'x', 'a'];
    expect(recallAtK(retrieved, ['a'], 10)).toBe(1);
    expect(recallAtK(retrieved, ['a'], 3)).toBe(0);
  });

  it('an EMPTY relevant set is null, not 0 — "nothing to find" is not "found nothing"', () => {
    // 0 is the strongest possible failure claim; reporting it for a query with
    // no ground truth would invent a regression out of missing test data.
    expect(recallAtK(['a', 'b'], [], 10)).toBeNull();
  });

  it('de-duplicates nothing on its own — the caller\'s set semantics are respected', () => {
    // A Set is built internally, so a repeated relevant id cannot inflate the
    // denominator and silently depress recall.
    expect(recallAtK(['a'], ['a', 'a', 'a'], 10)).toBe(1);
  });
});

/**
 * THE REASON THIS MODULE EXISTS. Every metric in `metrics.ts` scores only the
 * list handed to it, so a perfectly-ordered list that MISSED most of the
 * relevant documents is indistinguishable from a perfect result.
 *
 * This is not a hypothetical: the owner report that motivated the eval was
 * "the search returned 1 of the 4 sessions I remember" — a pure recall failure.
 */
describe('the shipped metrics are blind to a miss — the calibration for recallAtK', () => {
  it('nDCG is 1.0 for a correctly-ordered list that found only ONE of four relevant docs', () => {
    const ranked = [doc('s1', 3), doc('junk-1', 0), doc('junk-2', 0)];
    const groundTruth = { ids: ['s1', 's2', 's3', 's4'], idOf };

    // The metric contract alone: a flawless score.
    expect(ndcgAtK(ranked.map(grade), 10)).toBeCloseTo(1, 10);

    // The harness: the same ranking, and the failure is now visible.
    const out = evaluateRelevance({ ranked, grade, groundTruth });
    expect(out.ndcgAtK).toBeCloseTo(1, 10);
    expect(out.recallAtK).toBe(0.25);
    expect(out.missingIds).toEqual(['s2', 's3', 's4']);

    // And a recall floor catches what an nDCG floor waves through.
    expect(assertRelevance(out, { minNdcgAtK: 0.9 }).ok).toBe(true);
    expect(assertRelevance(out, { minRecallAtK: 0.9 }).ok).toBe(false);
  });
});

describe('evaluateRelevance', () => {
  it('reports ordering, pollution and recall together', () => {
    const ranked = [doc('a', 3), doc('junk', 0), doc('b', 2)];
    const out = evaluateRelevance({
      ranked,
      grade,
      k: 3,
      groundTruth: { ids: ['a', 'b', 'c'], idOf },
    });
    expect(out.coverage).toBe('complete');
    expect(out.grades).toEqual([3, 0, 2]);
    expect(out.precisionAtK).toBeCloseTo(2 / 3, 10); // grades >= 2
    expect(out.accessoryAtK).toBe(1); // grades <= 1
    expect(out.recallAtK).toBeCloseTo(2 / 3, 10);
    expect(out.foundIds).toEqual(['a', 'b']);
    expect(out.missingIds).toEqual(['c']);
  });

  it('recall is null — never 0 — when no ground truth was supplied', () => {
    const out = evaluateRelevance({ ranked: [doc('a', 3)], grade });
    expect(out.recallAtK).toBeNull();
    expect(out.coverage).toBe('complete'); // ordering WAS measured
  });

  it('grades the whole list but scores only the top k', () => {
    const ranked = [doc('a', 3), doc('b', 3), doc('c', 0)];
    const out = evaluateRelevance({ ranked, grade, k: 2 });
    expect(out.grades).toEqual([3, 3, 0]); // full list, for inspection
    expect(out.precisionAtK).toBe(1); // top-2 only
  });
});

describe('coverage — a zero is only evidence once you know the measurement ran', () => {
  it('an empty ranking is empty-ranking, not a relevance failure', () => {
    const out = evaluateRelevance({ ranked: [], grade, groundTruth: { ids: ['a'], idOf } });
    expect(out.coverage).toBe('empty-ranking');
    // Every score is 0 — identical to "found things, all irrelevant".
    expect(out.ndcgAtK).toBe(0);
    expect(out.recallAtK).toBe(0);
  });

  it('an empty ground-truth set is no-ground-truth, and recall is null', () => {
    const out = evaluateRelevance({ ranked: [doc('a', 3)], grade, groundTruth: { ids: [], idOf } });
    expect(out.coverage).toBe('no-ground-truth');
    expect(out.recallAtK).toBeNull();
  });

  it('empty-ranking WINS over no-ground-truth — nothing about that run is informative', () => {
    const out = evaluateRelevance({ ranked: [], grade, groundTruth: { ids: [], idOf } });
    expect(out.coverage).toBe('empty-ranking');
  });
});

describe('assertRelevance — refuses a vacuous run instead of scoring it', () => {
  it('passes when every stated bound holds', () => {
    const out = evaluateRelevance({
      ranked: [doc('a', 3), doc('b', 3)],
      grade,
      groundTruth: { ids: ['a', 'b'], idOf },
    });
    const got = assertRelevance(out, {
      minNdcgAtK: 0.99,
      minPrecisionAtK: 1,
      minRecallAtK: 1,
      maxAccessoryAtK: 0,
    });
    expect(got.ok).toBe(true);
    expect(got.failures).toEqual([]);
  });

  it('names EVERY bound that failed, not just the first', () => {
    const out = evaluateRelevance({
      ranked: [doc('junk', 0), doc('a', 3)],
      grade,
      groundTruth: { ids: ['a', 'b'], idOf },
    });
    const got = assertRelevance(out, { minPrecisionAtK: 1, minRecallAtK: 1, maxAccessoryAtK: 0 });
    expect(got.ok).toBe(false);
    expect(got.failures).toHaveLength(3);
    // The missing ids ride along — recall's actionable half.
    expect(got.failures.join(' ')).toContain('b');
  });

  it('an empty ranking FAILS as vacuous, and says so as a RETRIEVAL problem', () => {
    const out = evaluateRelevance({ ranked: [], grade, groundTruth: { ids: ['a'], idOf } });
    const got = assertRelevance(out, { minNdcgAtK: 0.5 });
    expect(got.ok).toBe(false);
    expect(got.failures).toHaveLength(1);
    expect(got.failures[0]).toContain('VACUOUS');
    // The point: it does NOT send the reader to tune ranking.
    expect(got.failures[0]).toContain('retrieval failure');
  });

  /**
   * The dangerous direction. An empty ground-truth set makes any recall floor
   * pass — a regression test that CANNOT FAIL, which looks exactly like a
   * regression test that keeps passing.
   */
  it('an empty ground-truth set FAILS a recall floor instead of silently passing it', () => {
    const out = evaluateRelevance({ ranked: [doc('a', 3)], grade, groundTruth: { ids: [], idOf } });
    expect(assertRelevance(out, { minRecallAtK: 1 }).ok).toBe(false);
    expect(assertRelevance(out, { minRecallAtK: 1 }).failures[0]).toContain('could never fail');
    // Ordering bounds are still honoured — only recall is unmeasurable here.
    expect(assertRelevance(out, { minNdcgAtK: 0.99 }).ok).toBe(true);
  });

  it('a recall floor with no ground truth AT ALL also fails vacuous', () => {
    const out = evaluateRelevance({ ranked: [doc('a', 3)], grade });
    const got = assertRelevance(out, { minRecallAtK: 0.5 });
    expect(got.ok).toBe(false);
    expect(got.failures[0]).toContain('VACUOUS');
  });
});
describe('field-bound freshness evidence', () => {
  function fixture(): { policy: SearchFieldFreshnessPolicy; input: SearchFieldFreshnessInput } {
    const version = { entityRef: 'record:one', fieldPath: 'status', valueFingerprint: 'hash:open',
      sourceRevision: 'rev:2', authority: 'canonical-store', authorityEpoch: 'scope:2' };
    return {
      policy: { fieldPath: 'status', authority: 'canonical-store', clock: 'canonical-read', maxAgeMs: 100,
        maxFutureSkewMs: 5, readAfterWrite: 'not-promised', onUnknown: 'refuse' },
      input: { returned: { entityRef: version.entityRef, fieldPath: version.fieldPath, valueFingerprint: version.valueFingerprint },
        indexed: { ...version, indexRevision: 'index:2', observedAtMs: 950 },
        canonical: { ...version, readAtMs: 990 }, requestStartedAtMs: 980, responseAtMs: 1000 },
    };
  }

  it('admits a field validated against the current authority during the request', () => {
    const { policy, input } = fixture();
    expect(assertSearchFieldFreshness(policy, input)).toEqual({ status: 'current', action: 'accept', reasons: [] });
  });

  it.each(['entityRef', 'fieldPath', 'valueFingerprint', 'sourceRevision', 'authority', 'authorityEpoch'] as const)(
    'a fresh envelope cannot hide a mismatched %s', key => {
      const { policy, input } = fixture(); input.indexed![key] = 'stale-or-foreign';
      input.indexed!.observedAtMs = 990;
      expect(assertSearchFieldFreshness(policy, input).status).toBe('stale');
    });

  it('rejects mixed-version field hydration even when the row revision matches', () => {
    const { policy, input } = fixture(); input.returned.valueFingerprint = 'hash:closed';
    expect(assertSearchFieldFreshness(policy, input).status).toBe('stale');
  });

  it('keeps an unavailable canonical read unknown, including on degraded output', () => {
    const { policy, input } = fixture(); input.canonical = null; policy.onUnknown = 'degrade';
    expect(assertSearchFieldFreshness(policy, input)).toMatchObject({ status: 'unknown', action: 'degrade' });
  });

  it('does not manufacture freshness from an unknown index revision or field clock', () => {
    for (const edit of [
      (i: SearchFieldFreshnessInput) => { i.indexed!.indexRevision = ''; },
      (i: SearchFieldFreshnessInput) => { i.indexed!.observedAtMs = Number.NaN; },
      (i: SearchFieldFreshnessInput) => { i.canonical!.readAtMs = Number.NaN; },
    ]) {
      const { policy, input } = fixture(); edit(input);
      expect(assertSearchFieldFreshness(policy, input).status).toBe('unknown');
    }
  });

  it('rejects a cached field replayed after revocation despite a fresh cache observation', () => {
    const { policy, input } = fixture(); input.canonical!.authorityEpoch = 'revoked:3';
    input.indexed!.observedAtMs = 995;
    expect(assertSearchFieldFreshness(policy, input).reasons).toContain('authority epoch changed');
  });

  it('requires response-time canonical validation for fields promised current', () => {
    const { policy, input } = fixture(); input.canonical!.readAtMs = 970;
    expect(assertSearchFieldFreshness(policy, input).reasons).toContain('canonical validation predates the request');
  });

  it('can use a declared index-age boundary without calling an old value current', () => {
    const { policy, input } = fixture(); policy.clock = 'index-observation';
    input.indexed!.observedAtMs = 899;
    expect(assertSearchFieldFreshness(policy, input).status).toBe('stale');
    input.indexed!.observedAtMs = 900;
    expect(assertSearchFieldFreshness(policy, input).status).toBe('current');
  });

  it('requires the committed revision when read-your-writes is promised', () => {
    const { policy, input } = fixture(); policy.readAfterWrite = 'required';
    expect(assertSearchFieldFreshness(policy, input).status).toBe('unknown');
    input.requiredWriteRevision = 'rev:3';
    expect(assertSearchFieldFreshness(policy, input).status).toBe('stale');
    input.requiredWriteRevision = 'rev:2';
    expect(assertSearchFieldFreshness(policy, input).status).toBe('current');
  });

  it('keeps rollback, out-of-bound future clocks and invalid policies unknown', () => {
    for (const edit of [
      (p: SearchFieldFreshnessPolicy, i: SearchFieldFreshnessInput) => { i.responseAtMs = 970; },
      (p: SearchFieldFreshnessPolicy, i: SearchFieldFreshnessInput) => { i.canonical!.readAtMs = 1006; },
      (p: SearchFieldFreshnessPolicy, i: SearchFieldFreshnessInput) => { i.indexed!.observedAtMs = 1006; },
      (p: SearchFieldFreshnessPolicy) => { p.maxAgeMs = Number.NaN; },
    ]) {
      const { policy, input } = fixture(); edit(policy, input);
      expect(assertSearchFieldFreshness(policy, input).status).not.toBe('current');
    }
  });
});

describe('frozen search evaluation populations', () => {
  function fixture(): { baseline: SearchEvaluationDeclaration; run: SearchEvaluationRun } {
    const baseline: SearchEvaluationDeclaration = {
      declaredAtMs: 100,
      pins: { sourceRevision: 'source-v1', corpus: 'corpus-v1', querySet: 'queries-v1', modelConfiguration: 'model-v1' },
      cells: ['cold', 'warm'].map(temperature => ({
        id: temperature, product: 'catalog', adapter: 'actual-catalog', queryClass: 'exact-id',
        temperature: temperature as 'cold' | 'warm', attemptIds: [`${temperature}-1`, `${temperature}-2`],
        minExecuted: 2, minJudged: 2, budgetSource: 'decision:pre-treatment-target',
        maxP95LatencyMs: 200, maxPerAttemptCostUsd: 0.01, maxErrorRate: 0, maxDegradedRate: 0,
        quality: { minRecallAtK: 1 },
      })),
    };
    const run: SearchEvaluationRun = {
      startedAtMs: 200, declaration: structuredClone(baseline), pins: { ...baseline.pins },
      attempts: baseline.cells.flatMap(cell => cell.attemptIds.map(id => ({
        id, cellId: cell.id, outcome: 'success' as const, executed: true, degraded: false,
        latencyMs: 50, costUsd: 0.001, resultCount: 1,
        quality: evaluateRelevance({ ranked: ['found'], grade: () => 3, groundTruth: { ids: ['found'], idOf: x => x } }),
      }))),
    };
    return { baseline, run };
  }

  it('accepts complete predeclared cells and retains separate denominators', () => {
    const { baseline, run } = fixture();
    const result = assertSearchEvaluation(baseline, run);
    expect(result.status).toBe('pass');
    expect(result.cells).toEqual(['cold', 'warm'].map(id => ({ id, attempted: 2, eligible: 2, executed: 2, judged: 2, censored: 0 })));
  });

  it.each(['sourceRevision', 'corpus', 'querySet', 'modelConfiguration'] as const)('rejects an absent %s pin', key => {
    const { baseline, run } = fixture();
    baseline.pins[key] = '';
    expect(assertSearchEvaluation(baseline, run).status).toBe('fail');
  });

  it('rejects changed actual inputs even when the declaration was copied faithfully', () => {
    const { baseline, run } = fixture();
    run.pins.corpus = 'different-corpus';
    expect(assertSearchEvaluation(baseline, run).failures).toContain('manifest: corpus changed');
  });

  it('rejects a post-run declaration and changes to budgets, populations, or minimum coverage', () => {
    for (const edit of [
      (r: SearchEvaluationRun) => { r.declaration.cells[0].maxP95LatencyMs = 999; },
      (r: SearchEvaluationRun) => { r.declaration.cells[0].attemptIds = ['cold-1']; },
      (r: SearchEvaluationRun) => { r.declaration.cells[0].minJudged = 1; },
      (r: SearchEvaluationRun) => { r.startedAtMs = 100; },
    ]) {
      const { baseline, run } = fixture(); edit(run);
      expect(assertSearchEvaluation(baseline, run).status).toBe('fail');
    }
  });

  it('keeps an omitted failure unknown instead of shrinking its denominator', () => {
    const { baseline, run } = fixture();
    run.attempts = run.attempts.filter(a => a.id !== 'cold-2');
    const result = assertSearchEvaluation(baseline, run);
    expect(result.status).toBe('unknown');
    expect(result.cells[0]).toMatchObject({ attempted: 2, executed: 1, judged: 1 });
  });

  it('fails on an observed timeout and cannot hide it behind the other cell', () => {
    const { baseline, run } = fixture();
    run.attempts[0].outcome = 'timeout';
    expect(assertSearchEvaluation(baseline, run).failures).toContain('cold: uncensored timeout cold-1');
  });

  it('allows predeclared censoring but requires its receipt and preserves its population', () => {
    const { baseline, run } = fixture();
    baseline.cells[0].censoring = { attemptIds: ['cold-2'], reason: 'predeclared calibration attempt' };
    baseline.cells[0].minExecuted = baseline.cells[0].minJudged = 1;
    run.declaration = structuredClone(baseline);
    run.attempts[1].outcome = 'timeout';
    expect(assertSearchEvaluation(baseline, run).cells[0]).toMatchObject({ attempted: 2, eligible: 1, censored: 1 });
    expect(assertSearchEvaluation(baseline, run).status).toBe('pass');
    run.attempts = run.attempts.filter(a => a.id !== 'cold-2');
    expect(assertSearchEvaluation(baseline, run).status).toBe('unknown');
  });

  it('rejects duplicate and cross-cell receipts', () => {
    const { baseline, run } = fixture();
    run.attempts = [...run.attempts, { ...run.attempts[0], cellId: 'warm' }];
    expect(assertSearchEvaluation(baseline, run).status).toBe('fail');
  });

  it('does not let cheap/fast warm results mask a cold budget miss', () => {
    const { baseline, run } = fixture();
    run.attempts[0].latencyMs = 201;
    run.attempts[0].costUsd = 0.02;
    run.attempts[0].degraded = true;
    expect(assertSearchEvaluation(baseline, run).failures).toEqual(expect.arrayContaining([
      'cold: p95 latency exceeded', 'cold: cost exceeded cold-1', 'cold: degraded rate exceeded',
    ]));
  });

  it('keeps missing costs, judgments and unknown outcomes unknown', () => {
    const { baseline, run } = fixture();
    run.attempts[0].costUsd = null;
    run.attempts[0].quality = undefined;
    run.attempts[1].outcome = 'unknown';
    expect(assertSearchEvaluation(baseline, run).status).toBe('unknown');
  });

  it('cannot pass nonfinite metric measurements through JavaScript comparisons', () => {
    const { baseline, run } = fixture();
    run.attempts[0].quality!.recallAtK = Number.NaN;
    expect(assertSearchEvaluation(baseline, run).status).toBe('unknown');
  });

  it('keeps out-of-scale quality measurements unknown', () => {
    const { baseline, run } = fixture();
    run.attempts[0].quality!.recallAtK = 2;
    expect(assertSearchEvaluation(baseline, run).status).toBe('unknown');
  });

  it('fails a measured quality miss using the existing relevance floor', () => {
    const { baseline, run } = fixture();
    run.attempts[0].quality = evaluateRelevance({ ranked: ['wrong'], grade: () => 3, groundTruth: { ids: ['found'], idOf: x => x } });
    expect(assertSearchEvaluation(baseline, run).status).toBe('fail');
  });

  it('rejects a duplicate population under a different label', () => {
    const { baseline, run } = fixture();
    baseline.cells[1].temperature = 'cold';
    run.declaration = structuredClone(baseline);
    expect(assertSearchEvaluation(baseline, run).failures).toContain('warm: duplicate population');
  });

  it('rejects unmeasured budget declarations and rate ceilings outside their units', () => {
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, -1, 2]) {
      const { baseline, run } = fixture();
      baseline.cells[0].maxErrorRate = value;
      run.declaration = structuredClone(baseline);
      expect(assertSearchEvaluation(baseline, run).status).toBe('fail');
    }
  });

  it('reports an undersized declared cell as unknown', () => {
    const { baseline, run } = fixture();
    baseline.cells[0].minJudged = 3;
    run.declaration = structuredClone(baseline);
    expect(assertSearchEvaluation(baseline, run).status).toBe('unknown');
  });

  it('judges a declared no-match case separately from vacuous ranking metrics', () => {
    const { baseline, run } = fixture();
    baseline.cells[0].quality = 'no-match';
    run.declaration = structuredClone(baseline);
    for (const a of run.attempts.filter(a => a.cellId === 'cold')) { a.resultCount = 0; a.quality = undefined; }
    expect(assertSearchEvaluation(baseline, run).status).toBe('pass');
    run.attempts[0].resultCount = 1;
    expect(assertSearchEvaluation(baseline, run).status).toBe('fail');
  });
});
