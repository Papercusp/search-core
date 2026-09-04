import { describe, it, expect } from 'vitest';
import { ndcgAtK } from './metrics';
import { assertRelevance, evaluateRelevance, recallAtK } from './relevance-eval';

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
