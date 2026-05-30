import { describe, it, expect } from 'vitest';
import { dcg, ndcg, ndcgAtK, precisionAtK, accessoryAtK } from './metrics';

describe('dcg / ndcg', () => {
  it('dcg = Σ (2^g − 1)/log2(i+2)', () => {
    expect(dcg([3])).toBeCloseTo(7, 10); // (8-1)/log2(2) = 7/1
    expect(dcg([3, 2])).toBeCloseTo(7 + 3 / Math.log2(3), 10);
    expect(dcg([0, 0, 0])).toBe(0);
  });

  it('ndcg of an already-ideal list = 1', () => {
    expect(ndcg([3, 2, 1])).toBeCloseTo(1, 10);
  });

  it('ndcg of a worst-ordered list < 1', () => {
    expect(ndcg([1, 2, 3])).toBeLessThan(1);
  });

  it('ndcg with all-zero grades = 0 (idcg 0 guard)', () => {
    expect(ndcg([0, 0])).toBe(0);
  });

  it('ndcgAtK truncates to the top k before scoring', () => {
    // first two are ideal → 1.0 regardless of the tail
    expect(ndcgAtK([3, 2, 0, 1], 2)).toBeCloseTo(1, 10);
  });
});

describe('precisionAtK / accessoryAtK', () => {
  it('P@5 = fraction of top-5 with grade ≥ 2', () => {
    expect(precisionAtK([3, 2, 1, 0, 0], 5, 2)).toBeCloseTo(0.4, 10); // 2 of 5
    expect(precisionAtK([3, 3, 2, 2, 2], 5, 2)).toBeCloseTo(1, 10);
  });

  it('P@k over an empty list = 0', () => {
    expect(precisionAtK([], 5)).toBe(0);
  });

  it('acc@5 = count of top-5 with grade ≤ 1 (parts/irrelevant pollution)', () => {
    expect(accessoryAtK([3, 2, 1, 0, 0], 5, 1)).toBe(3); // 1,0,0
    expect(accessoryAtK([3, 3, 2, 2, 2], 5, 1)).toBe(0);
  });

  it('only the top-k window counts', () => {
    expect(accessoryAtK([3, 3, 3, 3, 3, 0, 0, 0], 5, 1)).toBe(0); // tail beyond k ignored
  });
});
