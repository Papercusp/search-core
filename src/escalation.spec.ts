import { describe, it, expect } from 'vitest';
import { shouldEscalate } from './escalation';

describe('shouldEscalate', () => {
  it('tiering off (default) → always escalate', () => {
    expect(shouldEscalate([0.9, 0.2])).toBe(true);
    expect(shouldEscalate([])).toBe(true);
  });

  it('tiered + clear winner/dropoff → do NOT escalate', () => {
    // top 0.90, kth (top-5; only 2 scores → index 1) = 0.20; spread 0.70 ≥ 0.15
    expect(shouldEscalate([0.9, 0.2], { tiered: true })).toBe(false);
  });

  it('tiered + tight top cluster → escalate', () => {
    // top 0.92, 5th 0.87 → spread 0.05 < 0.15
    expect(shouldEscalate([0.92, 0.91, 0.9, 0.88, 0.87], { tiered: true })).toBe(true);
  });

  it('tiered but <2 numeric scores → be safe, escalate', () => {
    expect(shouldEscalate([0.9], { tiered: true })).toBe(true);
    // non-numbers are filtered out before the count check
    expect(shouldEscalate([0.9, undefined as unknown as number], { tiered: true })).toBe(true);
  });

  it('respects a custom gap', () => {
    // spread 0.10: escalate when gap 0.15, not when gap 0.05
    expect(shouldEscalate([0.9, 0.8], { tiered: true, gap: 0.15 })).toBe(true);
    expect(shouldEscalate([0.9, 0.8], { tiered: true, gap: 0.05 })).toBe(false);
  });

  it('uses the kth (default top-5) score for the spread, not the minimum', () => {
    // top 0.99, k=4 → sorted[4] = 0.90 → spread 0.09 < 0.15 → escalate.
    // The 0.10 tail is ignored; using the minimum (spread 0.89) would NOT escalate.
    expect(shouldEscalate([0.99, 0.97, 0.95, 0.92, 0.9, 0.1], { tiered: true })).toBe(true);
  });
});
