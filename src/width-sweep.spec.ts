/**
 * AUTO-BAR-R-6-P-003 (shared-vector-search-libraries-2026-09-29): the library
 * width sweep reports, for every width, the MRR and recall@1 that papercusp's
 * pre-move width sweep reported on the same deterministic corpus and embedder.
 *
 * The expected numbers are NOT computed by this code. They are the printed
 * output of packages/operator-core/lib/memory/bench/turn-truncation-width-cli.ts
 * as it stood before the move (git blob 2a5aa0a23b06ff48b97010832f9ea6a1b6046865,
 * last changed in commit b58e0d33dc58369421c56c5071660d51060d9b90), run by
 * .papercusp/scratch/rag-carry/p003-width-golden.mts on 2026-09-30 against the
 * fixture corpus in a throwaway database and the fixture embedder behind a fake
 * sidecar). The CLI prints MRR to 4 decimals and recall@1 as a percentage to 1
 * decimal, so MRR is compared at 4 decimals and recall@1 as an exact fraction
 * of the 12-document sample.
 */
import { describe, expect, it } from 'vitest';

import { runWidthSweep } from './width-sweep';
import { fixtureCorpus, fixtureEmbed } from './width-sweep.fixture';

interface Golden {
  width: number;
  mrr: number;
  hitsAt1: number;
  identicalToFirst: number;
}

const SAMPLE = 12;

function expectMatches(arms: Awaited<ReturnType<typeof runWidthSweep>>['arms'], golden: Golden[]): void {
  const widthArms = arms.filter((a) => a.width !== null);
  expect(widthArms.map((a) => a.width)).toEqual(golden.map((g) => g.width));
  for (const [i, g] of golden.entries()) {
    const arm = widthArms[i];
    expect(arm.n, `width ${g.width}: n`).toBe(SAMPLE);
    expect(arm.mrr, `width ${g.width}: MRR`).toBeCloseTo(g.mrr, 4);
    expect(arm.recallAt1, `width ${g.width}: recall@1`).toBeCloseTo(g.hitsAt1 / SAMPLE, 10);
    expect(arm.identicalToFirst, `width ${g.width}: identical to first width`).toBe(g.identicalToFirst);
  }
}

describe('runWidthSweep parity with the pre-move papercusp width sweep (R-6)', () => {
  it('reports the pre-move MRR and recall@1 for every width (probe at 3000, default widths)', async () => {
    const result = await runWidthSweep({
      docs: fixtureCorpus(),
      widths: [2000, 3000, 4000, 6000, 8000],
      probeStart: 3000,
      sample: SAMPLE,
      embed: fixtureEmbed,
    });
    expect(result.docs).toHaveLength(SAMPLE);
    expect(result.probes.every((p) => p.length === 240)).toBe(true);
    // Pre-move CLI, `--sample 12`:
    //   2000 0.6412 50.0% 12/12 · 3000 0.8000 75.0% 0/12 · 4000 0.9583 91.7% 0/12
    //   6000 1.0000 100.0% 0/12 · 8000 1.0000 100.0% 0/12
    expectMatches(result.arms, [
      { width: 2000, mrr: 0.6412, hitsAt1: 6, identicalToFirst: 12 },
      { width: 3000, mrr: 0.8, hitsAt1: 9, identicalToFirst: 0 },
      { width: 4000, mrr: 0.9583, hitsAt1: 11, identicalToFirst: 0 },
      { width: 6000, mrr: 1, hitsAt1: 12, identicalToFirst: 0 },
      { width: 8000, mrr: 1, hitsAt1: 12, identicalToFirst: 0 },
    ]);
  });

  it('reports the pre-move numbers under the probe-offset control (probe at 3300, widths 2000,3500,5000)', async () => {
    const result = await runWidthSweep({
      docs: fixtureCorpus(),
      widths: [2000, 3500, 5000],
      probeStart: 3300,
      sample: SAMPLE,
      embed: fixtureEmbed,
    });
    // Pre-move CLI, `--sample 12 --probe-at 3300 --widths 2000,3500,5000`:
    //   2000 0.4903 25.0% 12/12 · 3500 0.9306 91.7% 0/12 · 5000 0.8854 83.3% 0/12
    expectMatches(result.arms, [
      { width: 2000, mrr: 0.4903, hitsAt1: 3, identicalToFirst: 12 },
      { width: 3500, mrr: 0.9306, hitsAt1: 11, identicalToFirst: 0 },
      { width: 5000, mrr: 0.8854, hitsAt1: 10, identicalToFirst: 0 },
    ]);
  });

  it('scores a chunked arm by each document’s best chunk', async () => {
    const split = (text: string): string[] => {
      const out: string[] = [];
      for (let i = 0; i < text.length && out.length < 4; i += 1500) out.push(text.slice(i, i + 1500));
      return out;
    };
    const result = await runWidthSweep({
      docs: fixtureCorpus(),
      widths: [2000],
      probeStart: 3000,
      sample: SAMPLE,
      embed: fixtureEmbed,
      chunkArm: { label: 'chunk1500', split },
    });
    const chunk = result.arms.at(-1)!;
    expect(chunk.label).toBe('chunk1500');
    expect(chunk.width).toBeNull();
    expect(chunk.meanChunks).toBe(4);
    // The chunk holding characters 3000-4500 contains the probe, so the chunked arm beats the 2000 cut.
    expect(chunk.mrr!).toBeGreaterThan(result.arms[0].mrr!);
  });

  it('refuses to report on fewer usable documents than minDocs', async () => {
    await expect(
      runWidthSweep({ docs: fixtureCorpus(5), widths: [2000], probeStart: 3000, sample: SAMPLE, embed: fixtureEmbed }),
    ).rejects.toThrow('only 5 usable documents');
  });
});
