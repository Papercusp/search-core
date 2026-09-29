import { describe, it, expect } from 'vitest';
import {
  cosine,
  decideChunking,
  drawShortProbe,
  drawTailProbe,
  isDistinctiveProbe,
  pooledScore,
  rankKeys,
  runChunkingBench,
  summarizeRankings,
  type BenchEmbed,
  type BenchFold,
  type BenchRow,
  type BenchSplitter,
  type ChunkingBenchResult,
  type ChunkingDecisionRule,
} from './retrieval-bench';

// ─── fixtures ──────────────────────────────────────────────────────────────
// A bag-of-words embedder: each distinct token gets its own dimension (an
// index, not a hash, so no two tokens ever collide). Rows use disjoint
// vocabularies, so a probe matches exactly the text it was cut from and the
// bench's arithmetic can be checked by construction.

const DIMS = 1 << 16;
const CUT = 2000;
const vocab = new Map<string, number>();

function letters(n: number): string {
  let s = '';
  do {
    s = String.fromCharCode(97 + (n % 26)) + s;
    n = Math.floor(n / 26);
  } while (n > 0);
  return s;
}

/** `count` distinct letters-only words under `prefix`. */
function words(prefix: string, count: number): string {
  return Array.from({ length: count }, (_, j) => `${prefix}${letters(j)}`).join(' ');
}

function dim(token: string): number {
  let d = vocab.get(token);
  if (d === undefined) {
    d = vocab.size;
    if (d >= DIMS) throw new Error('fixture vocabulary exceeded DIMS');
    vocab.set(token, d);
  }
  return d;
}

function bow(text: string): number[] {
  const v = new Array<number>(DIMS).fill(0);
  for (const t of text.split(/\s+/).filter(Boolean)) v[dim(t)] += 1;
  return v;
}

const bowEmbed: BenchEmbed = async (_kind, texts) => texts.map(bow);

const windows: BenchSplitter = (row, max) => {
  const out: string[] = [];
  for (let s = 0; s < row.text.length; s += 1250) {
    out.push(row.text.slice(s, s + 1500));
    if (out.length >= max || s + 1500 >= row.text.length) break;
  }
  return out;
};

function row(key: string, text: string): BenchRow {
  return { key, text, parentDoc: text.slice(0, CUT) };
}

/** Head vocabulary the parent sees, tail vocabulary only chunks can reach. */
function longRow(i: number): BenchRow {
  return row(`L${i}`, `${words(`hd${letters(i)}x`, 300)} ${words(`tl${letters(i)}x`, 420)}`);
}

function shortRow(i: number): BenchRow {
  return row(`S${i}`, words(`sh${letters(i)}x`, 110));
}

function fold(longs: BenchRow[], shorts: BenchRow[]): BenchFold {
  const probes = [
    ...longs.map((r) => drawTailProbe(r, { cut: CUT, length: 240 })),
    ...shorts.map((r) => drawShortProbe(r, 240)),
  ].filter((p): p is NonNullable<typeof p> => p !== null);
  return { rows: [...longs, ...shorts], probes };
}

const RULE: ChunkingDecisionRule = {
  minTailProbes: 5,
  minShortProbes: 5,
  minTailMrrGain: 0.05,
  maxShortMrrDrop: 0,
  mrrTolerance: 0.01,
};

// ─── probes ────────────────────────────────────────────────────────────────

describe('isDistinctiveProbe', () => {
  it('accepts prose and rejects punctuation, blobs and short fragments', () => {
    expect(isDistinctiveProbe(words('abc', 20))).toBe(true);
    expect(isDistinctiveProbe('|---|---|'.repeat(20))).toBe(false);
    expect(isDistinctiveProbe('a'.repeat(240))).toBe(false); // one token
    expect(isDistinctiveProbe('')).toBe(false);
  });
});

describe('drawTailProbe', () => {
  const r = longRow(3);

  it('draws past the cut, from text the parent document does not contain', () => {
    const p = drawTailProbe(r, { cut: CUT, length: 240 });
    expect(p).not.toBeNull();
    expect(p!.start).toBeGreaterThanOrEqual(CUT);
    expect(p!.text).toHaveLength(240);
    expect(r.parentDoc.includes(p!.text)).toBe(false);
    expect(r.text.slice(p!.start, p!.start + 240)).toBe(p!.text);
  });

  it('is deterministic per key, and different keys land at different offsets', () => {
    expect(drawTailProbe(r, { cut: CUT, length: 240 })).toEqual(drawTailProbe(r, { cut: CUT, length: 240 }));
    const starts = new Set(Array.from({ length: 10 }, (_, i) => drawTailProbe(longRow(i), { cut: CUT, length: 240 })!.start));
    expect(starts.size).toBeGreaterThan(1);
  });

  it('honours a fixed offset, and refuses one inside the parent window', () => {
    expect(drawTailProbe(r, { cut: CUT, length: 240, at: 3000 })!.start).toBe(3000);
    expect(drawTailProbe(r, { cut: CUT, length: 240, at: 100 })).toBeNull();
  });

  it('returns null when the row has no tail long enough for a probe', () => {
    expect(drawTailProbe(shortRow(0), { cut: CUT, length: 240 })).toBeNull();
  });

  it('refuses a window the parent document also contains', () => {
    const text = words('rep', 300);
    const dup: BenchRow = { key: 'D', text: `${text} ${text}`, parentDoc: `${text} ${text}` };
    expect(drawTailProbe(dup, { cut: CUT, length: 240 })).toBeNull();
  });
});

describe('drawShortProbe', () => {
  it('takes the centred window of a row the parent sees whole', () => {
    const r = shortRow(1);
    const p = drawShortProbe(r, 240)!;
    expect(p.probeClass).toBe('short');
    expect(p.start).toBe(Math.floor((r.text.length - 240) / 2));
    expect(r.parentDoc.includes(p.text)).toBe(true);
  });

  it('returns null when the parent document does not contain the window', () => {
    const r: BenchRow = { key: 'X', text: words('zz', 110), parentDoc: 'unrelated' };
    expect(drawShortProbe(r, 240)).toBeNull();
  });
});

// ─── ranking ───────────────────────────────────────────────────────────────

describe('rankKeys', () => {
  it('orders best-first and places a tied row ABOVE the target', () => {
    expect(rankKeys(['a', 'b', 'c'], [0.5, 0.9, 0.5], 'a')).toEqual(['b', 'c', 'a']);
    expect(rankKeys(['a', 'b', 'c'], [0.5, 0.9, 0.5], 'c')).toEqual(['b', 'a', 'c']);
  });
});

describe('summarizeRankings', () => {
  it('computes MRR, recall@1, recall@5 and mean rank from the target positions', () => {
    const ranked = [
      ['t', 'x'],
      ['x', 't'],
      ['a', 'b', 'c', 'd', 'e', 't'],
    ];
    const s = summarizeRankings(ranked, ['t', 't', 't']);
    expect(s.ranks).toEqual([1, 2, 6]);
    expect(s.mrr).toBeCloseTo((1 + 1 / 2 + 1 / 6) / 3, 10);
    expect(s.recallAt1).toBeCloseTo(1 / 3, 10);
    expect(s.recallAt5).toBeCloseTo(2 / 3, 10);
    expect(s.meanRank).toBeCloseTo(3, 10);
  });

  it('reports null metrics, never zeros, for an empty class', () => {
    expect(summarizeRankings([], [])).toMatchObject({ n: 0, mrr: null, recallAt1: null, recallAt5: null, meanRank: null });
  });
});

describe('pooledScore', () => {
  it('takes the best of the parent and the first k chunks only', () => {
    const q = [1, 0];
    const parent = [0, 1];
    const chunks = [
      [1, 1],
      [1, 0],
    ];
    expect(pooledScore(q, parent, chunks, 0)).toBeCloseTo(cosine(q, parent), 10);
    expect(pooledScore(q, parent, chunks, 1)).toBeCloseTo(cosine(q, [1, 1]), 10);
    expect(pooledScore(q, parent, chunks, 2)).toBeCloseTo(1, 10);
  });

  it('lowers chunk similarities by the margin but never the parent’s', () => {
    expect(pooledScore([1, 0], [0, 1], [[1, 0]], 1, 0.3)).toBeCloseTo(0.7, 10);
    expect(pooledScore([1, 0], [1, 0], [[1, 0]], 1, 0.3)).toBeCloseTo(1, 10);
  });
});

// ─── the bench ─────────────────────────────────────────────────────────────

async function syntheticResult(): Promise<ChunkingBenchResult> {
  const longs = Array.from({ length: 6 }, (_, i) => longRow(i));
  const shorts = Array.from({ length: 6 }, (_, i) => shortRow(i));
  return runChunkingBench({
    folds: [fold(longs.slice(0, 3), shorts.slice(0, 3)), fold(longs.slice(3), shorts.slice(3))],
    splitters: { window: windows },
    maxChunks: [1, 4],
    minChunkTextChars: CUT,
    embed: bowEmbed,
  });
}

describe('runChunkingBench', () => {
  it('finds past-the-cut probes through chunks that the parent vector cannot see', async () => {
    const res = await syntheticResult();
    expect(res.folds).toBe(2);
    expect(res.poolSizes).toEqual([6, 6]);
    const parent = res.arms.find((a) => a.label === 'parent')!;
    const full = res.arms.find((a) => a.label === 'window@4')!;
    expect(parent.tail.n).toBe(6);
    // The tail vocabulary is absent from every parent vector: all scores tie at
    // zero, and the pessimistic tie rule ranks the true parent last.
    expect(parent.tail.recallAt1).toBe(0);
    expect(full.tail.recallAt1).toBe(1);
    expect(full.tail.containedRate).toBe(1);
    expect(full.tail.improved).toBe(6);
    expect(parent.tail.containedRate).toBeNull();
  });

  it('scores smaller caps as prefixes, so a cap short of the probe cannot reach it', async () => {
    const res = await syntheticResult();
    const one = res.arms.find((a) => a.label === 'window@1')!;
    // Chunk 0 is [0, 1500): inside the parent window, so it adds nothing past the cut.
    expect(one.tail.containedRate).toBe(0);
    expect(one.tail.recallAt1).toBe(0);
    expect(res.splitters.window.chunkedRows).toBe(6);
    expect(res.splitters.window.rowsAtCap[1]).toBe(6);
    expect(res.splitters.window.rowsAtCap[4]).toBe(6);
  });

  it('never chunks a row at or under the parent window', async () => {
    const res = await syntheticResult();
    const full = res.arms.find((a) => a.label === 'window@4')!;
    const parent = res.arms.find((a) => a.label === 'parent')!;
    expect(full.short.n).toBe(6);
    expect(full.short.mrr).toBe(parent.short.mrr);
    expect(full.short.unchanged).toBe(6);
  });

  it('counts a short row demoted by another row’s chunk', async () => {
    const s = shortRow(0);
    const probe = drawShortProbe(s, 240)!;
    // A long row whose tail is the probe's own tokens, shuffled (so it is not a
    // verbatim copy) and repeated, outscores the short row once pooled.
    const tokens = probe.text.split(/\s+/).filter(Boolean).reverse().join(' ');
    const l: BenchRow = row('L', `${words('headq', 300)} ${Array(12).fill(tokens).join(' ')}`);
    const res = await runChunkingBench({
      folds: [{ rows: [s, l, shortRow(1)], probes: [probe] }],
      splitters: { window: windows },
      maxChunks: [4],
      minChunkTextChars: CUT,
      embed: bowEmbed,
    });
    const parent = res.arms.find((a) => a.label === 'parent')!;
    const pooled = res.arms.find((a) => a.label === 'window@4')!;
    expect(parent.short.recallAt1).toBe(1);
    expect(pooled.short.worsened).toBe(1);
    expect(pooled.short.mrr!).toBeLessThan(parent.short.mrr!);
  });

  it('scores every cap at every margin, and a large enough margin undoes the demotion', async () => {
    const s = shortRow(0);
    const probe = drawShortProbe(s, 240)!;
    const tokens = probe.text.split(/\s+/).filter(Boolean).reverse().join(' ');
    const l: BenchRow = row('L', `${words('headq', 300)} ${Array(12).fill(tokens).join(' ')}`);
    const res = await runChunkingBench({
      folds: [{ rows: [s, l, shortRow(1)], probes: [probe] }],
      splitters: { window: windows },
      maxChunks: [2, 4],
      chunkMargins: [0, 1],
      minChunkTextChars: CUT,
      embed: bowEmbed,
    });
    expect(res.arms.map((a) => a.label)).toEqual(['parent', 'window@2', 'window@2-m1', 'window@4', 'window@4-m1']);
    const margined = res.arms.find((a) => a.label === 'window@4-m1')!;
    expect(margined.chunkMargin).toBe(1);
    // Cosine never exceeds 1, so a margin of 1 leaves every chunk below its parent.
    expect(margined.short.worsened).toBe(0);
  });

  it('never improves a short probe’s rank by pooling (pooling only raises OTHER rows)', async () => {
    for (let seed = 0; seed < 5; seed++) {
      const longs = Array.from({ length: 4 }, (_, i) => longRow(seed * 10 + i));
      const shorts = Array.from({ length: 4 }, (_, i) => shortRow(seed * 10 + i));
      const res = await runChunkingBench({
        folds: [fold(longs, shorts)],
        splitters: { window: windows },
        maxChunks: [2, 4],
        minChunkTextChars: CUT,
        embed: bowEmbed,
      });
      for (const arm of res.arms) expect(arm.short.improved).toBe(0);
    }
  });

  it('drops a probe whose text also occurs in another row of its fold', async () => {
    const a = longRow(1);
    const p = drawTailProbe(a, { cut: CUT, length: 240 })!;
    const copy = row('copy', `${words('cpy', 50)} ${p.text}`);
    const res = await runChunkingBench({
      folds: [{ rows: [a, copy], probes: [p] }],
      splitters: { window: windows },
      maxChunks: [4],
      minChunkTextChars: CUT,
      embed: bowEmbed,
    });
    expect(res.ambiguousProbesDropped).toBe(1);
    expect(res.arms[0].tail.n).toBe(0);
  });

  it('refuses an empty cap sweep', async () => {
    await expect(
      runChunkingBench({ folds: [], splitters: {}, maxChunks: [0], minChunkTextChars: CUT, embed: bowEmbed }),
    ).rejects.toThrow(/at least one positive cap/);
  });
});

// ─── the decision ──────────────────────────────────────────────────────────

describe('decideChunking', () => {
  it('registers the cheapest arm within tolerance of the best tail MRR', async () => {
    const d = decideChunking(await syntheticResult(), RULE);
    expect(d.verdict).toBe('register');
    expect(d.arm!.label).toBe('window@4');
  });

  it('withholds a verdict on too small a sample', async () => {
    const d = decideChunking(await syntheticResult(), { ...RULE, minTailProbes: 50 });
    expect(d.verdict).toBe('insufficient-sample');
    expect(d.arm).toBeNull();
    expect(d.reasons.join()).toMatch(/only 6 tail probes/);
  });

  it('skips when no arm gains enough', async () => {
    const d = decideChunking(await syntheticResult(), { ...RULE, minTailMrrGain: 2 });
    expect(d.verdict).toBe('skip');
  });

  it('excludes an arm that demotes short rows past the budget', () => {
    const cls = (mrr: number) => ({
      n: 10, mrr, recallAt1: mrr, recallAt5: mrr, meanRank: 1, improved: 0, worsened: 0, unchanged: 10, containedRate: null,
    });
    const result: ChunkingBenchResult = {
      folds: 1,
      poolSizes: [20],
      ambiguousProbesDropped: 0,
      splitters: {},
      arms: [
        { label: 'parent', splitter: null, maxChunks: null, chunkMargin: null, tail: cls(0.2), short: cls(0.9) },
        { label: 'window@8', splitter: 'window', maxChunks: 8, chunkMargin: 0, tail: cls(0.8), short: cls(0.7) },
        { label: 'window@4', splitter: 'window', maxChunks: 4, chunkMargin: 0, tail: cls(0.6), short: cls(0.9) },
      ],
    };
    const d = decideChunking(result, RULE);
    expect(d.verdict).toBe('register');
    expect(d.arm!.label).toBe('window@4');
    expect(d.reasons.join()).toMatch(/window@8: short-row MRR/);
  });
});
