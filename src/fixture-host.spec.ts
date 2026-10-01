/**
 * A host that is not papercusp runs the retrieval benchmark
 * (shared-vector-search-libraries-2026-09-29 acceptance R-12, the benchmark
 * half of "coverage gate and benchmark"; the gate half is in @papercusp/search's
 * fixture-host test).
 *
 * The host is a herbarium catalogue. It brings its own model list, two hashed
 * bag-of-words embedders of different vector widths and input windows, and its
 * own corpus of specimen notes. Nothing below names a papercusp model, width,
 * table or source. Everything is configured through the library's public entry.
 *
 * The corpus is built so the expected numbers follow from the inputs: every
 * note opens with the same 1,200 characters of boilerplate, and only the text
 * after it says which specimen it is. A vector that sees only the boilerplate
 * cannot tell the notes apart, and the benchmark's tie rule ranks the true note
 * last; a vector that sees past it finds the right note.
 */
import { describe, expect, it } from 'vitest';
import {
  drawTailProbe,
  runChunkingBench,
  runWidthSweep,
  type BenchEmbed,
  type BenchRow,
  type BenchSplitter,
} from './index';

interface HostModel {
  id: string;
  /** Vector width the model emits. */
  dims: number;
  /** Characters the model reads before it truncates its input. */
  windowChars: number;
}

/** The catalogue's model list. */
const MODELS: readonly HostModel[] = [
  { id: 'herbarium-bow-short', dims: 512, windowChars: 900 },
  { id: 'herbarium-bow-long', dims: 1000, windowChars: 4000 },
];
const SHORT = MODELS[0]!;
const LONG = MODELS[1]!;

/**
 * FNV-1a followed by murmur3's finalizer. Plain FNV-1a spreads words that differ
 * only in their last letters poorly modulo a non-power-of-two width, which put
 * unrelated notes within 0.01 cosine of the true one; the finalizer mixes every
 * input bit into the low bits.
 */
function wordHash(word: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < word.length; i++) h = Math.imul(h ^ word.charCodeAt(i), 0x01000193);
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/**
 * The model's embedder, recording the width of every vector it emits. It marks
 * word PRESENCE, not frequency: with counts, the repeated boilerplate dominates
 * every vector's norm and hash collisions with it decide the ranking.
 */
function embedderFor(model: HostModel, widths: number[]): BenchEmbed {
  return async (_kind, texts) =>
    texts.map((text) => {
      const v = new Array<number>(model.dims).fill(0);
      for (const word of text.slice(0, model.windowChars).split(/\s+/).filter(Boolean)) v[wordHash(word) % model.dims] = 1;
      widths.push(v.length);
      return v;
    });
}

const HEAD_CHARS = 1200;
const FILLER = 'the specimen was pressed dried and mounted on archival paper before the collector label was written ';
const HEAD = `${FILLER.repeat(Math.ceil(HEAD_CHARS / FILLER.length)).slice(0, HEAD_CHARS - 1)} `;
const SPECIES = ['quercus', 'betula', 'acer', 'fagus', 'tilia', 'ulmus', 'fraxinus', 'sorbus', 'prunus', 'malus', 'pyrus', 'salix'];

/** a, b, …, z, aa, ab, …: suffixes that keep every word letters-only. */
function suffix(n: number): string {
  let s = '';
  let x = n;
  do {
    s = String.fromCharCode(97 + (x % 26)) + s;
    x = Math.floor(x / 26) - 1;
  } while (x >= 0);
  return s;
}

/** Boilerplate, then 100 words only this specimen's note contains. */
const NOTES = SPECIES.map((species) => ({
  key: `sheet-${species}`,
  text: HEAD + Array.from({ length: 100 }, (_, j) => `${species}${suffix(j)}`).join(' '),
}));

describe('fixture host: width sweep over the catalogue model list', () => {
  const widths = [1000, 1600, 2400];

  it('the corpus is shaped as described: shared boilerplate, then specimen text', () => {
    expect(HEAD).toHaveLength(HEAD_CHARS);
    expect(new Set(NOTES.map((n) => n.text.slice(0, HEAD_CHARS))).size).toBe(1);
    expect(NOTES.every((n) => n.text.length > 1600)).toBe(true);
  });

  it('a model that reads past the boilerplate finds every note once the width reaches the probe', async () => {
    const emitted: number[] = [];
    const res = await runWidthSweep({ docs: NOTES, widths, probeStart: HEAD_CHARS, sample: 12, embed: embedderFor(LONG, emitted) });
    expect(res.docs).toHaveLength(12);
    expect(res.arms.map((a) => a.width)).toEqual(widths);
    // Width 1000 embeds boilerplate only: every note ties, and a tie ranks the true note last.
    expect(res.arms.map((a) => a.recallAt1)).toEqual([0, 1, 1]);
    expect(res.arms[0]!.meanRank).toBe(12);
    expect(res.arms.map((a) => a.identicalToFirst)).toEqual([12, 0, 0]);
    expect(new Set(emitted)).toEqual(new Set([LONG.dims]));
  });

  it('a model whose window ends inside the boilerplate never finds a note, at any width', async () => {
    const emitted: number[] = [];
    const res = await runWidthSweep({ docs: NOTES, widths, probeStart: HEAD_CHARS, sample: 12, embed: embedderFor(SHORT, emitted) });
    expect(res.arms.map((a) => a.recallAt1)).toEqual([0, 0, 0]);
    // Every width is truncated to the same 900 characters, so every arm equals the first.
    expect(res.arms.map((a) => a.identicalToFirst)).toEqual([12, 12, 12]);
    expect(new Set(emitted)).toEqual(new Set([SHORT.dims]));
  });
});

describe('fixture host: chunking benchmark for the short-window model', () => {
  /** The catalogue's chunker: 600-character windows every 500 characters. */
  const windows: BenchSplitter = (row, max) => {
    const out: string[] = [];
    for (let s = 0; s < row.text.length && out.length < max; s += 500) out.push(row.text.slice(s, s + 600));
    return out;
  };
  const rows: BenchRow[] = NOTES.map((n) => ({ key: n.key, text: n.text, parentDoc: n.text.slice(0, SHORT.windowChars) }));
  // Probes at 1300-1540 lie wholly inside the third chunk (1000-1600).
  const probes = rows.map((r) => drawTailProbe(r, { cut: SHORT.windowChars, length: 240, at: 1300 }));

  it('chunks reach specimen text the parent vector cannot see, but only once the cap includes it', async () => {
    expect(probes.every((p) => p !== null)).toBe(true);
    const emitted: number[] = [];
    const res = await runChunkingBench({
      folds: [{ rows, probes: probes.filter((p): p is NonNullable<typeof p> => p !== null) }],
      splitters: { sheet: windows },
      maxChunks: [2, 4],
      minChunkTextChars: SHORT.windowChars,
      embed: embedderFor(SHORT, emitted),
    });
    const arm = (cap: number | null) => res.arms.find((a) => a.maxChunks === cap && (cap === null || a.splitter === 'sheet'))!;
    expect(res.poolSizes).toEqual([12]);
    expect(arm(null).tail.n).toBe(12);
    // The parent vector and the first two chunks see boilerplate only.
    expect(arm(null).tail.recallAt1).toBe(0);
    expect(arm(2).tail.recallAt1).toBe(0);
    expect(arm(2).tail.containedRate).toBe(0);
    // The third chunk holds the whole probe.
    expect(arm(4).tail.containedRate).toBe(1);
    expect(arm(4).tail.recallAt1).toBeGreaterThan(0.5);
    expect(arm(4).tail.improved).toBeGreaterThan(6);
    expect(new Set(emitted)).toEqual(new Set([SHORT.dims]));
  });
});
