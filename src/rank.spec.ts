import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock the cross-encoder primitive so we can drive the orchestrator deterministically.
vi.mock('@papercusp/rerank', () => ({ rerank: vi.fn() }));

import { rerank } from '@papercusp/rerank';
import { rankWithReranker } from './rank';

const rerankMock = vi.mocked(rerank);
type Row = { name: string; q?: number };
const rr = (row: Row, score: number, reranked = true) => ({ row, score, reranked });
const jsonResp = (obj: unknown) =>
  ({ ok: true, json: async () => ({ choices: [{ message: { content: JSON.stringify(obj) } }] }) }) as Response;

describe('rankWithReranker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });
  afterEach(() => vi.unstubAllGlobals());

  it('buckets near-equal scores and breaks ties by qualityScore, then slices to limit', async () => {
    const A = { name: 'A', q: 0 }, B = { name: 'B', q: 1 }, C = { name: 'C', q: 5 };
    // A,B share a 0.05 bucket (0.90/0.90 → 18); C lower (0.50 → 10).
    rerankMock.mockResolvedValue([rr(A, 0.9), rr(B, 0.9), rr(C, 0.5)]);
    const out = await rankWithReranker('q', [A, B, C].map((row) => ({ id: row.name, text: row.name, row })), {
      limit: 2,
      qualityScore: (row) => row.q ?? 0,
    });
    expect(out.map((r) => r.name)).toEqual(['B', 'A']); // same bucket → higher quality first; C sliced off
  });

  it('rerank unavailable (passthrough) → retrieval order, sliced', async () => {
    const rows = [{ name: 'A' }, { name: 'B' }, { name: 'C' }];
    rerankMock.mockResolvedValue(rows.map((row) => rr(row, 0, false)));
    const out = await rankWithReranker('q', rows.map((row) => ({ id: row.name, text: row.name, row })), { limit: 2 });
    expect(out.map((r) => r.name)).toEqual(['A', 'B']);
  });

  /**
   * WI-37670: stage-1 engagement must be OBSERVABLE. A call that scored nothing
   * returns retrieval order, which carries no provenance in the returned rows —
   * so without this callback a caller measuring the reranker's effect reads a
   * confident 0.000 delta from a stage that never ran.
   */
  describe('onRerankStage', () => {
    const rows = [{ name: 'A' }, { name: 'B' }, { name: 'C' }];
    const docs = rows.map((row) => ({ id: row.name, text: row.name, row }));

    it('reports scored: 0 and forwards the degrade reason when the stage passed through', async () => {
      rerankMock.mockImplementation(async (_q, d, opts) => {
        (opts as { onDegrade?: (r: string) => void } | undefined)?.onDegrade?.('timeout');
        return (d as typeof docs).map(({ row }) => rr(row, 0, false));
      });
      const seen: Array<{ scored: number; total: number; degradeReason?: string }> = [];
      await rankWithReranker('q', docs, { limit: 2, onRerankStage: (i) => void seen.push(i) });
      expect(seen).toEqual([{ scored: 0, total: 3, degradeReason: 'timeout' }]);
    });

    /** The control: a healthy rerank must NOT look like a degrade. */
    it('reports the scored count with NO degradeReason when the stage ran', async () => {
      rerankMock.mockResolvedValue(rows.map((row, i) => rr(row, 0.9 - i * 0.1)));
      const seen: Array<{ scored: number; total: number; degradeReason?: string }> = [];
      await rankWithReranker('q', docs, { limit: 2, onRerankStage: (i) => void seen.push(i) });
      expect(seen).toEqual([{ scored: 3, total: 3 }]);
    });

    /**
     * A PARTIAL degrade — some pairs scored, some dropped by the lib's per-pair
     * fail-safe. `scored < total` is the only signal of it, and reporting it
     * before the window/limit slicing is what keeps it a fact about the RERANKER
     * rather than about the page size.
     */
    it('counts the reranker output, not the sliced page', async () => {
      rerankMock.mockResolvedValue([rr(rows[0], 0.9), rr(rows[1], 0.8, false), rr(rows[2], 0.7)]);
      const seen: Array<{ scored: number; total: number }> = [];
      await rankWithReranker('q', docs, { limit: 1, onRerankStage: (i) => void seen.push(i) });
      expect(seen).toEqual([{ scored: 2, total: 3 }]);
    });
  });

  /**
   * WIDENING `limit` MUST NOT PERTURB THE PREFIX.
   *
   * A caller that reorders the whole candidate pool without cutting it (the
   * documented `limit: rows.length` case) has to be able to treat its top-k as
   * THE SAME PAGE a `limit: k` call would have produced — P-038's `--pool-all`
   * bench arm compares against runs that judged only the page, and if the page
   * moved, the two runs are measuring different rankings and every number is
   * uncomparable.
   *
   * The property is not free: `limit` feeds `W = window ?? max(limit, 15)` as
   * well as the final slice, so it is only safe because both cut an
   * already-fully-sorted list. This pins it against a future change that, say,
   * windows BEFORE sorting or makes the bucket size depend on the limit.
   *
   * ⚠ Scoped to the no-`llm` path deliberately: the tiered escalation reorders
   * (rather than cuts) whatever list it is handed, so with an LLM pass
   * configured a longer slice legitimately CAN produce a different prefix.
   */
  it('a WIDER limit keeps the same prefix — reordering the whole pool leaves the page unchanged', async () => {
    const rows = Array.from({ length: 24 }, (_, i) => ({ name: `R${i}`, q: i }));
    // Deliberately adversarial, and it has to be: the BEST candidates sit at the
    // TAIL of retrieval order (the buried-hit case a reranker exists for), so
    // the page is drawn from beyond `W` at limit=10. Score them the other way
    // round — best at the head — and a window-BEFORE-sort implementation would
    // produce the very same prefix and this test would pass while asserting
    // nothing. Scores also collide into shared buckets so the qualityScore
    // tiebreak and V8's sort stability both do real work.
    const scored = rows.map((row, i) => rr(row, (i + 1) / 24 + (i % 3) * 0.001));
    const docs = rows.map((row) => ({ id: row.name, text: row.name, row }));

    rerankMock.mockResolvedValue(scored);
    const page = await rankWithReranker('q', docs, { limit: 10, qualityScore: (r) => r.q ?? 0 });
    rerankMock.mockResolvedValue(scored);
    const pool = await rankWithReranker('q', docs, { limit: rows.length, qualityScore: (r) => r.q ?? 0 });

    expect(pool).toHaveLength(24); // the pool arm really did keep every candidate
    expect(page).toHaveLength(10);
    expect(pool.slice(0, 10).map((r) => r.name)).toEqual(page.map((r) => r.name));
  });

  it('tiered escalation with a clear winner → LLM pass is skipped', async () => {
    const rows = [{ name: 'A' }, { name: 'B' }];
    rerankMock.mockResolvedValue([rr(rows[0], 0.95), rr(rows[1], 0.2)]);
    const fetchSpy = vi.fn(); // would throw the test if called
    vi.stubGlobal('fetch', fetchSpy);
    const out = await rankWithReranker('q', rows.map((row) => ({ id: row.name, text: row.name, row })), {
      limit: 5,
      escalation: { tiered: true }, // 0.95 vs 0.20 spread → no escalation
      llm: { apiKey: 'k', getTitle: (r) => r.name },
    });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(out.map((r) => r.name)).toEqual(['A', 'B']);
  });

  it('escalates (tiering off) and applies the LLM category-match reorder', async () => {
    const palm = { name: 'Dell Palmrest' }, laptop = { name: 'Dell Laptop' };
    rerankMock.mockResolvedValue([rr(palm, 0.91), rr(laptop, 0.9)]); // tight cluster, palmrest on top
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResp({
      wanted_type: 'laptop', items: [{ n: 1, match: false }, { n: 2, match: true }],
    })));
    const out = await rankWithReranker('dell laptop',
      [palm, laptop].map((row) => ({ id: row.name, text: row.name, row })), {
        limit: 5,
        llm: { apiKey: 'k', getTitle: (r) => r.name }, // escalation omitted → always escalate
      });
    expect(out.map((r) => r.name)).toEqual(['Dell Laptop', 'Dell Palmrest']);
  });
});
