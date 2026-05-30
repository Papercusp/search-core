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
