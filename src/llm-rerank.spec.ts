import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { llmRerank, defaultLlmRerankPrompt } from './llm-rerank';

type Row = { title: string };
const rows = (...titles: string[]): Row[] => titles.map((title) => ({ title }));
const getTitle = (r: Row) => r.title;
const jsonResp = (obj: unknown) =>
  ({ ok: true, json: async () => ({ choices: [{ message: { content: JSON.stringify(obj) } }] }) }) as Response;

describe('llmRerank', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('fewer than 2 rows → unchanged, no network', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const r = rows('only one');
    expect(await llmRerank('q', r, { apiKey: 'k', getTitle })).toBe(r);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('disabled → unchanged, no network', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const r = rows('a', 'b');
    expect(await llmRerank('q', r, { enabled: false, apiKey: 'k', getTitle })).toBe(r);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('no API key → unchanged', async () => {
    vi.stubEnv('OPENAI_API_KEY', '');
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const r = rows('a', 'b');
    expect(await llmRerank('q', r, { getTitle })).toBe(r);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('sorts MATCH-first, preserving incoming order within a bucket', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResp({
      wanted_type: 'laptop',
      items: [{ n: 1, match: false }, { n: 2, match: true }],
    })));
    const out = await llmRerank('dell laptop', rows('Dell Palmrest', 'Dell Laptop'), { apiKey: 'k', getTitle });
    expect(out.map(getTitle)).toEqual(['Dell Laptop', 'Dell Palmrest']);
  });

  it('an index left unset by the items list defaults to match=true (never demotes)', async () => {
    // Right length (4 items for 4 rows, so the shape guard passes) but n=2 is
    // duplicated and n=4 omitted → idx 3 (row 4) is never set → defaults to match.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResp({
      wanted_type: 'laptop',
      items: [{ n: 2, match: true }, { n: 1, match: false }, { n: 3, match: false }, { n: 2, match: true }],
    })));
    const out = await llmRerank('q', rows('p1', 'laptop', 'p3', 'unjudged'), { apiKey: 'k', getTitle });
    // match bucket (orig order): laptop(2), unjudged(4-default); non-match: p1(1), p3(3)
    expect(out.map(getTitle)).toEqual(['laptop', 'unjudged', 'p1', 'p3']);
  });

  it('shape mismatch (items length ≠ rows) → unchanged', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResp({ items: [{ n: 1, match: true }] })));
    const r = rows('a', 'b');
    expect(await llmRerank('q', r, { apiKey: 'k', getTitle })).toBe(r);
  });

  it('non-ok / error → unchanged (fail-safe)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false } as Response));
    const r = rows('a', 'b');
    expect(await llmRerank('q', r, { apiKey: 'k', getTitle })).toBe(r);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('boom')));
    const r2 = rows('a', 'b');
    expect(await llmRerank('q', r2, { apiKey: 'k', getTitle })).toBe(r2);
  });

  it('defaultLlmRerankPrompt interpolates query + titles and keeps the JSON contract', () => {
    const p = defaultLlmRerankPrompt('dell laptop', '1. A\n2. B');
    expect(p).toContain('for the query "dell laptop"');
    expect(p).toContain('1. A\n2. B');
    expect(p).toContain('"items":[{"n":1,"match":true|false}'); // parse contract preserved
  });

  it('a custom buildPrompt overrides the default in the API call', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(jsonResp({ items: [{ n: 1, match: true }, { n: 2, match: true }] }));
    vi.stubGlobal('fetch', fetchSpy);
    await llmRerank('q', rows('a', 'b'), {
      apiKey: 'k', getTitle, buildPrompt: (query, list) => `CUSTOM ${query} :: ${list}`,
    });
    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body.messages[0].content).toBe('CUSTOM q :: 1. a\n2. b');
  });
});
