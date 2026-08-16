import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { rewriteQuery, clearRewriteCache, defaultRewritePrompt } from './rewrite';

const okResp = (content: string) =>
  ({ ok: true, json: async () => ({ choices: [{ message: { content } }] }) }) as Response;

describe('rewriteQuery', () => {
  beforeEach(() => {
    clearRewriteCache();
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('disabled → returns the query untouched, no network', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    expect(await rewriteQuery('del laptop', { enabled: false, apiKey: 'k' })).toBe('del laptop');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('skips single-token SKU/model numbers (contain a digit)', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    expect(await rewriteQuery('MAX3221CAE', { apiKey: 'k' })).toBe('MAX3221CAE');
    expect(await rewriteQuery('379429-001', { apiKey: 'k' })).toBe('379429-001');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('skips empty and over-long (>60) queries', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    expect(await rewriteQuery('   ', { apiKey: 'k' })).toBe('   ');
    const long = 'a '.repeat(40);
    expect(await rewriteQuery(long, { apiKey: 'k' })).toBe(long);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('never rewrites an exact known-brand query (isKnownBrand callback)', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const out = await rewriteQuery('ametric', { apiKey: 'k', isKnownBrand: async (k) => k === 'ametric' });
    expect(out).toBe('ametric');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('no API key → returns raw, no network', async () => {
    vi.stubEnv('OPENAI_API_KEY', '');
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    expect(await rewriteQuery('del laptop')).toBe('del laptop');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('corrects a real-token typo, fires onRewrite, and caches (one network call)', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(okResp('dell laptop'));
    vi.stubGlobal('fetch', fetchSpy);
    const onRewrite = vi.fn();
    expect(await rewriteQuery('del laptop', { apiKey: 'k', onRewrite })).toBe('dell laptop');
    expect(onRewrite).toHaveBeenCalledWith('del laptop', 'dell laptop');
    // second call served from cache — no extra network
    expect(await rewriteQuery('del laptop', { apiKey: 'k' })).toBe('dell laptop');
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it('strips wrapping quotes from the model output', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResp('"dell laptop"')));
    expect(await rewriteQuery('del laptop', { apiKey: 'k' })).toBe('dell laptop');
  });

  it('non-ok response → raw', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false } as Response));
    expect(await rewriteQuery('del laptop', { apiKey: 'k' })).toBe('del laptop');
  });

  it('odd (over-long) model output → falls back to the original query', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResp('x'.repeat(80))));
    expect(await rewriteQuery('del laptop', { apiKey: 'k' })).toBe('del laptop');
  });

  it('network error → raw (fail-safe)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('boom')));
    expect(await rewriteQuery('del laptop', { apiKey: 'k' })).toBe('del laptop');
  });

  it('defaultRewritePrompt includes the query and the ONLY-corrected-text instruction', () => {
    const p = defaultRewritePrompt('del laptop');
    expect(p).toContain('Query: del laptop');
    expect(p).toContain('Return ONLY the corrected query text');
  });

  it('a custom buildPrompt overrides the default in the API call', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(okResp('fixed'));
    vi.stubGlobal('fetch', fetchSpy);
    await rewriteQuery('typo', { apiKey: 'k', buildPrompt: (q) => `MYPROMPT ${q}` });
    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body.messages[0].content).toBe('MYPROMPT typo');
  });
});
