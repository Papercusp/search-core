import { describe, it, expect } from 'vitest';
import { buildRerankDoc } from './doc';

describe('buildRerankDoc', () => {
  it('title mode = bare trimmed title', () => {
    expect(buildRerankDoc({ title: '  Dell Latitude  ' }, 'title')).toBe('Dell Latitude');
  });

  it('title_category appends normalized (underscores→spaces, lowercased) product type', () => {
    expect(buildRerankDoc({ title: 'Dell Latitude', productType: 'NOTEBOOK_COMPUTER' }, 'title_category'))
      .toBe('Dell Latitude — notebook computer');
  });

  it('title_category with empty type keeps the trailing em-dash trimmed (faithful to source)', () => {
    expect(buildRerankDoc({ title: 'Dell' }, 'title_category')).toBe('Dell —');
  });

  it('title_desc (default) = title + ". " + 240-char description slice', () => {
    expect(buildRerankDoc({ title: 'Dell', description: 'A 14-inch business laptop' }))
      .toBe('Dell. A 14-inch business laptop');
  });

  it('title_desc truncates the description to 240 chars', () => {
    const long = 'x'.repeat(500);
    const out = buildRerankDoc({ title: 'T', description: long });
    expect(out).toBe(`T. ${'x'.repeat(240)}`);
  });

  it('title_desc with no/blank/non-string description falls back to bare title', () => {
    expect(buildRerankDoc({ title: '  T  ', description: '   ' })).toBe('T');
    expect(buildRerankDoc({ title: '  T  ', description: null })).toBe('T');
    expect(buildRerankDoc({ title: '  T  ' })).toBe('T');
  });

  it('null title coalesces to empty string', () => {
    expect(buildRerankDoc({ title: null }, 'title')).toBe('');
  });
});
