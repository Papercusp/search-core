/**
 * The library stands alone (shared-vector-search-libraries-2026-09-29
 * acceptance R-13).
 *
 * @papercusp/search-core is published on its own, so every import in src/ must
 * be one of: a relative module, a Node builtin, or a package declared in this
 * library's own package.json. Anything else (a host package such as
 * operator-core, a path into packages/ or apps/, an undeclared dependency that
 * only resolves because the monorepo hoists it) would break the library for
 * anyone who installs it outside papercusp.
 *
 * The classifier mirrors @papercusp/search's src/import-boundary.test.ts. It is
 * a copy rather than an import because importing it would itself be a relative
 * path out of this library, which is the thing this test forbids.
 *
 * The positive control plants forbidden imports in a synthetic source and
 * requires the classifier to flag them, and a calibration requires the scan to
 * see every import kind the library actually uses, so the real scan cannot pass
 * by finding nothing.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { builtinModules } from 'node:module';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const LIB_ROOT = fileURLToPath(new URL('..', import.meta.url));
const SRC = join(LIB_ROOT, 'src');

interface Manifest {
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
}

const manifest = JSON.parse(readFileSync(join(LIB_ROOT, 'package.json'), 'utf8')) as Manifest;
const DECLARED = new Set(
  [manifest.dependencies, manifest.devDependencies, manifest.peerDependencies, manifest.optionalDependencies].flatMap((d) =>
    Object.keys(d ?? {}),
  ),
);
const BUILTINS = new Set(builtinModules);

/**
 * Module specifiers in one TypeScript source: static imports and re-exports
 * (`from '…'`), bare side-effect imports, dynamic import() and require().
 * Comments are removed first so prose that mentions an import is not counted.
 */
function importSpecifiers(source: string): string[] {
  const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`])\/\/.*$/gm, '$1');
  const found: string[] = [];
  const patterns = [
    /\bfrom\s*(['"])([^'"\n]+)\1/g,
    /(?:^|[;\n])\s*import\s*(['"])([^'"\n]+)\1/g,
    /\bimport\s*\(\s*(['"])([^'"\n]+)\1\s*\)/g,
    /\brequire\s*\(\s*(['"])([^'"\n]+)\1\s*\)/g,
  ];
  for (const re of patterns) for (const m of code.matchAll(re)) found.push(m[2]!);
  return found;
}

/** The package a bare specifier names: '@scope/name' or 'name'. */
function packageOf(specifier: string): string {
  const parts = specifier.split('/');
  return specifier.startsWith('@') ? parts.slice(0, 2).join('/') : parts[0]!;
}

/** Why a specifier is outside the allowed set, or null when it is allowed. */
function importViolation(specifier: string, declared: ReadonlySet<string> = DECLARED): string | null {
  if (specifier.startsWith('./') || specifier.startsWith('../')) return null;
  if (specifier.startsWith('node:') || BUILTINS.has(specifier) || BUILTINS.has(packageOf(specifier))) return null;
  if (specifier.startsWith('/')) return 'absolute path';
  const pkg = packageOf(specifier);
  return declared.has(pkg) ? null : `package '${pkg}' is not declared in the library's package.json`;
}

/** True when a relative specifier resolves outside the library root. */
function escapesLibrary(fromFile: string, specifier: string): boolean {
  if (!specifier.startsWith('./') && !specifier.startsWith('../')) return false;
  const target = join(fromFile, '..', specifier);
  const rel = relative(LIB_ROOT, target);
  return rel.startsWith('..') || rel === '';
}

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = join(dir, e.name);
    if (e.isDirectory()) return e.name === 'node_modules' ? [] : sourceFiles(p);
    return /\.(?:[cm]?ts|tsx)$/.test(e.name) ? [p] : [];
  });
}

/** Every violation in the library's real src/ tree, as `file: 'spec' (why)`. */
function scanViolations(): string[] {
  const violations: string[] = [];
  for (const file of sourceFiles(SRC)) {
    for (const spec of importSpecifiers(readFileSync(file, 'utf8'))) {
      const why = importViolation(spec) ?? (escapesLibrary(file, spec) ? 'relative path leaves the library' : null);
      if (why) violations.push(`${relative(LIB_ROOT, file)}: '${spec}' (${why})`);
    }
  }
  return violations;
}

/**
 * The keyword, assembled at runtime. This file's own source is scanned too, so a
 * control written as a plain `from '<pkg>'` literal would match ITSELF and turn
 * the real scan red; built from parts, the planted text exists only at runtime.
 */
const FROM = ['fr', 'om'].join('');

describe('import boundary: @papercusp/search-core stands alone (R-13)', () => {
  it('positive control: planted host, sibling-library and undeclared imports are flagged', () => {
    const host = ['@papercusp', 'operator-core'].join('/');
    const planted = [
      `import { x } ${FROM} '${host}/lib/search';`,
      `export * ${FROM} '${['@papercusp', 'search'].join('/')}';`,
      `const m = await import('${['..', '..', '..', '..', 'packages', 'operator-core', 'lib'].join('/')}');`,
      `const t = require('${['type', 'sense'].join('')}');`,
    ].join('\n');
    const specs = importSpecifiers(planted);
    expect(specs).toHaveLength(4);
    expect(importViolation(specs[0]!)).toMatch(/not declared/);
    // A sibling generic library is still a package this one does not declare.
    expect(importViolation(specs[1]!)).toMatch(/not declared/);
    // A relative path is allowed by the rule itself; leaving the library is what
    // the relative-escape check catches.
    expect(importViolation(specs[2]!)).toBeNull();
    expect(escapesLibrary(join(SRC, 'rank.ts'), specs[2]!)).toBe(true);
    expect(importViolation(specs[3]!)).toMatch(/not declared/);
  });

  it('calibration: the scan sees every import kind the library uses', () => {
    const all = sourceFiles(SRC).flatMap((f) => importSpecifiers(readFileSync(f, 'utf8')));
    expect(all.length).toBeGreaterThan(20);
    expect(all.some((s) => s.startsWith('./') && !escapesLibrary(join(SRC, 'index.ts'), s))).toBe(true);
    expect(all).toContain('@papercusp/rerank');
    expect(all).toContain('vitest');
    // This file's own builtin imports are seen and allowed.
    expect(all).toContain('node:fs');
    expect(importViolation('node:fs')).toBeNull();
    // Declared packages are allowed; the declared set is read from package.json.
    expect(DECLARED.has('@papercusp/rerank')).toBe(true);
    expect(importViolation('@papercusp/rerank')).toBeNull();
    // Comments are not imports.
    expect(importSpecifiers(`// see import x ${FROM} 'nowhere'\n/* ${FROM} 'nor-here' */`)).toEqual([]);
    // …and the planted form really is detector-shaped outside a comment.
    expect(importSpecifiers(`import x ${FROM} 'somewhere'`)).toEqual(['somewhere']);
  });

  it('every import in src/ is relative, a Node builtin, or a declared dependency', () => {
    expect(scanViolations()).toEqual([]);
  });
});
