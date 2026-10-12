import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineVitestConfig } from '@papercusp/test-config/vitest-config';
import { mergeConfig } from 'vitest/config';

export default mergeConfig(
  defineVitestConfig({
    layer: 'unit',
    include: ['src/**/*.spec.ts'],
  }),
  {
    // import.meta.url, not the CommonJS `__dirname`: the committed-source loader makes Vite import
    // this config natively as ES, where `__dirname` is undefined (WI-6554700010009145).
    root: dirname(fileURLToPath(import.meta.url)),
    test: { globals: true },
  },
);
