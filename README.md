# @papercusp/search-core

Engine-agnostic **search-relevance core**, shared across Papercusp and Restart.
Builds on [`@papercusp/rerank`](../rerank) (the cross-encoder primitive) and adds
the valuable two-stage relevance logic that used to live inside Restart's
`catalog.service.ts`:

- **`DEFAULT_RERANK_INSTRUCTION`** — the §1d instruction-following template
  (head-noun item-type rule: rank the product above its parts/accessories,
  defer to part-queries, prefer brand/model match).
- **`buildRerankDoc(fields, mode)`** — the rerank "document" text
  (`title` / `title_category` / `title_desc`).
- **`rewriteQuery(raw, opts)`** — brand-aware spell/typo + intent rewrite
  ("del laptop" → "dell laptop"), cached + fail-safe.
- **`llmRerank(query, rows, opts)`** — §1e live LLM category-match pass
  (infers the wanted item type and demotes accessories), fail-safe.
- **`shouldEscalate(scores, opts)`** — §5 tiered-escalation gate.
- **`rankWithReranker(query, docs, opts)`** — the two-stage orchestrator tying
  it all together (instruction rerank → bucket + quality tiebreak → escalate →
  LLM pass → slice).
- **Metrics** — `dcg`, `ndcg`, `ndcgAtK`, `precisionAtK`, `accessoryAtK`: the
  shared eval-harness contract so both repos measure relevance identically.

## Design

**No project dependencies.** Typesense, Postgres, the catalog schema, and the
brand vocabulary are NOT imported here — they're injected as config/callbacks:

| Project-specific input | Injected as |
|---|---|
| brand vocabulary (skip rewriting exact brand queries) | `rewriteQuery({ isKnownBrand })` |
| per-row title for the LLM pass | `llmRerank({ getTitle })` |
| completeness tiebreak | `rankWithReranker({ qualityScore })` |
| env gates (`QUERY_REWRITE`, `LLM_RERANK`, …) | `enabled` / `escalation.tiered` flags |
| API keys | `apiKey` opts (fall back to `OPENAI_API_KEY` / `ZEROENTROPY_API_KEY`) |

The contract mirrors `@papercusp/rerank`: `{ query, docs: [{ id, text, row }], opts }`
→ ordered rows — just with more stages. **Fail-safe throughout**: a rerank or
LLM outage degrades to retrieval order, never an error.

## Consuming it

`@papercusp/*` libs here are **src-as-entry** workspaces (`main: src/index.ts`),
resolved via a direct `node_modules/@papercusp/<name>` symlink + the
`tsconfig.base.json` path map — no build step, no stale-dist hazard. Runtimes
that transpile TS (tsx — shop-api, scout-service, the eval scripts) load the
source directly.

```ts
import {
  rankWithReranker, buildRerankDoc, DEFAULT_RERANK_INSTRUCTION,
} from '@papercusp/search-core';

const docs = items.map((it) => ({
  id: it.groupId ?? it.id,
  text: buildRerankDoc({ title: it.title, description: it.desc, productType: it.type },
                       process.env.RERANK_DOC_MODE as any),
  row: it,
}));

const ordered = await rankWithReranker(query, docs, {
  limit,
  instruction: process.env.RERANK_INSTRUCTION === '1' ? DEFAULT_RERANK_INSTRUCTION : undefined,
  qualityScore: (row) => completenessScore(row),
  escalation: { tiered: process.env.LLM_RERANK_TIER === '1' },
  llm: { enabled: process.env.LLM_RERANK === '1', getTitle: (row) => row.title ?? '' },
});
```

## Status / portability

Restart is the **first consumer** (`apps/shop-api/.../catalog.service.ts`).
Papercusp's search migrates onto this next. The lib is `private` — promotion to a
`github.com/Papercusp/search-core` git submodule (the established `@papercusp/*`
mechanism) is packaging hygiene deferred to an explicit go-ahead; nothing about
the API changes when it moves.
