/**
 * The two-stage relevance orchestrator — the headline value of search-core.
 *
 * Same contract as @papercusp/rerank ({query, docs:[{id,text,row}], opts} →
 * ordered rows), just more stages:
 *   1. instruction-following cross-encoder rerank (zerank-2 via @papercusp/rerank)
 *   2. bucket scores (so genuine relevance leads) + a completeness tiebreak
 *   3. tiered-escalation gate (§5)
 *   4. live LLM category-match pass (§1e) over a kept window
 *   5. slice to `limit`
 *
 * Engine/project-agnostic: the caller supplies the doc text (see buildRerankDoc),
 * the per-row `qualityScore` tiebreak, the `getTitle` for the LLM pass, and the
 * gate flags. Fail-safe throughout — a rerank or LLM outage degrades to
 * retrieval order, never an error.
 */
import { rerank, type RerankEngine } from '@papercusp/rerank';
import { shouldEscalate, type EscalationOptions } from './escalation';
import { llmRerank, type LlmRerankOptions } from './llm-rerank';

export interface RankDoc<T> {
  /** Stable id (for caching / debugging). */
  id: string;
  /** The text the cross-encoder scores against the query (see buildRerankDoc). */
  text: string;
  /** Caller's payload, carried through untouched. */
  row: T;
}

export interface RankWithRerankerOptions<T> {
  /** Final page size. */
  limit: number;
  /** Instruction-following steering for the reranker; omit for a plain rerank. */
  instruction?: string;
  /** Reranker model. Default @papercusp/rerank's default (zerank-2). */
  rerankModel?: string;
  /** Reranker API key. Default ZEROENTROPY_API_KEY (handled by @papercusp/rerank). */
  rerankApiKey?: string;
  /**
   * Which cross-encoder scores stage 1, forwarded to @papercusp/rerank:
   * `local` (an ONNX cross-encoder — no key, no network, no per-call cost) or
   * `zeroentropy` (the hosted zerank API). Default: the lib's default
   * (`zeroentropy`), so an existing consumer's behavior is unchanged.
   */
  engine?: RerankEngine;
  /**
   * Scoring override for `engine: 'local'` — returns scores index-aligned with
   * `docs`, and MAY throw.
   *
   * This is the seam a REMOTE local-model host (e.g. a sidecar process serving
   * one warm model for the whole box) reaches the stages below through, without
   * search-core learning anything about transport. A throw is caught by
   * @papercusp/rerank's single fail-safe seam, so a scorer outage degrades to
   * retrieval order exactly like a missing runtime does.
   */
  scorer?: (query: string, texts: string[]) => Promise<number[]>;
  /**
   * Window kept after the rerank so the LLM pass can pull a buried product up
   * into the page and push accessories out. Default max(limit, 15).
   */
  window?: number;
  /**
   * Score bucketing granularity: scores are bucketed to `round(score * bucket)`
   * so near-equal relevance ties break on `qualityScore`. Default 20 (0.05 buckets).
   */
  scoreBucket?: number;
  /** Per-row completeness tiebreak (higher = better), applied only within a bucket. */
  qualityScore?: (row: T) => number;
  /** §5 tiered-escalation config. Omit → always escalate (LLM runs every query). */
  escalation?: EscalationOptions;
  /** Live LLM category-match pass (§1e). Omit to skip it entirely. */
  llm?: LlmRerankOptions<T>;
}

export async function rankWithReranker<T>(
  query: string,
  docs: Array<RankDoc<T>>,
  opts: RankWithRerankerOptions<T>,
): Promise<T[]> {
  const reranked = await rerank(query, docs, {
    instruction: opts.instruction,
    model: opts.rerankModel,
    apiKey: opts.rerankApiKey,
    engine: opts.engine,
    scorer: opts.scorer,
  });

  const W = opts.window ?? Math.max(opts.limit, 15);
  const bucket = opts.scoreBucket ?? 20;
  const quality = opts.qualityScore ?? (() => 0);

  let ordered: T[];
  if (reranked.some((r) => r.reranked)) {
    // Bucket rerank scores so genuine relevance leads, then break ties by quality.
    ordered = reranked
      .map((r) => ({ row: r.row, b: Math.round((r.score ?? 0) * bucket), qs: quality(r.row) }))
      .sort((a, x) => x.b - a.b || x.qs - a.qs)
      .slice(0, W)
      .map((r) => r.row);
  } else {
    // Rerank unavailable → keep retrieval order.
    ordered = reranked.slice(0, W).map((r) => r.row);
  }

  const escalate = shouldEscalate(
    reranked.filter((r) => r.reranked).map((r) => r.score),
    opts.escalation,
  );
  if (escalate && opts.llm) ordered = await llmRerank(query, ordered, opts.llm);

  return ordered.slice(0, opts.limit);
}
