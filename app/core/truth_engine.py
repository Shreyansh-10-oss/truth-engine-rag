"""
core/truth_engine.py — Multi-Source RAG Orchestrator (Tier 2)

Tier 2 upgrades over Tier 1:
  ✓ Hybrid Search: BM25 + FAISS fused via Reciprocal Rank Fusion (RRF)
  ✓ Cross-Encoder Re-Ranking: ms-marco-MiniLM scores chunks for true relevance
  ✓ Confidence Scoring: High / Medium / Low per source based on CE scores
  ✓ Conflict Detection: Detects A vs C contradictions, shows BOTH versions

Pipeline (Tier 2):
  1. Embed query once              [EmbeddingEngine — shared]
  2. Hybrid retrieve per source    [HybridRetriever: FAISS + BM25 → RRF]
  3. Re-rank per source            [CrossEncoderReranker]
  4. Compute confidence per source [CrossEncoderReranker.compute_source_confidence]
  5. Detect A vs C conflict        [ConflictDetector]
  6. Build prompt + call LLM       [LLMRouter]
  7. Parse + return structured response
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from app.core.conflict_detector import ConflictReport, detect_conflicts
from app.core.embeddings import EmbeddingEngine
from app.core.hybrid_retriever import HybridResult, HybridRetriever
from app.core.llm_router import LLMRouter, RouterResult
from app.core.reranker import CrossEncoderReranker, RankedChunk, SourceConfidence
from app.db.bm25_store import BM25Store
from app.db.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

SOURCE_PRIORITY = {"A": 1, "B": 2, "C": 3}
SOURCE_DESCRIPTIONS = {
    "A": "Golden Source of Truth (highest priority)",
    "B": "Real-world Fixes & Support Logs (experimental)",
    "C": "Legacy Wiki (may be deprecated or incorrect)",
}


@dataclass
class SourceResult:
    source_label: str
    chunks: list[RankedChunk]
    summary: str = ""
    loaded: bool = True
    confidence: SourceConfidence | None = None


@dataclass
class TruthEngineResponse:
    query: str
    source_a: SourceResult | None
    source_b: SourceResult | None
    source_c: SourceResult | None
    conclusion: str
    provider_used: str
    fallback_triggered: bool
    conflict_detected: bool = False
    conflict_details: str = ""
    conflict_report: ConflictReport | None = None


class TruthEngine:
    """
    Multi-source RAG pipeline with Tier 2 features:
    Hybrid Search + Re-Ranking + Confidence Scoring + Conflict Detection.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        faiss_stores: dict[str, FAISSStore],
        bm25_stores: dict[str, BM25Store],
        llm_router: LLMRouter,
        reranker: CrossEncoderReranker,
        top_k: int = 3,
        hybrid_candidates: int = 10,
    ) -> None:
        self._embedder   = embedding_engine
        self._faiss      = faiss_stores
        self._bm25       = bm25_stores
        self._router     = llm_router
        self._reranker   = reranker
        self._top_k      = top_k
        self._candidates = hybrid_candidates

    def query(
        self,
        user_question: str,
        preferred_provider: str = "gemini",
    ) -> TruthEngineResponse:
        logger.info("TruthEngine (Tier 2) query: '%s'", user_question[:80])

        # ── Step 1: Embed query once ──────────────────────────────────────────
        query_vector = self._embedder.embed([user_question])[0]

        # ── Step 2: Hybrid retrieve per source ────────────────────────────────
        hybrid_results: dict[str, list[HybridResult]] = {}
        for label in ["A", "B", "C"]:
            faiss = self._faiss.get(label)
            bm25  = self._bm25.get(label)
            if faiss and faiss.is_ready:
                # Use dummy BM25 if not available
                if not bm25:
                    bm25 = BM25Store(source_label=label)
                retriever = HybridRetriever(
                    faiss_store=faiss,
                    bm25_store=bm25,
                    vector_k=self._candidates,
                    bm25_k=self._candidates,
                )
                results = retriever.retrieve(
                    query_vector=query_vector,
                    query_text=user_question,
                    top_k=self._top_k * 2,  # extra candidates for reranker
                )
                hybrid_results[label] = results
                logger.debug("[Source %s] Hybrid: %d candidates.", label, len(results))
            else:
                hybrid_results[label] = []

        # Check at least one source returned results
        if not any(hybrid_results.values()):
            from app.utils.exceptions import SourceNotLoadedError
            raise SourceNotLoadedError("all")

        # ── Step 3: Re-rank per source ────────────────────────────────────────
        ranked_results: dict[str, list[RankedChunk]] = {}
        for label, candidates in hybrid_results.items():
            if candidates:
                ranked = self._reranker.rerank(
                    query=user_question,
                    chunks=candidates,
                    top_k=self._top_k,
                )
                ranked_results[label] = ranked
            else:
                ranked_results[label] = []

        # ── Step 4: Compute confidence per source ─────────────────────────────
        confidences: dict[str, SourceConfidence] = {}
        for label, ranked in ranked_results.items():
            if ranked:
                confidences[label] = self._reranker.compute_source_confidence(ranked)
                logger.info(
                    "[Source %s] Confidence: %s (%.2f)",
                    label,
                    confidences[label].label,
                    confidences[label].score,
                )

        # ── Step 5: Detect A vs C conflict ────────────────────────────────────
        conflict_report: ConflictReport | None = None
        source_results: dict[str, SourceResult] = {}

        for label in ["A", "B", "C"]:
            ranked = ranked_results.get(label, [])
            is_loaded = label in self._faiss and self._faiss[label].is_ready
            source_results[label] = SourceResult(
                source_label=label,
                chunks=ranked,
                loaded=is_loaded,
                confidence=confidences.get(label),
            )

        # Run conflict detection if both A and C have results
        a_chunks = ranked_results.get("A", [])
        c_chunks = ranked_results.get("C", [])

        if a_chunks and c_chunks:
            # Get embeddings of top chunks for semantic similarity
            a_emb = self._embedder.embed([a_chunks[0].text]) if a_chunks else None
            c_emb = self._embedder.embed([c_chunks[0].text]) if c_chunks else None
            conflict_report = detect_conflicts(
                source_a_chunks=a_chunks,
                source_c_chunks=c_chunks,
                source_a_embeddings=a_emb,
                source_c_embeddings=c_emb,
            )
            if conflict_report.conflict_detected:
                logger.info(
                    "Conflict detected: type=%s severity=%s",
                    conflict_report.conflict_type,
                    conflict_report.severity,
                )

        # ── Step 6: Build prompt + call LLM ───────────────────────────────────
        context = self._build_context(source_results, conflict_report)
        prompt  = self._build_synthesis_prompt(user_question, source_results, conflict_report)

        router_result: RouterResult = self._router.generate(
            prompt=prompt,
            context=context,
            preferred_provider=preferred_provider,
        )

        # ── Step 7: Parse response ────────────────────────────────────────────
        parsed = _parse_structured_response(router_result.answer, source_results)

        conflict_detected = conflict_report is not None and conflict_report.conflict_detected
        conflict_details  = conflict_report.explanation if conflict_detected else ""

        return TruthEngineResponse(
            query=user_question,
            source_a=source_results.get("A"),
            source_b=source_results.get("B"),
            source_c=source_results.get("C"),
            conclusion=parsed["conclusion"],
            provider_used=router_result.provider_used,
            fallback_triggered=router_result.fallback_triggered,
            conflict_detected=conflict_detected,
            conflict_details=conflict_details,
            conflict_report=conflict_report,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_context(
        self,
        source_results: dict[str, SourceResult],
        conflict_report: ConflictReport | None,
    ) -> str:
        sections: list[str] = []

        for label in ["A", "B", "C"]:
            result = source_results.get(label)
            if not result or not result.loaded or not result.chunks:
                continue

            desc = SOURCE_DESCRIPTIONS.get(label, f"Source {label}")
            conf_label = result.confidence.label if result.confidence else "Unknown"
            header = f"=== SOURCE {label} ({desc}) | Confidence: {conf_label} ==="
            chunk_texts = [
                f"[{label}-Chunk {i} | Section ~{c.page_hint} | CE Score: {c.confidence:.2f}]\n{c.text}"
                for i, c in enumerate(result.chunks, start=1)
            ]
            sections.append(header + "\n\n" + "\n\n".join(chunk_texts))

        context = "\n\n" + ("=" * 60) + "\n\n".join(sections)

        # Append conflict notice to context if detected
        if conflict_report and conflict_report.conflict_detected:
            context += (
                f"\n\n{'=' * 60}\n"
                f"⚠️ CONFLICT DETECTED (Severity: {conflict_report.severity.upper()})\n"
                f"Source A excerpt: {conflict_report.source_a_excerpt[:200]}\n"
                f"Source C excerpt: {conflict_report.source_c_excerpt[:200]}\n"
            )

        return context

    def _build_synthesis_prompt(
        self,
        user_question: str,
        source_results: dict[str, SourceResult],
        conflict_report: ConflictReport | None,
    ) -> str:
        loaded_sources = [
            label for label in ["A", "B", "C"]
            if source_results.get(label) and source_results[label].loaded and source_results[label].chunks
        ]

        source_lines = []
        for label in loaded_sources:
            desc = SOURCE_DESCRIPTIONS.get(label, f"Source {label}")
            conf = source_results[label].confidence
            conf_str = f" (Confidence: {conf.label})" if conf else ""
            source_lines.append(f"- Source {label}: {desc}{conf_str}")

        sources_desc = "\n".join(source_lines)
        loaded_str   = ", ".join(f"Source {l}" for l in loaded_sources)

        conflict_instruction = ""
        if conflict_report and conflict_report.conflict_detected:
            conflict_instruction = f"""
⚠️ CONFLICT ALERT: A conflict has been detected between Source A and Source C.
- Conflict type: {conflict_report.conflict_type}
- Severity: {conflict_report.severity}
- {conflict_report.explanation}

IMPORTANT: Present BOTH Source A's version AND Source C's version clearly.
Show the user exactly what each source says about the conflicting point.
In the Conclusion, explain the conflict and recommend trusting Source A (Golden Truth),
but explicitly show Source C's version so the user can decide.
"""

        prompt = f"""You are the Truth Engine — a precise, multi-source document analyst.

LOADED SOURCES:
{sources_desc}

PRIORITY RULES:
1. Source A is the GOLDEN TRUTH. Always trust Source A over others.
2. Source B contains real-world fixes — useful but may be experimental.
3. Source C is LEGACY and may be deprecated or wrong.
4. If you cannot find the answer, say "I don't know" — do NOT hallucinate.
{conflict_instruction}
QUESTION: {user_question}

Using ONLY the provided context from {loaded_str}, respond in this EXACT format:

{self._build_response_template(loaded_sources, conflict_report)}"""

        return prompt

    def _build_response_template(
        self,
        loaded_sources: list[str],
        conflict_report: ConflictReport | None,
    ) -> str:
        lines = []
        for label in loaded_sources:
            desc = SOURCE_DESCRIPTIONS.get(label, f"Source {label}")
            if label == "C" and conflict_report and conflict_report.conflict_detected:
                lines.append(
                    f"**Source {label} says:**\n"
                    f"[IMPORTANT: This source may conflict with Source A. "
                    f"State exactly what Source {label} ({desc}) says, even if it differs from Source A. "
                    f"If Source {label} has no relevant information, write 'Source {label} does not address this.']"
                )
            else:
                lines.append(
                    f"**Source {label} says:**\n"
                    f"[What Source {label} ({desc}) states. "
                    f"Quote specific details. If no relevant info, write 'Source {label} does not address this.']"
                )
            lines.append("")

        lines.append("**Conclusion:**")
        if conflict_report and conflict_report.conflict_detected:
            lines.append(
                "[REQUIRED: 1) Acknowledge the conflict between Source A and Source C. "
                "2) Show BOTH versions side by side. "
                "3) Recommend trusting Source A as the golden truth, but preserve Source C's version "
                "so the user can make their own decision. "
                "4) State the final recommended answer clearly.]"
            )
        else:
            lines.append(
                "[Synthesise all sources. State which is most authoritative. "
                "Flag any contradictions. Give the final recommended answer based on source priority.]"
            )

        return "\n".join(lines)


# ── Response parser (identical to Tier 1) ────────────────────────────────────

def _parse_structured_response(
    raw_answer: str,
    source_results: dict[str, SourceResult],
) -> dict:
    import re

    result = {
        "source_a_text": "",
        "source_b_text": "",
        "source_c_text": "",
        "conclusion": "",
    }

    for label, key in [("A", "source_a_text"), ("B", "source_b_text"), ("C", "source_c_text")]:
        pattern = rf"\*\*Source {label} says:\*\*\s*(.*?)(?=\*\*Source [A-Z] says:\*\*|\*\*Conclusion:\*\*|$)"
        match = re.search(pattern, raw_answer, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    conc_match = re.search(r"\*\*Conclusion:\*\*\s*(.*?)$", raw_answer, re.DOTALL | re.IGNORECASE)
    result["conclusion"] = conc_match.group(1).strip() if conc_match else raw_answer.strip()

    for label, key in [("A", "source_a_text"), ("B", "source_b_text"), ("C", "source_c_text")]:
        if label in source_results and result[key]:
            source_results[label].summary = result[key]

    return result
