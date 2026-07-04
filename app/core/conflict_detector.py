"""
core/conflict_detector.py — Conflict Detection (Tier 2)
Responsibility: Detect when Source C (legacy) contradicts Source A (golden truth).
Shows BOTH versions and lets the user decide — per your requirement.

Detection strategy:
  1. Semantic similarity: If top chunks from A and C are very DISSIMILAR on the
     same query, they may be contradicting each other.
  2. Keyword contradiction: Scan for numeric values, time durations, version
     numbers, or negation patterns that differ between A and C.
  3. LLM-assisted: The synthesis prompt explicitly asks the LLM to flag conflicts.

We use strategies 1 + 2 as a pre-LLM signal (fast, no extra API call).
Strategy 3 is baked into the prompt in truth_engine.py.
"""

import logging
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Semantic similarity threshold below which sources are considered potentially contradictory
# (cosine similarity — 0 = orthogonal, 1 = identical, -1 = opposite)
CONFLICT_SIMILARITY_THRESHOLD = 0.75

# Patterns that often signal contradictory numeric/time values
_NUMERIC_PATTERN    = re.compile(r"\b\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|ms|mb|gb|%|v\d)\b", re.I)
_VERSION_PATTERN    = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_NEGATION_PATTERN   = re.compile(r"\b(?:not|never|no longer|deprecated|removed|invalid|wrong|incorrect|do not)\b", re.I)


@dataclass
class ConflictReport:
    """Result of conflict detection between Source A and Source C."""
    conflict_detected: bool
    severity: str               # 'none', 'low', 'medium', 'high'
    conflict_type: str          # 'semantic', 'numeric', 'negation', 'none'
    source_a_excerpt: str       # Most relevant chunk from Source A
    source_c_excerpt: str       # Most relevant chunk from Source C
    explanation: str            # Human-readable explanation of the conflict
    numeric_conflicts: list[str] = field(default_factory=list)  # Specific conflicting values


def detect_conflicts(
    source_a_chunks: list,   # list[RankedChunk] from Source A
    source_c_chunks: list,   # list[RankedChunk] from Source C
    source_a_embeddings: np.ndarray | None = None,
    source_c_embeddings: np.ndarray | None = None,
) -> ConflictReport:
    """
    Detect conflicts between Source A (golden) and Source C (legacy).

    Args:
        source_a_chunks:     Re-ranked chunks from Source A.
        source_c_chunks:     Re-ranked chunks from Source C.
        source_a_embeddings: Optional embeddings for semantic comparison.
        source_c_embeddings: Optional embeddings for semantic comparison.

    Returns:
        ConflictReport with detected conflicts and human-readable explanation.
    """
    # No conflict possible if either source is empty
    if not source_a_chunks or not source_c_chunks:
        return _no_conflict()

    a_text = source_a_chunks[0].text
    c_text = source_c_chunks[0].text

    # ── Check 1: Semantic similarity ──────────────────────────────────────────
    semantic_conflict = False
    similarity = 1.0   # default — assume similar

    if source_a_embeddings is not None and source_c_embeddings is not None:
        try:
            sim = _cosine_similarity(source_a_embeddings[0], source_c_embeddings[0])
            similarity = float(sim)
            semantic_conflict = similarity < CONFLICT_SIMILARITY_THRESHOLD
            logger.debug("Conflict detection: A-C cosine similarity = %.3f", similarity)
        except Exception as exc:
            logger.warning("Semantic similarity check failed: %s", exc)

    # ── Check 2: Numeric/value conflicts ──────────────────────────────────────
    numeric_a = set(_NUMERIC_PATTERN.findall(a_text.lower()))
    numeric_c = set(_NUMERIC_PATTERN.findall(c_text.lower()))

    version_a = set(_VERSION_PATTERN.findall(a_text))
    version_c = set(_VERSION_PATTERN.findall(c_text))

    # Values that appear in one source but not the other (potential contradiction)
    a_unique_nums = numeric_a - numeric_c
    c_unique_nums = numeric_c - numeric_a

    numeric_conflict    = bool(a_unique_nums or c_unique_nums)
    version_conflict    = bool(version_a and version_c and version_a != version_c)
    numeric_conflicts   = list(a_unique_nums | c_unique_nums)[:5]

    # ── Check 3: Negation patterns ────────────────────────────────────────────
    negation_in_c = bool(_NEGATION_PATTERN.search(c_text))
    negation_conflict = negation_in_c and not _NEGATION_PATTERN.search(a_text)

    # ── Determine overall conflict severity ───────────────────────────────────
    conflict_flags = [semantic_conflict, numeric_conflict, version_conflict, negation_conflict]
    num_flags = sum(conflict_flags)

    if num_flags == 0:
        return _no_conflict()

    if num_flags >= 3 or (semantic_conflict and numeric_conflict):
        severity = "high"
    elif num_flags == 2:
        severity = "medium"
    else:
        severity = "low"

    # Determine primary conflict type
    if numeric_conflict or version_conflict:
        conflict_type = "numeric"
    elif semantic_conflict:
        conflict_type = "semantic"
    elif negation_conflict:
        conflict_type = "negation"
    else:
        conflict_type = "semantic"

    explanation = _build_explanation(
        conflict_type=conflict_type,
        severity=severity,
        a_unique_nums=a_unique_nums,
        c_unique_nums=c_unique_nums,
        version_a=version_a,
        version_c=version_c,
        similarity=similarity,
        negation_in_c=negation_in_c,
    )

    logger.info(
        "Conflict detected: type=%s severity=%s similarity=%.2f numeric_diff=%s",
        conflict_type, severity, similarity, numeric_conflicts,
    )

    return ConflictReport(
        conflict_detected=True,
        severity=severity,
        conflict_type=conflict_type,
        source_a_excerpt=a_text[:300],
        source_c_excerpt=c_text[:300],
        explanation=explanation,
        numeric_conflicts=numeric_conflicts,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _no_conflict() -> ConflictReport:
    return ConflictReport(
        conflict_detected=False,
        severity="none",
        conflict_type="none",
        source_a_excerpt="",
        source_c_excerpt="",
        explanation="",
        numeric_conflicts=[],
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_explanation(
    conflict_type: str,
    severity: str,
    a_unique_nums: set,
    c_unique_nums: set,
    version_a: set,
    version_c: set,
    similarity: float,
    negation_in_c: bool,
) -> str:
    """Build a human-readable conflict explanation."""
    parts = []

    if conflict_type == "numeric" and (a_unique_nums or c_unique_nums):
        a_vals = ", ".join(sorted(a_unique_nums)[:3]) or "none"
        c_vals = ", ".join(sorted(c_unique_nums)[:3]) or "none"
        parts.append(
            f"Numerical values differ: Source A references **{a_vals}** "
            f"while Source C references **{c_vals}**."
        )

    if version_a and version_c and version_a != version_c:
        parts.append(
            f"Version numbers conflict: Source A mentions **{', '.join(version_a)}** "
            f"vs Source C mentions **{', '.join(version_c)}**."
        )

    if similarity < CONFLICT_SIMILARITY_THRESHOLD:
        parts.append(
            f"The two sources discuss this topic differently "
            f"(semantic similarity: {similarity:.0%})."
        )

    if negation_in_c:
        parts.append(
            "Source C contains deprecation or negation language "
            "(e.g. 'no longer', 'deprecated', 'do not') that may conflict with Source A."
        )

    severity_note = {
        "high":   "⚠️ **High confidence conflict** — the sources likely contradict each other.",
        "medium": "⚠️ **Possible conflict** — the sources may be inconsistent.",
        "low":    "ℹ️ **Minor difference** — the sources may address slightly different aspects.",
    }.get(severity, "")

    return severity_note + " " + " ".join(parts)
