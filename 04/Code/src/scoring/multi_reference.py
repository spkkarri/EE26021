"""
scoring/multi_reference.py — Multi-Reference Answer Matching

Problem: A question may have multiple valid reference answers (e.g., the
teacher provided three model answers of varying depth). We need to aggregate
similarity against all references without double-counting.

Aggregation strategies:
  1. MAX  — take the highest similarity to any single reference.
             Good when references are alternatives (same concept, different phrasing).
  2. WEIGHTED — weighted average where weights reflect answer quality/depth.
             Good when references are graded (Gold=1.0, Silver=0.7, ...).
  3. SOFT-MAX — differentiable approximation of MAX; softens the winner-takes-all
             behaviour. Temperature parameter controls sharpness.

Design note: We return a full similarity MATRIX (student_sents × references)
rather than a scalar so the explainability module can show which reference
each student sentence matched best.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from scipy.special import softmax as scipy_softmax

from src.config import DEFAULT_CONFIG


@dataclass
class MultiRefResult:
    """Output of multi-reference matching."""
    # Raw matrix: rows = student sentences, cols = references
    similarity_matrix: np.ndarray             # shape: (n_refs, 1) for whole-answer mode
    # Per-reference similarity (whole-answer to whole-reference)
    per_reference_similarity: List[float]
    # Final aggregated score (method-dependent)
    aggregated_similarity: float
    best_reference_idx: int
    aggregation_method: str
    reference_weights: List[float]


class MultiReferenceMatcher:
    """
    Match a student answer against multiple reference answers.

    Aggregation methods:
        "max"      — argmax similarity (no hyperparameters)
        "weighted" — weighted average (provide reference_weights)
        "softmax"  — temperature-scaled soft-max aggregation
    """

    def __init__(
        self,
        aggregation: Literal["max", "weighted", "softmax"] = "max",
        temperature: float = 0.5,   # Lower → sharper (more like hard max)
    ):
        self.aggregation = aggregation
        self.temperature = temperature

    def match(
        self,
        student_answer: str,
        reference_answers: List[str],
        encoder,                         # SemanticEncoder
        reference_weights: Optional[List[float]] = None,
    ) -> MultiRefResult:
        """
        Compute aggregated similarity of student_answer to all references.

        Args:
            student_answer:    Student's text.
            reference_answers: List of reference texts (1 or more).
            encoder:           SemanticEncoder instance.
            reference_weights: Optional importance weights per reference.
                               Normalised internally.

        Returns:
            MultiRefResult with full breakdown.
        """
        n_refs = len(reference_answers)

        # Default: uniform weights
        if reference_weights is None:
            reference_weights = [1.0 / n_refs] * n_refs
        else:
            total = sum(reference_weights) or 1.0
            reference_weights = [w / total for w in reference_weights]

        # Embed student answer and all references
        student_emb = encoder.encode([student_answer])             # (1, D)
        ref_embs    = encoder.encode(reference_answers)            # (n_refs, D)

        # Pairwise similarities: (n_refs,)
        sims = (ref_embs @ student_emb.T).flatten()
        sims = np.clip(sims, 0.0, 1.0).tolist()

        # Aggregate
        agg = self._aggregate(sims, reference_weights)
        best_idx = int(np.argmax(sims))

        return MultiRefResult(
            similarity_matrix=np.array(sims).reshape(-1, 1),
            per_reference_similarity=sims,
            aggregated_similarity=float(agg),
            best_reference_idx=best_idx,
            aggregation_method=self.aggregation,
            reference_weights=reference_weights,
        )

    def _aggregate(self, sims: List[float], weights: List[float]) -> float:
        sims_arr = np.array(sims)
        weights_arr = np.array(weights)

        if self.aggregation == "max":
            return float(np.max(sims_arr))

        if self.aggregation == "weighted":
            return float(np.dot(weights_arr, sims_arr))

        if self.aggregation == "softmax":
            # Soft-max: weights are sharpened by similarity-based attention
            # α_i = softmax(sim_i / τ)
            # agg = Σ α_i * sim_i
            alphas = scipy_softmax(sims_arr / self.temperature)
            return float(np.dot(alphas, sims_arr))

        raise ValueError(f"Unknown aggregation method: {self.aggregation!r}")

    # ------------------------------------------------------------------ #
    #  Sentence-Level Matching                                             #
    # ------------------------------------------------------------------ #

    def sentence_level_matrix(
        self,
        student_sentences: List[str],
        reference_answers: List[str],
        encoder,
    ) -> np.ndarray:
        """
        Build a (n_student_sents × n_refs) similarity matrix for heatmap
        visualisation and fine-grained alignment.
        """
        if not student_sentences or not reference_answers:
            return np.zeros((1, 1))
        return encoder.similarity_matrix(student_sentences, reference_answers)
