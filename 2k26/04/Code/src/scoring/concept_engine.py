"""
scoring/concept_engine.py — Concept-Level Scoring Engine

Core idea: instead of treating a reference answer as a monolithic blob,
we decompose it into *atomic concepts* — minimal meaningful units — and
score the student answer against each concept independently.

This gives us:
  1. Partial marks (student covered 3/5 concepts → 60% credit)
  2. Explainability (we know *which* concepts were missed)
  3. Robustness to order-scrambling / paraphrasing

Decomposition strategy:
  - Primary:  spaCy sentence segmentation + noun-chunk extraction
  - Fallback: rule-based splitting on punctuation + conjunctions

Alignment strategy:
  - For each concept, find the *best matching* student sentence
    (max-similarity greedy) or use the Hungarian algorithm for
    optimal bipartite matching (controlled by `use_hungarian` flag).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

from src.config import ScoringConfig, DEFAULT_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConceptScore:
    """Result for a single atomic concept."""
    concept_text: str
    best_match_sentence: str       # Which student sentence matched best
    similarity: float              # Raw cosine similarity in [0, 1]
    credit: float                  # Awarded credit (0, 0–1 scaled, or 1)
    weight: float                  # Concept importance weight
    weighted_credit: float         # credit * weight
    covered: bool                  # True if similarity ≥ partial_credit threshold


@dataclass
class ConceptEngineResult:
    """Full output from the concept scoring engine."""
    concepts: List[str]
    concept_scores: List[ConceptScore]
    similarity_matrix: np.ndarray          # Shape: (n_concepts, n_student_sentences)
    alignment: List[Tuple[int, int]]       # (concept_idx, student_sent_idx) pairs
    raw_concept_score: float               # Weighted mean of credits (0–1)
    covered_concepts: List[str]
    missing_concepts: List[str]
    coverage_ratio: float                  # |covered| / |total|


# ─────────────────────────────────────────────────────────────────────────────
# Concept Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ConceptExtractor:
    """
    Decomposes a reference answer into atomic concepts.

    Two modes:
      - spaCy mode: sentence segmentation → clause splitting
      - Fallback:   regex splitting on '.', ';', ' and ', ' because '
    """

    # Conjunctions/discourse markers that split at clause level
    _SPLIT_PATTERN = re.compile(
        r"\s*(?:,\s*(?:and|but|or|however|therefore|because|since|which|that)|"
        r";\s*|—\s*|\.\s+)",
        re.IGNORECASE,
    )

    def extract(self, reference_answer: str, min_length: int = 10) -> List[str]:
        """
        Extract atomic concepts from `reference_answer`.

        Args:
            reference_answer: The model answer text.
            min_length:       Discard fragments shorter than this (noise filter).

        Returns:
            List of concept strings, deduplicated and ordered.
        """
        if SPACY_AVAILABLE:
            concepts = self._extract_spacy(reference_answer)
        else:
            concepts = self._extract_regex(reference_answer)

        # Post-process: strip, filter short, deduplicate
        seen = set()
        clean = []
        for c in concepts:
            c = c.strip()
            if len(c) >= min_length and c.lower() not in seen:
                seen.add(c.lower())
                clean.append(c)
        return clean

    def _extract_spacy(self, text: str) -> List[str]:
        doc = _nlp(text)
        concepts = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Further split on coordinating conjunctions
            sub_clauses = self._split_on_conjunctions(sent_text)
            concepts.extend(sub_clauses)
        return concepts

    def _split_on_conjunctions(self, sentence: str) -> List[str]:
        """Split on ', and', 'because', 'since', 'which' etc."""
        parts = self._SPLIT_PATTERN.split(sentence)
        return [p.strip() for p in parts if p.strip()]

    def _extract_regex(self, text: str) -> List[str]:
        """Fallback: split on sentence boundaries."""
        parts = re.split(r"[.;]\s+|\n", text)
        return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Concept Scoring Engine
# ─────────────────────────────────────────────────────────────────────────────

class ConceptScoringEngine:
    """
    Scores student answers at the concept level.

    Partial credit formula per concept c_i:
      credit(c_i) = 0.0                           if sim < partial_threshold
                  = (sim - partial_threshold)
                    / (full_threshold - partial_threshold)  if partial ≤ sim < full
                  = 1.0                           if sim ≥ full_threshold

    Weighted concept score:
      S_concept = Σ (weight_i * credit_i) / Σ weight_i

    Concept weights:
      By default, all concepts are equally weighted (uniform).
      You may pass a custom weight vector (e.g., from TF-IDF importance).
    """

    def __init__(self, config: ScoringConfig = DEFAULT_CONFIG.scoring):
        self.config = config
        self.extractor = ConceptExtractor()

    def _credit(self, similarity: float) -> float:
        """Convert raw similarity to partial credit in [0, 1]."""
        lo = self.config.concept_partial_credit
        hi = self.config.concept_full_credit
        if similarity >= hi:
            return 1.0
        if similarity >= lo:
            return (similarity - lo) / (hi - lo)
        return 0.0

    def score(
        self,
        reference_answer: str,
        student_answer: str,
        encoder,                        # SemanticEncoder instance
        concept_weights: Optional[List[float]] = None,
        use_hungarian: bool = True,
    ) -> ConceptEngineResult:
        """
        Full concept-level scoring pipeline.

        Args:
            reference_answer: Model answer (or concatenation of multiple).
            student_answer:   Student's response.
            encoder:          SemanticEncoder for embedding.
            concept_weights:  Optional importance weights per concept.
                              Defaults to uniform.
            use_hungarian:    Use optimal assignment (Hungarian) instead of
                              greedy max-similarity alignment.

        Returns:
            ConceptEngineResult with full breakdown.
        """
        # 1. Extract atomic concepts from reference
        concepts = self.extractor.extract(reference_answer)
        if not concepts:
            # Edge case: very short reference
            concepts = [reference_answer.strip()]

        # 2. Segment student answer into sentences
        student_sentences = self._segment(student_answer)
        if not student_sentences:
            student_sentences = [student_answer.strip()]

        # 3. Build similarity matrix: (n_concepts × n_student_sents)
        sim_matrix = encoder.similarity_matrix(concepts, student_sentences)
        # Clip to [0, 1] — cosine can be slightly negative for unrelated text
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

        # 4. Alignment: for each concept, find best-matching student sentence
        if use_hungarian and len(concepts) <= len(student_sentences):
            alignment = self._hungarian_align(sim_matrix)
        else:
            alignment = self._greedy_align(sim_matrix)

        # 5. Uniform weights unless provided
        if concept_weights is None:
            concept_weights = [1.0 / len(concepts)] * len(concepts)
        else:
            total = sum(concept_weights) or 1.0
            concept_weights = [w / total for w in concept_weights]

        # 6. Compute per-concept scores
        concept_scores: List[ConceptScore] = []
        for c_idx, s_idx in alignment:
            sim = float(sim_matrix[c_idx, s_idx])
            cr = self._credit(sim)
            w = concept_weights[c_idx]
            concept_scores.append(ConceptScore(
                concept_text=concepts[c_idx],
                best_match_sentence=student_sentences[s_idx],
                similarity=sim,
                credit=cr,
                weight=w,
                weighted_credit=cr * w,
                covered=sim >= self.config.concept_partial_credit,
            ))

        # 7. Aggregate
        raw_score = sum(cs.weighted_credit for cs in concept_scores)
        covered = [cs.concept_text for cs in concept_scores if cs.covered]
        missing = [cs.concept_text for cs in concept_scores if not cs.covered]
        coverage_ratio = len(covered) / len(concepts) if concepts else 0.0

        return ConceptEngineResult(
            concepts=concepts,
            concept_scores=concept_scores,
            similarity_matrix=sim_matrix,
            alignment=alignment,
            raw_concept_score=float(raw_score),
            covered_concepts=covered,
            missing_concepts=missing,
            coverage_ratio=coverage_ratio,
        )

    # ------------------------------------------------------------------ #
    #  Alignment Algorithms                                                #
    # ------------------------------------------------------------------ #

    def _greedy_align(
        self, sim_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Greedy alignment: for each concept (row), pick the student sentence
        (column) with the highest similarity.

        Time: O(n_concepts × n_student_sents).
        Multiple concepts can match the same sentence (expected behaviour —
        a student sentence may address several concepts simultaneously).
        """
        return [
            (c_idx, int(np.argmax(sim_matrix[c_idx])))
            for c_idx in range(sim_matrix.shape[0])
        ]

    def _hungarian_align(
        self, sim_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Optimal bipartite matching via the Hungarian algorithm.
        Maximises total alignment similarity so each student sentence is
        assigned to at most one concept.

        Uses scipy.optimize.linear_sum_assignment on the negated matrix
        (it solves minimisation problems).

        Falls back to greedy when concepts > student sentences.
        """
        n_c, n_s = sim_matrix.shape
        if n_c > n_s:
            return self._greedy_align(sim_matrix)
        row_idx, col_idx = linear_sum_assignment(-sim_matrix)
        return list(zip(row_idx.tolist(), col_idx.tolist()))

    # ------------------------------------------------------------------ #
    #  Sentence Segmentation                                               #
    # ------------------------------------------------------------------ #

    def _segment(self, text: str) -> List[str]:
        """Split student answer into sentences."""
        if SPACY_AVAILABLE:
            doc = _nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        # Regex fallback
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s]
