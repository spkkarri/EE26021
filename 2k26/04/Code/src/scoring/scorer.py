"""
scoring/scorer.py — Final Scoring Function & Pipeline Orchestrator

Mathematical formula
--------------------
Let:
  S_sem     = aggregated multi-reference semantic similarity      ∈ [0, 1]
  S_concept = weighted concept coverage score                     ∈ [0, 1]
  P_contra  = total NLI contradiction penalty                     ∈ [0, 1]
  P_length  = length mismatch penalty                             ∈ [0, 1]

Raw score (before scaling to marks):
  S_raw = w_sem * S_sem
        + w_concept * S_concept
        - w_contra  * clamp(P_contra, 0, 1)
        - w_length  * P_length

Final mark out of MAX_MARKS:
  Mark = clamp(S_raw, 0, 1) * MAX_MARKS

Length penalty:
  ratio = len(student_words) / len(reference_words)

  P_length = max(0, 1 - ratio / min_ratio)   if ratio < min_ratio  (too short)
           = max(0, 1 - max_ratio / ratio)   if ratio > max_ratio  (too long)
           = 0                               otherwise

Design decisions:
  - Concept score carries the highest weight (0.50) because it's the most
    faithful to "did the student explain the key ideas?"
  - Semantic similarity (0.30) handles fluency and overall proximity.
  - Contradiction penalty (0.15) catches factually wrong but
    keyword-heavy answers.
  - Length penalty (0.05) discourages trivially short or bloated answers.
  - All weights are configurable in config.py — teachers can re-tune per subject.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.config import ScoringConfig, DEFAULT_CONFIG
from src.embedding.encoder import SemanticEncoder, get_encoder
from src.scoring.concept_engine import ConceptScoringEngine, ConceptEngineResult
from src.scoring.multi_reference import MultiReferenceMatcher, MultiRefResult
from src.nli.contradiction import ContradictionDetector, ContradictionResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Complete evaluation output for a single (question, student_answer) pair.
    All sub-module outputs are preserved for explainability.
    """
    # Inputs
    question: str
    student_answer: str
    reference_answers: List[str]

    # Intermediate scores (all in [0, 1])
    semantic_score: float
    concept_score: float
    contradiction_penalty: float
    length_penalty: float

    # Final outputs
    raw_score: float                # weighted combination, [0, 1]
    final_marks: float              # raw_score * max_marks
    max_marks: float
    confidence: float               # proxy: mean entailment score

    # Sub-module results (kept for explainability)
    multi_ref_result: MultiRefResult
    concept_result: ConceptEngineResult
    contradiction_result: ContradictionResult

    # Score breakdown (for display)
    score_breakdown: dict = field(default_factory=dict)

    def __post_init__(self):
        self.score_breakdown = {
            "Semantic similarity":      round(self.semantic_score, 4),
            "Concept coverage":         round(self.concept_score, 4),
            "Contradiction penalty":    round(self.contradiction_penalty, 4),
            "Length penalty":           round(self.length_penalty, 4),
            "Raw composite score":      round(self.raw_score, 4),
            "Final marks":              round(self.final_marks, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

class AnswerEvaluator:
    """
    Main pipeline orchestrator.

    Instantiate once, call `evaluate()` for each answer.
    Sub-modules are lazy-loaded on first use.
    """

    def __init__(
        self,
        config: ScoringConfig = DEFAULT_CONFIG.scoring,
        encoder: Optional[SemanticEncoder] = None,
        enable_nli: bool = True,
    ):
        self.config = config
        self._encoder = encoder
        self.enable_nli = enable_nli

        self._concept_engine = ConceptScoringEngine(config)
        self._matcher = MultiReferenceMatcher(aggregation="max")
        self._detector = ContradictionDetector() if enable_nli else None

    @property
    def encoder(self) -> SemanticEncoder:
        if self._encoder is None:
            self._encoder = get_encoder()
        return self._encoder

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        question: str,
        student_answer: str,
        reference_answers: List[str],
        reference_weights: Optional[List[float]] = None,
        concept_weights: Optional[List[float]] = None,
        use_hungarian: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a student answer against one or more reference answers.

        Args:
            question:            The exam question (used for context in explanations).
            student_answer:      The student's response text.
            reference_answers:   List of model answers (at least 1).
            reference_weights:   Relative importance of each reference answer.
            concept_weights:     Relative importance of each extracted concept.
            use_hungarian:       Use Hungarian algorithm for concept-sentence alignment.

        Returns:
            EvaluationResult with final marks and full breakdown.
        """
        # Guard: empty answer
        if not student_answer.strip():
            return self._zero_result(question, student_answer, reference_answers)

        # --- Step 1: Multi-reference semantic similarity ---
        best_reference = self._select_best_reference(reference_answers)
        multi_ref = self._matcher.match(
            student_answer=student_answer,
            reference_answers=reference_answers,
            encoder=self.encoder,
            reference_weights=reference_weights,
        )
        S_sem = multi_ref.aggregated_similarity

        # --- Step 2: Concept-level scoring ---
        concept_res = self._concept_engine.score(
            reference_answer=best_reference,
            student_answer=student_answer,
            encoder=self.encoder,
            concept_weights=concept_weights,
            use_hungarian=use_hungarian,
        )
        S_concept = concept_res.raw_concept_score

        # --- Step 3: NLI contradiction detection ---
        if self.enable_nli and self._detector is not None:
            contra_res = self._detector.analyse(
                student_answer=student_answer,
                reference_answer=best_reference,
            )
            P_contra = min(1.0, contra_res.total_penalty)
            confidence = contra_res.confidence
        else:
            contra_res = self._dummy_contra_result()
            P_contra = 0.0
            confidence = S_sem

        # --- Step 4: Length penalty ---
        P_length = self._length_penalty(student_answer, best_reference)

        # --- Step 5: Composite score ---
        cfg = self.config
        S_raw = (
            cfg.w_semantic      * S_sem
            + cfg.w_concept     * S_concept
            - cfg.w_contradiction * P_contra
            - cfg.w_length      * P_length
        )
        S_raw = float(np.clip(S_raw, 0.0, 1.0))
        final_marks = round(S_raw * cfg.max_marks, 2)

        logger.debug(
            f"S_sem={S_sem:.3f}  S_concept={S_concept:.3f}  "
            f"P_contra={P_contra:.3f}  P_length={P_length:.3f}  "
            f"→ raw={S_raw:.3f}  marks={final_marks}"
        )

        return EvaluationResult(
            question=question,
            student_answer=student_answer,
            reference_answers=reference_answers,
            semantic_score=S_sem,
            concept_score=S_concept,
            contradiction_penalty=P_contra,
            length_penalty=P_length,
            raw_score=S_raw,
            final_marks=final_marks,
            max_marks=cfg.max_marks,
            confidence=confidence,
            multi_ref_result=multi_ref,
            concept_result=concept_res,
            contradiction_result=contra_res,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _select_best_reference(self, references: List[str]) -> str:
        """
        Choose the primary reference for concept extraction.
        We pick the longest (most detailed) reference as the gold standard.
        """
        return max(references, key=lambda r: len(r.split()))

    def _length_penalty(self, student: str, reference: str) -> float:
        """
        Compute a length-normalisation penalty in [0, 1].

        Too short:  ratio < min_ratio → penalty scales up to 1 as ratio → 0
        Too long:   ratio > max_ratio → penalty scales up as ratio → ∞
        Just right: min_ratio ≤ ratio ≤ max_ratio → 0 penalty
        """
        ref_words = len(reference.split())
        stu_words = len(student.split())
        if ref_words == 0:
            return 0.0
        ratio = stu_words / ref_words
        lo = self.config.min_length_ratio
        hi = self.config.max_length_ratio

        if ratio < lo:
            # Linear ramp: 0 at ratio=lo, 1 at ratio=0
            return 1.0 - ratio / lo
        if ratio > hi:
            # Linear ramp: 0 at ratio=hi, saturates at 1
            return min(1.0, (ratio - hi) / hi)
        return 0.0

    def _zero_result(self, q, sa, refs) -> EvaluationResult:
        from src.nli.contradiction import ContradictionResult
        from src.scoring.concept_engine import ConceptEngineResult
        from src.scoring.multi_reference import MultiRefResult
        return EvaluationResult(
            question=q, student_answer=sa, reference_answers=refs,
            semantic_score=0.0, concept_score=0.0,
            contradiction_penalty=0.0, length_penalty=1.0,
            raw_score=0.0, final_marks=0.0,
            max_marks=self.config.max_marks, confidence=0.0,
            multi_ref_result=MultiRefResult(
                similarity_matrix=np.zeros((1,1)),
                per_reference_similarity=[0.0],
                aggregated_similarity=0.0, best_reference_idx=0,
                aggregation_method="max", reference_weights=[1.0],
            ),
            concept_result=ConceptEngineResult(
                concepts=[], concept_scores=[], similarity_matrix=np.zeros((1,1)),
                alignment=[], raw_concept_score=0.0,
                covered_concepts=[], missing_concepts=[], coverage_ratio=0.0,
            ),
            contradiction_result=self._dummy_contra_result(),
        )

    def _dummy_contra_result(self) -> ContradictionResult:
        from src.nli.contradiction import ContradictionResult
        return ContradictionResult(
            nli_pairs=[], contradiction_count=0, entailment_count=0,
            total_penalty=0.0, has_contradiction=False,
            confidence=0.5, summary="NLI disabled.",
        )
