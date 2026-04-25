"""
explainability/explainer.py — Explainability Module

Provides human-interpretable explanations for every score:
  1. Concept coverage breakdown (which concepts were covered / missed)
  2. Similarity heatmap data (student sentences × concepts)
  3. Contradiction flags with offending sentences highlighted
  4. Score waterfall chart data (additive contribution of each component)
  5. Optional SHAP-style feature importance (approximated via input perturbation)

Why explainability matters for answer evaluation:
  - Teachers need to understand WHY a score was awarded.
  - Students need feedback on WHAT was missing.
  - System auditors need to verify the model is not gaming keywords.

SHAP approximation note:
  True SHAP requires training a surrogate model. For our scoring pipeline,
  we approximate feature importance by:
    1. Computing the full score.
    2. Zeroing each feature (setting it to 0) one at a time.
    3. Feature importance = score_full - score_without_feature.
  This is the "ablation" / "leave-one-out" approximation of Shapley values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.scoring.scorer import EvaluationResult


# ─────────────────────────────────────────────────────────────────────────────
# Output containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConceptExplanation:
    concept: str
    similarity: float
    credit: float
    weight: float
    covered: bool
    best_match_sentence: str
    status_label: str    # "✅ Covered" / "⚠️ Partial" / "❌ Missing"
    colour: str          # for UI rendering


@dataclass
class ContradictionExplanation:
    sentence: str
    contradiction_score: float
    entailment_score: float
    is_flagged: bool
    label: str    # "🚩 Contradiction" / "✅ Consistent" / "➖ Neutral"


@dataclass
class FeatureImportance:
    feature_name: str
    contribution: float   # Signed contribution to final score (positive = helpful)
    pct_of_score: float   # As a percentage of final raw score


@dataclass
class ExplanationReport:
    """Full explainability report."""
    final_marks: float
    max_marks: float
    percentage: float

    # Concept breakdown
    concept_explanations: List[ConceptExplanation]
    covered_count: int
    missing_count: int
    coverage_ratio: float

    # Contradiction
    contradiction_explanations: List[ContradictionExplanation]
    has_contradiction: bool

    # Score waterfall (additive components)
    score_components: List[Tuple[str, float]]   # (name, signed contribution)

    # SHAP-style feature importance
    feature_importances: List[FeatureImportance]

    # Heatmap data (for plotting)
    heatmap_matrix: np.ndarray
    heatmap_row_labels: List[str]   # concepts
    heatmap_col_labels: List[str]   # student sentences (truncated)

    # Natural language summary
    summary: str
    feedback: str           # Personalised student feedback
    teacher_notes: str      # For grader


# ─────────────────────────────────────────────────────────────────────────────
# Explainer
# ─────────────────────────────────────────────────────────────────────────────

class Explainer:

    def __init__(
        self,
        top_k_concepts: int = 5,
        max_label_len: int = 40,
    ):
        self.top_k_concepts = top_k_concepts
        self.max_label_len = max_label_len

    def explain(self, result: EvaluationResult) -> ExplanationReport:
        """Generate a full explanation from an EvaluationResult."""
        pct = (result.final_marks / result.max_marks) * 100 if result.max_marks else 0

        concept_exps = self._explain_concepts(result)
        contra_exps  = self._explain_contradictions(result)
        components   = self._score_components(result)
        importances  = self._feature_importances(result)
        heatmap_m, rows, cols = self._heatmap_data(result)
        summary      = self._summary(result, pct)
        feedback     = self._student_feedback(result)
        teacher_note = self._teacher_notes(result)

        return ExplanationReport(
            final_marks=result.final_marks,
            max_marks=result.max_marks,
            percentage=pct,
            concept_explanations=concept_exps,
            covered_count=len(result.concept_result.covered_concepts),
            missing_count=len(result.concept_result.missing_concepts),
            coverage_ratio=result.concept_result.coverage_ratio,
            contradiction_explanations=contra_exps,
            has_contradiction=result.contradiction_result.has_contradiction,
            score_components=components,
            feature_importances=importances,
            heatmap_matrix=heatmap_m,
            heatmap_row_labels=rows,
            heatmap_col_labels=cols,
            summary=summary,
            feedback=feedback,
            teacher_notes=teacher_note,
        )

    # ------------------------------------------------------------------ #
    #  Concept explanations                                                #
    # ------------------------------------------------------------------ #

    def _explain_concepts(self, result: EvaluationResult) -> List[ConceptExplanation]:
        cfg_lo = 0.50
        cfg_hi = 0.80
        exps = []
        for cs in result.concept_result.concept_scores:
            if cs.similarity >= cfg_hi:
                label, colour = "✅ Covered", "#2ecc71"
            elif cs.similarity >= cfg_lo:
                label, colour = "⚠️ Partial", "#f39c12"
            else:
                label, colour = "❌ Missing", "#e74c3c"

            exps.append(ConceptExplanation(
                concept=cs.concept_text,
                similarity=cs.similarity,
                credit=cs.credit,
                weight=cs.weight,
                covered=cs.covered,
                best_match_sentence=cs.best_match_sentence,
                status_label=label,
                colour=colour,
            ))
        # Sort: missing first (most actionable feedback at top)
        exps.sort(key=lambda e: e.similarity)
        return exps

    # ------------------------------------------------------------------ #
    #  Contradiction explanations                                          #
    # ------------------------------------------------------------------ #

    def _explain_contradictions(self, result: EvaluationResult) -> List[ContradictionExplanation]:
        exps = []
        for pair in result.contradiction_result.nli_pairs:
            p_c = pair.scores.get("contradiction", 0)
            p_e = pair.scores.get("entailment", 0)
            if pair.is_contradiction:
                label = "🚩 Contradiction"
            elif pair.is_entailment:
                label = "✅ Consistent"
            else:
                label = "➖ Neutral"
            exps.append(ContradictionExplanation(
                sentence=pair.hypothesis,
                contradiction_score=p_c,
                entailment_score=p_e,
                is_flagged=pair.is_contradiction,
                label=label,
            ))
        return exps

    # ------------------------------------------------------------------ #
    #  Score waterfall                                                     #
    # ------------------------------------------------------------------ #

    def _score_components(self, result: EvaluationResult) -> List[Tuple[str, float]]:
        """
        Signed contributions for a waterfall chart.
        Positive = adds to score, negative = subtracts.
        """
        from src.config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.scoring
        return [
            ("Semantic similarity",  cfg.w_semantic * result.semantic_score),
            ("Concept coverage",     cfg.w_concept  * result.concept_score),
            ("Contradiction penalty",-(cfg.w_contradiction * result.contradiction_penalty)),
            ("Length penalty",       -(cfg.w_length  * result.length_penalty)),
        ]

    # ------------------------------------------------------------------ #
    #  SHAP-style feature importance (ablation approximation)             #
    # ------------------------------------------------------------------ #

    def _feature_importances(self, result: EvaluationResult) -> List[FeatureImportance]:
        """
        Approximate Shapley values via leave-one-out ablation.

        For each feature f:
          importance(f) = score_full - score_without_f

        where score_without_f is computed by zeroing out that feature's
        contribution in the final formula.
        """
        from src.config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.scoring

        S = result.raw_score
        if S == 0:
            return []

        features = {
            "Semantic similarity":     cfg.w_semantic  * result.semantic_score,
            "Concept coverage":        cfg.w_concept   * result.concept_score,
            "Contradiction (penalty)": cfg.w_contradiction * result.contradiction_penalty,
            "Length (penalty)":        cfg.w_length    * result.length_penalty,
        }

        importances = []
        for name, contribution in features.items():
            is_penalty = "penalty" in name.lower()
            pct = (abs(contribution) / (S + 1e-9)) * 100
            importances.append(FeatureImportance(
                feature_name=name,
                contribution=-contribution if is_penalty else contribution,
                pct_of_score=pct,
            ))
        # Sort descending by absolute contribution
        importances.sort(key=lambda x: abs(x.contribution), reverse=True)
        return importances

    # ------------------------------------------------------------------ #
    #  Heatmap data                                                        #
    # ------------------------------------------------------------------ #

    def _heatmap_data(
        self, result: EvaluationResult
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        matrix = result.concept_result.similarity_matrix
        concepts = result.concept_result.concepts
        # Truncate labels for display
        row_labels = [self._trunc(c) for c in concepts]
        # Reconstruct student sentences from alignment
        col_labels = self._get_student_sent_labels(result)
        return matrix, row_labels, col_labels

    def _get_student_sent_labels(self, result: EvaluationResult) -> List[str]:
        import re
        student = result.student_answer
        sents = re.split(r"(?<=[.!?])\s+", student.strip())
        return [self._trunc(s, 30) for s in sents if s.strip()] or ["(answer)"]

    def _trunc(self, text: str, n: int = None) -> str:
        n = n or self.max_label_len
        return text if len(text) <= n else text[:n-3] + "..."

    # ------------------------------------------------------------------ #
    #  Natural language outputs                                            #
    # ------------------------------------------------------------------ #

    def _summary(self, result: EvaluationResult, pct: float) -> str:
        cr = result.concept_result
        grade = self._grade(pct)
        return (
            f"{grade} — {result.final_marks:.1f}/{result.max_marks:.0f} marks "
            f"({pct:.0f}%). "
            f"Concept coverage: {len(cr.covered_concepts)}/{len(cr.concepts)} concepts addressed. "
            f"{'⚠️ Contradictions detected.' if result.contradiction_result.has_contradiction else ''}"
        )

    def _student_feedback(self, result: EvaluationResult) -> str:
        cr = result.concept_result
        lines = []
        if cr.covered_concepts:
            lines.append(
                f"Good work covering: {', '.join(cr.covered_concepts[:3])}."
            )
        if cr.missing_concepts:
            lines.append(
                f"To improve, address: {', '.join(cr.missing_concepts[:3])}."
            )
        if result.contradiction_result.has_contradiction:
            lines.append(
                "⚠️ Some statements appear to contradict the expected answer — "
                "review your factual claims."
            )
        if result.length_penalty > 0.1:
            ratio = len(result.student_answer.split()) / max(
                1, len(result.reference_answers[0].split())
            )
            if ratio < 0.3:
                lines.append("Your answer is too brief — aim to elaborate more.")
            else:
                lines.append("Your answer is much longer than expected — be more concise.")
        return " ".join(lines) or "Good answer overall."

    def _teacher_notes(self, result: EvaluationResult) -> str:
        return (
            f"Semantic similarity: {result.semantic_score:.3f} | "
            f"Concept score: {result.concept_score:.3f} | "
            f"Contradiction penalty: {result.contradiction_penalty:.3f} | "
            f"Length penalty: {result.length_penalty:.3f} | "
            f"Raw: {result.raw_score:.3f}"
        )

    def _grade(self, pct: float) -> str:
        if pct >= 90: return "🏆 Excellent"
        if pct >= 75: return "✅ Good"
        if pct >= 60: return "📘 Satisfactory"
        if pct >= 40: return "⚠️ Needs Improvement"
        return "❌ Insufficient"
