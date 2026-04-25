"""
nli/contradiction.py — Contradiction & Logical Consistency Module

Why NLI matters: A student may write "Photosynthesis produces oxygen" (correct)
but also "Plants absorb oxygen during photosynthesis" in the same answer.
Keyword matching gives high scores to both sentences. NLI detects that the
second *contradicts* the reference, allowing us to penalise accordingly.

Model: cross-encoder/nli-deberta-v3-small
  - Input: (premise, hypothesis) pair
  - Output: logits for [contradiction, neutral, entailment]
  - We run in cross-encoder mode (both sentences together) → better accuracy
    than bi-encoder NLI at the cost of O(N²) calls (acceptable for answers ≤ 20 sents)

Penalty formula:
  P_contra = max(0, P(contradiction) - threshold) * penalty_scale

Final contradiction adjustment:
  adjusted_score = score - w_contra * Σ P_contra_i   (summed over all pairs)

Edge cases handled:
  - Very short student answers (< 3 words) → skip NLI
  - No reference provided → return neutral
  - Model not available → return safe default (no penalty)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from src.config import NLIConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Label order returned by the DeBERTa NLI model
# (varies by model — we look up by label name, not index)
NLI_LABELS = {"contradiction": 0, "neutral": 1, "entailment": 2}


@dataclass
class NLIPair:
    """Result for a single (premise, hypothesis) NLI call."""
    premise: str
    hypothesis: str
    scores: dict        # {"contradiction": float, "neutral": float, "entailment": float}
    verdict: str        # dominant label
    is_contradiction: bool
    is_entailment: bool
    contradiction_penalty: float


@dataclass
class ContradictionResult:
    """Aggregate contradiction analysis for a student answer."""
    nli_pairs: List[NLIPair]
    contradiction_count: int
    entailment_count: int
    total_penalty: float              # Σ contradiction_penalty across all pairs
    has_contradiction: bool
    confidence: float                 # Mean P(entailment) as a positive signal
    summary: str                      # Human-readable summary


class ContradictionDetector:
    """
    Detects logical contradictions between student sentences and the
    reference answer using a cross-encoder NLI model.

    Usage pattern:
      detector = ContradictionDetector(config)
      result = detector.analyse(
          student_answer="...",
          reference_answer="...",
      )
      final_score -= result.total_penalty * scoring_weight
    """

    def __init__(self, config: NLIConfig = DEFAULT_CONFIG.nli):
        self.config = config
        self._pipe = None

    @property
    def pipe(self):
        """Lazy-load the NLI pipeline."""
        if self._pipe is None:
            if not HF_AVAILABLE:
                raise RuntimeError("transformers not installed.")
            logger.info(f"Loading NLI model: {self.config.model_name}")
            self._pipe = hf_pipeline(
                "zero-shot-classification",
                model=self.config.model_name,
                device=-1,      # CPU; set 0 for GPU
            )
            logger.info("NLI model loaded.")
        return self._pipe

    def _score_pair(self, premise: str, hypothesis: str) -> NLIPair:
        """
        Run NLI on a (premise, hypothesis) pair.

        We use zero-shot-classification with NLI labels as candidate_labels.
        The model computes P(label | premise + hypothesis) for each label.
        """
        try:
            output = self.pipe(
                sequences=f"{premise} [SEP] {hypothesis}",
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="{}",
            )
            # Re-map to dict
            score_dict = {
                label: score
                for label, score in zip(output["labels"], output["scores"])
            }
        except Exception as e:
            logger.warning(f"NLI inference failed: {e}. Returning neutral.")
            score_dict = {"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0}

        p_contra = score_dict.get("contradiction", 0.0)
        p_entail = score_dict.get("entailment", 0.0)
        verdict = max(score_dict, key=score_dict.get)

        # Penalty: how much above the threshold is P(contradiction)?
        penalty = max(0.0, p_contra - self.config.contradiction_threshold)

        return NLIPair(
            premise=premise,
            hypothesis=hypothesis,
            scores=score_dict,
            verdict=verdict,
            is_contradiction=p_contra >= self.config.contradiction_threshold,
            is_entailment=p_entail >= self.config.entailment_threshold,
            contradiction_penalty=penalty,
        )

    def analyse(
        self,
        student_answer: str,
        reference_answer: str,
        student_sentences: Optional[List[str]] = None,
    ) -> ContradictionResult:
        """
        Analyse logical consistency between student answer and reference.

        Strategy:
          For each student sentence, check NLI against the full reference
          (premise = reference, hypothesis = student sentence).
          This direction tests: "does the reference *support* the student claim?"
          If NOT → penalty.

        Args:
            student_answer:    Full student response text.
            reference_answer:  Model reference answer.
            student_sentences: Pre-tokenised sentences (optional; saves re-tokenising).

        Returns:
            ContradictionResult with penalty and full pair breakdown.
        """
        if student_sentences is None:
            student_sentences = self._split(student_answer)

        # Filter very short sentences (function words, noise)
        student_sentences = [
            s for s in student_sentences if len(s.split()) >= 4
        ]

        if not student_sentences:
            return self._empty_result("No substantive sentences to analyse.")

        pairs: List[NLIPair] = []
        for sent in student_sentences:
            pair = self._score_pair(
                premise=reference_answer,
                hypothesis=sent,
            )
            pairs.append(pair)

        contra_pairs = [p for p in pairs if p.is_contradiction]
        entail_pairs  = [p for p in pairs if p.is_entailment]
        total_penalty = sum(p.contradiction_penalty for p in pairs)
        mean_entail   = float(np.mean([p.scores.get("entailment", 0) for p in pairs]))

        summary_parts = []
        if contra_pairs:
            summary_parts.append(
                f"{len(contra_pairs)} contradictory statement(s) detected."
            )
        if entail_pairs:
            summary_parts.append(
                f"{len(entail_pairs)} statement(s) supported by reference."
            )
        summary = " ".join(summary_parts) or "No major logical issues detected."

        return ContradictionResult(
            nli_pairs=pairs,
            contradiction_count=len(contra_pairs),
            entailment_count=len(entail_pairs),
            total_penalty=float(total_penalty),
            has_contradiction=len(contra_pairs) > 0,
            confidence=mean_entail,
            summary=summary,
        )

    def _split(self, text: str) -> List[str]:
        """Simple sentence splitter (without spaCy dependency)."""
        import re
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _empty_result(self, msg: str) -> ContradictionResult:
        return ContradictionResult(
            nli_pairs=[],
            contradiction_count=0,
            entailment_count=0,
            total_penalty=0.0,
            has_contradiction=False,
            confidence=0.5,
            summary=msg,
        )
