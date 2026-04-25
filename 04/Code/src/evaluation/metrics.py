"""
evaluation/metrics.py — Evaluation Framework

Compares model-assigned scores against human-annotated scores using:
  - Pearson correlation (linear relationship)
  - Spearman correlation (rank-order relationship — more robust to outliers)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Quadratic Weighted Kappa (QWK) — standard metric for automated grading

Also provides:
  - Error case analysis (worst-performing examples)
  - Grade-band confusion matrix
  - Score distribution plots
  - Baseline comparison (TF-IDF cosine similarity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    pearson_r:   float
    pearson_p:   float
    spearman_r:  float
    spearman_p:  float
    mae:         float
    rmse:        float
    qwk:         float          # Quadratic Weighted Kappa
    n_samples:   int
    error_cases: pd.DataFrame   # Worst predictions
    report:      str            # Printable summary


def compute_metrics(
    human_scores: List[float],
    model_scores: List[float],
    max_marks: float = 10.0,
    n_error_cases: int = 5,
) -> MetricsResult:
    """
    Compute all evaluation metrics between human and model scores.

    Args:
        human_scores: Ground-truth scores from annotators.
        model_scores: Predicted scores from the system.
        max_marks:    Maximum possible score (for QWK binning).
        n_error_cases: Number of worst examples to show.

    Returns:
        MetricsResult with all statistics.
    """
    h = np.array(human_scores, dtype=float)
    m = np.array(model_scores, dtype=float)

    assert len(h) == len(m), "Score arrays must have the same length."

    # Correlations
    pr, pp = pearsonr(h, m)
    sr, sp = spearmanr(h, m)

    # Error metrics
    mae  = mean_absolute_error(h, m)
    rmse = float(np.sqrt(mean_squared_error(h, m)))

    # QWK: bin scores into integer grades
    n_bins = int(max_marks) + 1
    h_bins = np.clip(np.round(h).astype(int), 0, int(max_marks))
    m_bins = np.clip(np.round(m).astype(int), 0, int(max_marks))
    qwk = cohen_kappa_score(h_bins, m_bins, weights="quadratic")

    # Error cases: |human - model| largest
    errors = np.abs(h - m)
    top_err_idx = np.argsort(errors)[::-1][:n_error_cases]
    error_df = pd.DataFrame({
        "Index":        top_err_idx,
        "Human score":  h[top_err_idx],
        "Model score":  m[top_err_idx],
        "Absolute error": errors[top_err_idx],
        "Direction":    ["Over-scored" if m[i] > h[i] else "Under-scored"
                         for i in top_err_idx],
    })

    report = (
        f"{'='*50}\n"
        f"  Evaluation Report (n={len(h)})\n"
        f"{'='*50}\n"
        f"  Pearson r   : {pr:+.4f}  (p={pp:.3e})\n"
        f"  Spearman r  : {sr:+.4f}  (p={sp:.3e})\n"
        f"  MAE         : {mae:.3f} marks\n"
        f"  RMSE        : {rmse:.3f} marks\n"
        f"  QWK         : {qwk:.4f}\n"
        f"{'='*50}\n"
    )

    return MetricsResult(
        pearson_r=float(pr), pearson_p=float(pp),
        spearman_r=float(sr), spearman_p=float(sp),
        mae=float(mae), rmse=float(rmse),
        qwk=float(qwk), n_samples=len(h),
        error_cases=error_df, report=report,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: TF-IDF cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def tfidf_baseline_scores(
    student_answers: List[str],
    reference_answers: List[str],
    max_marks: float = 10.0,
) -> List[float]:
    """
    Compute TF-IDF + cosine similarity baseline scores.

    This is the "dumb" baseline that the semantic system should outperform.
    Limitations (which our system addresses):
      - Treats paraphrases as dissimilar (lexical mismatch)
      - Rewards keyword stuffing
      - No concept coverage, no NLI
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = student_answers + reference_answers
    vectoriser = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectoriser.fit_transform(corpus)

    n = len(student_answers)
    scores = []
    for i in range(n):
        stu_vec = tfidf_matrix[i]
        ref_vec = tfidf_matrix[n + i]
        sim = float(cosine_similarity(stu_vec, ref_vec)[0, 0])
        scores.append(round(sim * max_marks, 2))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    evaluator,                       # AnswerEvaluator instance
    dataset: List[Dict],             # [{"question", "student_answer", "references", "human_score"}]
    max_marks: float = 10.0,
) -> Tuple[MetricsResult, MetricsResult, pd.DataFrame]:
    """
    Run the full evaluation pipeline on a labelled dataset.

    Returns:
        (semantic_metrics, tfidf_metrics, results_df)
    """
    rows = []
    human_scores = []
    model_scores = []
    baseline_scores = []

    for item in dataset:
        q  = item["question"]
        sa = item["student_answer"]
        refs = item["references"] if isinstance(item["references"], list) else [item["references"]]
        hs = float(item["human_score"])

        # Semantic system
        result = evaluator.evaluate(
            question=q,
            student_answer=sa,
            reference_answers=refs,
        )

        # TF-IDF baseline
        tfidf_score = tfidf_baseline_scores([sa], refs[:1], max_marks)[0]

        human_scores.append(hs)
        model_scores.append(result.final_marks)
        baseline_scores.append(tfidf_score)

        rows.append({
            "question":       q[:60] + "...",
            "student_answer": sa[:80] + "...",
            "human_score":    hs,
            "model_score":    result.final_marks,
            "tfidf_score":    tfidf_score,
            "error_sem":      abs(hs - result.final_marks),
            "error_tfidf":    abs(hs - tfidf_score),
            "concept_coverage": result.concept_result.coverage_ratio,
            "has_contradiction": result.contradiction_result.has_contradiction,
        })

    results_df = pd.DataFrame(rows)
    sem_metrics  = compute_metrics(human_scores, model_scores, max_marks)
    tfidf_metrics = compute_metrics(human_scores, baseline_scores, max_marks)

    print("\n=== Semantic System ===")
    print(sem_metrics.report)
    print("=== TF-IDF Baseline ===")
    print(tfidf_metrics.report)

    return sem_metrics, tfidf_metrics, results_df
