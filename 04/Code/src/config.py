"""
config.py — Central configuration for the Semantic Evaluation System.

Design decision: All hyperparameters and model choices are centralised here
so the system can be re-tuned without touching business logic.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingConfig:
    """SBERT embedding layer configuration."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    # all-mpnet-base-v2 achieves 63.30 on STSB — better than MiniLM for accuracy.
    # Swap to "sentence-transformers/all-MiniLM-L6-v2" for ~3x faster inference.
    batch_size: int = 32
    max_seq_length: int = 512
    device: str = "cpu"               # Set "cuda" if GPU available
    normalize_embeddings: bool = True  # Required for cosine similarity via dot product


@dataclass
class NLIConfig:
    """Natural Language Inference model configuration."""
    model_name: str = "cross-encoder/nli-deberta-v3-small"
    # DeBERTa-v3-small: strong NLI quality at modest size.
    # Fallback: "facebook/bart-large-mnli" (larger, slower, slightly better)
    contradiction_threshold: float = 0.5   # P(contradiction) above this → penalise
    entailment_threshold: float = 0.5      # P(entailment) above this → reward
    contradiction_penalty: float = 0.25   # Fraction subtracted from final score
    batch_size: int = 8


@dataclass
class ScoringConfig:
    """
    Final scoring formula weights.

    Score = w_sem * semantic_score
           + w_concept * concept_score
           - w_contra * contradiction_penalty
           - w_length * length_penalty
           + w_bonus  * bonus_score

    All weights should sum to approximately 1.0 for interpretability.
    """
    w_semantic:      float = 0.30   # Overall embedding similarity
    w_concept:       float = 0.50   # Per-concept coverage (most important signal)
    w_contradiction: float = 0.15   # Deducted when NLI detects contradiction
    w_length:        float = 0.05   # Penalise answers that are too long or too short

    # Partial marking thresholds
    concept_full_credit:    float = 0.80   # concept similarity ≥ this → full credit
    concept_partial_credit: float = 0.50   # ≥ this → partial credit (scaled linearly)
    concept_no_credit:      float = 0.50   # < this → 0 credit

    # Length normalisation
    min_length_ratio: float = 0.20   # student / reference < this → penalise
    max_length_ratio: float = 3.00   # student / reference > this → penalise

    # Output scale
    max_marks: float = 10.0


@dataclass
class FAISSConfig:
    """FAISS index configuration for fast nearest-neighbour search."""
    index_type: str = "flat"      # "flat" (exact) or "ivf" (approximate, faster)
    nlist: int = 100              # IVF clusters (only used when index_type="ivf")
    nprobe: int = 10              # Cells to visit at query time
    embedding_dim: int = 768      # Must match SBERT model output dim


@dataclass
class SystemConfig:
    """Top-level system configuration — compose all sub-configs here."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)

    # Explainability
    top_k_concepts: int = 5         # Show top-K concepts in explanation
    heatmap_figsize: tuple = (10, 6)

    # Logging
    log_level: str = "INFO"
    verbose: bool = False

    # Fine-tuning (optional)
    finetune_dataset: Optional[str] = None   # "stsb" | "quora" | path to CSV
    finetune_epochs: int = 3
    finetune_lr: float = 2e-5


# Singleton default config — importable everywhere
DEFAULT_CONFIG = SystemConfig()
