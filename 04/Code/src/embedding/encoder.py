"""
embedding/encoder.py — Semantic Embedding Layer

Responsibilities:
  - Load and cache SBERT models
  - Batch-encode sentences efficiently
  - Expose a FAISS index for fast ANN search
  - Support optional fine-tuning pipeline

Design decisions:
  - We use all-mpnet-base-v2 (768-dim) over MiniLM (384-dim) because the extra
    capacity captures subtle paraphrasing better on STS benchmarks.
  - Embeddings are L2-normalised so cosine similarity == dot product — this lets
    us swap to FAISS inner-product index for O(log N) search at scale.
  - The encoder is a singleton (lazy-loaded) to avoid re-loading weights on
    every evaluation call in the Streamlit app.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.config import EmbeddingConfig, FAISSConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class SemanticEncoder:
    """
    Thin wrapper around SentenceTransformer with:
      - LRU-style in-memory embedding cache (keyed by SHA-256 of text)
      - Batch encoding with progress bar
      - FAISS index for fast retrieval
      - Fine-tuning entrypoint
    """

    def __init__(
        self,
        config: EmbeddingConfig = DEFAULT_CONFIG.embedding,
        faiss_config: FAISSConfig = DEFAULT_CONFIG.faiss,
    ):
        self.config = config
        self.faiss_config = faiss_config
        self._model: Optional[SentenceTransformer] = None
        self._cache: dict[str, np.ndarray] = {}
        self._faiss_index = None
        self._faiss_texts: List[str] = []

    # ------------------------------------------------------------------ #
    #  Model Loading                                                       #
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load model on first access."""
        if self._model is None:
            logger.info(f"Loading SBERT model: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            self._model.max_seq_length = self.config.max_seq_length
            logger.info("Model loaded successfully.")
        return self._model

    # ------------------------------------------------------------------ #
    #  Encoding                                                            #
    # ------------------------------------------------------------------ #

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode one or more texts into L2-normalised embedding vectors.

        Args:
            texts:         Single string or list of strings.
            show_progress: Show tqdm progress bar for large batches.
            use_cache:     Skip re-encoding texts seen before.

        Returns:
            np.ndarray of shape (N, embedding_dim), float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Split into cached / uncached
        results: dict[int, np.ndarray] = {}
        to_encode: List[tuple[int, str]] = []

        for idx, text in enumerate(texts):
            key = self._cache_key(text)
            if use_cache and key in self._cache:
                results[idx] = self._cache[key]
            else:
                to_encode.append((idx, text))

        if to_encode:
            raw_texts = [t for _, t in to_encode]
            embeddings = self.model.encode(
                raw_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            for (orig_idx, text), emb in zip(to_encode, embeddings):
                key = self._cache_key(text)
                self._cache[key] = emb
                results[orig_idx] = emb

        return np.vstack([results[i] for i in range(len(texts))])

    def similarity_matrix(
        self,
        texts_a: List[str],
        texts_b: List[str],
    ) -> np.ndarray:
        """
        Compute an (|A| x |B|) cosine similarity matrix.

        Because embeddings are L2-normalised, this is just emb_a @ emb_b.T.

        Returns:
            np.ndarray of shape (len(texts_a), len(texts_b)), values in [-1, 1].
        """
        emb_a = self.encode(texts_a)
        emb_b = self.encode(texts_b)
        return emb_a @ emb_b.T  # cosine via dot product after normalisation

    def pairwise_similarity(self, text_a: str, text_b: str) -> float:
        """Scalar cosine similarity between two texts."""
        return float(self.similarity_matrix([text_a], [text_b])[0, 0])

    # ------------------------------------------------------------------ #
    #  FAISS Index                                                         #
    # ------------------------------------------------------------------ #

    def build_faiss_index(self, texts: List[str]) -> None:
        """
        Build a FAISS flat (exact) or IVF (approximate) index over `texts`.

        Call this once with all reference answers / concepts so that student
        answers can be retrieved in O(log N) rather than O(N).
        """
        if not FAISS_AVAILABLE:
            logger.warning("faiss not installed — skipping index build.")
            return

        embeddings = self.encode(texts, show_progress=True)
        dim = embeddings.shape[1]

        if self.faiss_config.index_type == "ivf":
            quantiser = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantiser, dim, self.faiss_config.nlist)
            index.nprobe = self.faiss_config.nprobe
            index.train(embeddings)
        else:
            # Exact inner-product search (works because embeddings are normalised)
            index = faiss.IndexFlatIP(dim)

        index.add(embeddings)
        self._faiss_index = index
        self._faiss_texts = list(texts)
        logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}.")

    def faiss_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[tuple[str, float]]:
        """
        Find the top-k most similar texts to `query` using FAISS.

        Returns:
            List of (text, similarity_score) tuples, sorted descending.
        """
        if self._faiss_index is None:
            raise RuntimeError("Call build_faiss_index() before faiss_search().")
        q_emb = self.encode([query])
        distances, indices = self._faiss_index.search(q_emb, top_k)
        return [
            (self._faiss_texts[i], float(d))
            for i, d in zip(indices[0], distances[0])
            if i >= 0
        ]

    # ------------------------------------------------------------------ #
    #  Fine-tuning                                                         #
    # ------------------------------------------------------------------ #

    def finetune(
        self,
        pairs: List[tuple[str, str, float]],
        output_path: str = "models/finetuned-sbert",
        epochs: int = 3,
        lr: float = 2e-5,
    ) -> None:
        """
        Fine-tune SBERT on semantic similarity pairs using CosineSimilarityLoss.

        Args:
            pairs:       List of (sentence_a, sentence_b, similarity_score_0_to_1).
            output_path: Directory to save the fine-tuned model.
            epochs:      Number of training epochs.
            lr:          Learning rate.

        The loss function: L = (cos(emb_a, emb_b) - label)^2
        This directly optimises the embedding space for cosine similarity.
        """
        train_examples = [
            InputExample(texts=[a, b], label=float(score))
            for a, b, score in pairs
        ]
        loader = DataLoader(train_examples, shuffle=True, batch_size=self.config.batch_size)
        loss_fn = losses.CosineSimilarityLoss(self.model)

        logger.info(f"Fine-tuning for {epochs} epochs on {len(pairs)} pairs.")
        self.model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=epochs,
            optimizer_params={"lr": lr},
            output_path=output_path,
            show_progress_bar=True,
        )
        logger.info(f"Fine-tuned model saved to {output_path}.")


# ---------------------------------------------------------------------------
# Module-level singleton — import and reuse across the whole pipeline
# ---------------------------------------------------------------------------
_encoder_instance: Optional[SemanticEncoder] = None


def get_encoder(config: EmbeddingConfig = DEFAULT_CONFIG.embedding) -> SemanticEncoder:
    """Return the module-level singleton encoder (lazy-initialised)."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = SemanticEncoder(config)
    return _encoder_instance
