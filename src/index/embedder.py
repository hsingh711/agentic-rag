"""Thin wrapper around sentence-transformers that returns normalised float32 vectors."""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Encode texts into unit-length vectors using a sentence-transformers model."""

    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True):
        """Load the model.

        Args:
            model_name: Hugging Face / sentence-transformers model name.
            batch_size: Number of texts encoded per forward pass.
            normalize: If True, vectors are scaled to length 1 (needed for cosine/dot search).
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        logger.info("Loaded embedder %s (dimension %d)", model_name, self.dimension)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Embedder":
        """Build an Embedder from the ``embedding`` section of the config."""
        embedding_config = config["embedding"]
        return cls(
            model_name=embedding_config["model_name"],
            batch_size=embedding_config["batch_size"],
            normalize=embedding_config["normalize"],
        )

    @property
    def dimension(self) -> int:
        """Length of each embedding vector."""
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode many texts in batches.

        Returns:
            Array of shape (len(texts), dimension), dtype float32.
        """
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1-D vector."""
        return self.encode([query])[0]
