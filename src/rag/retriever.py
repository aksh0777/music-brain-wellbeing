"""
Semantic Research Retriever Module

WHAT: Executes semantic search over indexed research chunks by converting research queries into dense embedding vectors
and querying the local ChromaDB vector store.

WHY: Retrieves scientifically relevant literature chunks matching user acoustic profiles or research topics,
enabling evidence-grounded downstream explanations.

RETRIEVAL PIPELINE:
    Natural Language / Adapted Research Query
                     ↓
    Embedding Model (Encodes Query to 384-dim Vector)
                     ↓
    ChromaDB k-NN Vector Search (Cosine Distance)
                     ↓
    Top-K Structured Chunks + Distance Metrics + Source Metadata
"""

from typing import Dict, Any, List, Optional
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class ResearchRetriever:
    """
    Executes vector-based semantic retrieval against the local ChromaDB research corpus.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.embedder = embedder if embedder is not None else EmbeddingModel()
        self.vector_store = vector_store if vector_store is not None else VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic retrieval for a natural language or adapted research query.

        Args:
            query: Input search string.
            top_k: Number of top relevant chunks to retrieve.
            filter_metadata: Optional metadata filter dict.

        Returns:
            List of retrieved result dictionaries containing chunk details, source metadata, and distance.
        """
        if not query or not query.strip():
            return []

        # Step 1: Embed query text into 384-dim vector
        query_vector = self.embedder.encode(query.strip())

        # Step 2: Query ChromaDB for nearest neighbor chunks
        results = self.vector_store.query_similar(
            query_embedding=query_vector,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

        return results
