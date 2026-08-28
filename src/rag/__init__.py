"""
Research-Grounded Retrieval-Augmented Generation (RAG) Layer Package

WHAT: Core RAG module powering semantic research retrieval over verified PubMed literature.
"""

from .chunker import DocumentChunker
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .ingest import IngestionPipeline
from .retriever import ResearchRetriever
from .adapter import RecommendationQueryAdapter
from .evidence import EvidencePackage, build_evidence_package

__all__ = [
    "DocumentChunker",
    "EmbeddingModel",
    "VectorStore",
    "IngestionPipeline",
    "ResearchRetriever",
    "RecommendationQueryAdapter",
    "EvidencePackage",
    "build_evidence_package",
]
