"""
Evidence Package Construction Module

WHAT: Assembles retrieved research chunks, source metadata, search parameters, and model configuration
into a standardized `EvidencePackage` data structure.

WHY: Defines the exact data contract boundary between Phase 4 (RAG Semantic Retrieval) and Phase 5 (LLM Natural Language Generation).

PHASE BOUNDARY:
    Recommendation Engine + Profile
                   ↓
    RAG Semantic Retrieval
                   ↓
    [PHASE 4 ENDS HERE: EvidencePackage Data Contract]
                   ↓
    [PHASE 5 FUTURE: EvidencePackage → LLM → Grounded Explanation]
"""

from typing import Dict, Any, List, Optional
import datetime


class EvidencePackage:
    """
    Encapsulates retrieved scientific evidence chunks and associated metadata into a structured payload.
    """

    def __init__(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "music_wellbeing_research"
    ):
        self.query = query
        self.retrieved_chunks = retrieved_chunks
        self.model_name = model_name
        self.collection_name = collection_name
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        self.sources = self._extract_distinct_sources(retrieved_chunks)

    def _extract_distinct_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique academic paper sources from retrieved chunks.
        """
        seen_docs = set()
        sources = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("document_id") or chunk.get("document_id") or chunk.get("chunk_id", "").split("_chunk_")[0]

            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                source_record = {
                    "document_id": doc_id,
                    "title": meta.get("title") or chunk.get("title", "Unknown Title"),
                    "authors": meta.get("authors_str") or meta.get("authors") or chunk.get("authors", []),
                    "year": meta.get("year") or chunk.get("year", 0),
                    "pmid": meta.get("pmid") or chunk.get("pmid", ""),
                    "doi": meta.get("doi") or chunk.get("doi", ""),
                    "source": meta.get("source") or chunk.get("source", "PubMed"),
                    "topic": meta.get("topic") or chunk.get("topic", "General")
                }
                sources.append(source_record)

        return sources

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert EvidencePackage instance into a serializable JSON-compatible dictionary.
        """
        return {
            "query": self.query,
            "retrieved_chunks": self.retrieved_chunks,
            "sources": self.sources,
            "retrieval_metadata": {
                "timestamp": self.timestamp,
                "embedding_model": self.model_name,
                "collection_name": self.collection_name,
                "total_chunks_retrieved": len(self.retrieved_chunks),
                "total_distinct_sources": len(self.sources)
            }
        }


def build_evidence_package(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    collection_name: str = "music_wellbeing_research"
) -> Dict[str, Any]:
    """
    Helper function to build a structured Evidence Package dictionary.
    """
    package = EvidencePackage(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model_name=model_name,
        collection_name=collection_name
    )
    return package.to_dict()
