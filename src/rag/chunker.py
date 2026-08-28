"""
Research Document Chunking Module

WHAT: Transforms raw research JSONL documents into structured, metadata-rich text chunks with stable chunk IDs.

WHY: Small, focused text chunks optimize embedding vector representation quality and semantic retrieval precision.
Whole research papers or long abstracts become diluted when embedded as single large vectors.

TRANSFORMATION HIERARCHY:
    Research Document (JSONL Record)
                 ↓
    Chunker (Splits into 150-300 word chunks with overlap)
                 ↓
    Structured Chunks with Stable Chunk IDs & Metadata
"""

import math
from typing import Dict, Any, List, Optional


class DocumentChunker:
    """
    Splits research documents into semantic chunks while preserving full document provenance and metadata.
    """

    def __init__(self, max_chunk_words: int = 200, overlap_words: int = 30):
        """
        Initialize the DocumentChunker.

        Args:
            max_chunk_words: Maximum number of words per text chunk.
            overlap_words: Number of overlapping words between consecutive chunks.
        """
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words

    def validate_document(self, doc: Dict[str, Any]) -> bool:
        """
        Validate that a document contains required schema fields.

        Required fields: document_id, title, text
        """
        required_fields = ["document_id", "title", "text"]
        for field in required_fields:
            if field not in doc or not doc[field]:
                return False
        return True

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single research document into one or more structured chunk dictionaries.

        Args:
            doc: Research document dictionary from JSONL record.

        Returns:
            List of chunk dictionaries with stable chunk_id and preserved metadata.
        """
        if not self.validate_document(doc):
            raise ValueError(f"Invalid document schema: missing required fields in {doc.get('document_id', 'UNKNOWN')}")

        doc_id = str(doc["document_id"])
        title = doc.get("title", "")
        text = doc.get("text", "").strip()

        words = text.split()

        # If text is short (e.g. standard abstract <= max_chunk_words), keep as single chunk
        if len(words) <= self.max_chunk_words:
            chunk_texts = [text]
        else:
            chunk_texts = []
            start = 0
            step = max(1, self.max_chunk_words - self.overlap_words)
            while start < len(words):
                end = min(start + self.max_chunk_words, len(words))
                chunk_words = words[start:end]
                chunk_str = " ".join(chunk_words)
                chunk_texts.append(chunk_str)
                if end == len(words):
                    break
                start += step


        chunks = []
        total_chunks = len(chunk_texts)

        # Standardize authors format
        authors = doc.get("authors", [])
        authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)

        for idx, chunk_text in enumerate(chunk_texts):
            # Stable chunk ID format: {document_id}_chunk_{idx}
            chunk_id = f"{doc_id}_chunk_{idx}"

            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": title,
                "authors": authors,
                "authors_str": authors_str,
                "year": doc.get("year", 0),
                "pmid": doc.get("pmid", ""),
                "doi": doc.get("doi", ""),
                "source": doc.get("source", "PubMed"),
                "topic": doc.get("topic", "General"),
                "data_type": doc.get("data_type", "real_research"),
                "content_scope": doc.get("content_scope", "abstract"),
                "text": chunk_text,
                "chunk_index": idx,
                "total_chunks": total_chunks
            }
            chunks.append(chunk_record)

        return chunks

    def chunk_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of research documents into a flattened list of chunk dictionaries.
        """
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
