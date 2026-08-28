"""
Research Corpus Ingestion Pipeline Module

WHAT: Orchestrates the end-to-end ingestion workflow:
      JSONL Research Records → Validation → Chunking → Embedding Generation → ChromaDB Upsert & Local Persistence.

WHY: Automates corpus loading in a repeatable, idempotent pipeline. Repeated ingestion runs update
existing records by stable chunk ID rather than creating duplicate entries.

PIPELINE:
    Research JSONL File
            ↓
    Load & Validate Schema
            ↓
    Document Chunker
            ↓
    SentenceTransformer Embedding Generator
            ↓
    ChromaDB Local Vector Store (Upsert)
"""

import json
import os
from typing import Dict, Any, List, Optional
from .chunker import DocumentChunker
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class IngestionPipeline:
    """
    Idempotent ingestion pipeline for loading, chunking, embedding, and persisting scientific research records.
    """

    def __init__(
        self,
        chunker: Optional[DocumentChunker] = None,
        embedder: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.chunker = chunker if chunker is not None else DocumentChunker()
        self.embedder = embedder if embedder is not None else EmbeddingModel()
        self.vector_store = vector_store if vector_store is not None else VectorStore()

    def load_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        Load structured research documents from a JSONL file.

        Args:
            jsonl_path: Absolute or relative path to .jsonl file.

        Returns:
            List of document dictionaries.
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Research JSONL file not found at path: {jsonl_path}")

        documents = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    doc = json.loads(line_str)
                    if self.chunker.validate_document(doc):
                        documents.append(doc)
                    else:
                        print(f"Warning: Document at line {line_idx} failed schema validation.")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_idx}: {e}")

        return documents

    def run(self, jsonl_path: str) -> Dict[str, Any]:
        """
        Execute full ingestion pipeline.

        Args:
            jsonl_path: Path to research corpus JSONL file.

        Returns:
            Summary dictionary of ingestion metrics (documents_loaded, chunks_created, vectors_stored).
        """
        documents = self.load_jsonl(jsonl_path)
        if not documents:
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "vectors_stored": 0,
                "status": "empty_corpus"
            }

        # Step 1: Chunk documents
        all_chunks = self.chunker.chunk_documents(documents)

        # Step 2: Generate embeddings
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.encode_batch(chunk_texts)

        # Step 3: Upsert into ChromaDB
        count_upserted = self.vector_store.upsert_chunks(all_chunks, embeddings)

        return {
            "documents_loaded": len(documents),
            "chunks_created": len(all_chunks),
            "vectors_stored": count_upserted,
            "status": "success",
            "total_collection_count": self.vector_store.count()
        }
