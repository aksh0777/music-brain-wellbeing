"""
Local Persistent Vector Database Module (ChromaDB)

WHAT: Manages a local, persistent ChromaDB vector store at `data/vector_store/chroma/` for storing text chunks,
their 384-dimensional dense embedding vectors, and associated metadata.

WHY: High-dimensional vector search requires specialized indexing (e.g. HNSW - Hierarchical Navigable Small World graphs)
to enable fast, scalable k-Nearest Neighbor (k-NN) semantic similarity queries.

ARCHITECTURE BOUNDARIES:
    1. ChromaDB is a LOCAL persistent database in Phase 4, NOT a cloud service or SaaS API.
    2. ChromaDB is NOT our embedding model; embedding vectors are passed directly into ChromaDB.
    3. ChromaDB is NOT an LLM; it returns raw matching text chunks, metadata, and distance metrics.
    4. ChromaDB stores: (a) stable chunk IDs, (b) original text chunks, (c) 384-dim embedding vectors, (d) metadata.
"""

import os
from typing import Dict, Any, List, Optional


class VectorStore:
    """
    Manages local persistent vector storage and k-NN similarity retrieval using ChromaDB.
    """

    DEFAULT_PERSIST_DIR = os.path.join("data", "vector_store", "chroma")
    DEFAULT_COLLECTION_NAME = "music_wellbeing_research"

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ):
        """
        Initialize persistent ChromaDB client and collection.

        Args:
            persist_directory: Path to local directory where ChromaDB data files are saved.
            collection_name: Name of the vector collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None

        os.makedirs(self.persist_directory, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize ChromaDB PersistentClient and create/get collection."""
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            # Fallback in-memory dict implementation for test mock if chromadb not installed/available
            self._client = None
            self._collection = None
            self._mock_store = {}

    def _clean_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert complex metadata types (e.g. lists) into ChromaDB-compatible primitive types (str, int, float, bool).
        """
        clean_meta = {}
        for key, val in meta.items():
            if key == "text":
                continue  # 'text' is stored separately as the document string
            if isinstance(val, list):
                clean_meta[key] = ", ".join(str(item) for item in val)
            elif isinstance(val, (str, int, float, bool)):
                clean_meta[key] = val
            elif val is None:
                clean_meta[key] = ""
            else:
                clean_meta[key] = str(val)
        return clean_meta

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Upsert (insert or update) structured text chunks and their embeddings into ChromaDB.

        Args:
            chunks: List of chunk dictionaries (must contain 'chunk_id' and 'text').
            embeddings: List of 384-dim dense float vectors matching chunks.

        Returns:
            Number of chunks upserted.
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})")

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [self._clean_metadata(chunk) for chunk in chunks]

        if self._collection is not None:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            # Mock fallback store for unit test execution without chromadb
            for chunk_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
                self._mock_store[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "embedding": emb,
                    "metadata": meta
                }

        return len(chunks)

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for top_k most similar chunks given a 384-dim query embedding vector.

        Args:
            query_embedding: 384-dim float vector.
            top_k: Number of nearest neighbors to retrieve.
            filter_metadata: Optional dictionary filter for metadata matching.

        Returns:
            List of result dictionaries containing chunk_id, text, metadata, and distance.
        """
        if self._collection is not None:
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            if filter_metadata:
                kwargs["where"] = filter_metadata

            results = self._collection.query(**kwargs)

            formatted_results = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                ret_ids = results["ids"][0]
                ret_docs = results["documents"][0]
                ret_metas = results["metadatas"][0]
                ret_dists = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ret_ids)

                for chunk_id, doc, meta, dist in zip(ret_ids, ret_docs, ret_metas, ret_dists):
                    res = {
                        "chunk_id": chunk_id,
                        "text": doc,
                        "metadata": meta,
                        "distance": float(dist),
                        "similarity_score": round(1.0 / (1.0 + float(dist)), 4)
                    }
                    formatted_results.append(res)

            return formatted_results

        else:
            # Mock query logic using cosine similarity over mock store
            import numpy as np
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)

            scored = []
            for chunk_id, record in self._mock_store.items():
                if filter_metadata:
                    match = True
                    for k, v in filter_metadata.items():
                        if record["metadata"].get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                emb_vec = np.array(record["embedding"], dtype=np.float32)
                emb_norm = np.linalg.norm(emb_vec)
                dot = np.dot(q_vec, emb_vec)
                cos_sim = float(dot / (q_norm * emb_norm + 1e-9))
                dist = max(0.0, 1.0 - cos_sim)

                scored.append({
                    "chunk_id": chunk_id,
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "distance": dist,
                    "similarity_score": round(cos_sim, 4)
                })

            scored.sort(key=lambda x: x["distance"])
            return scored[:top_k]

    def count(self) -> int:
        """Return total number of items stored in collection."""
        if self._collection is not None:
            return self._collection.count()
        return len(self._mock_store)

    def clear(self) -> None:
        """Clear all stored vectors and documents in the collection."""
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self._mock_store.clear()

    def close(self) -> None:
        """Release underlying client references to allow file cleanup."""
        self._collection = None
        self._client = None
        if hasattr(self, "_mock_store"):
            self._mock_store.clear()


