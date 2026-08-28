"""
Unit Tests for Phase 4 Research-Grounded RAG Layer

WHAT: Comprehensive test suite validating research JSONL loading, schema validation, document chunking,
stable chunk IDs, 384-dim embedding generation, vector insertion, local persistence, semantic retrieval,
metadata preservation, top-k filtering, empty corpus handling, evidence package structure, and duplicate ingestion prevention.

WHY: Ensures robust offline verification without external cloud API dependencies.
"""

import unittest
import tempfile
import json
import os
import shutil
from src.rag.chunker import DocumentChunker
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.ingest import IngestionPipeline
from src.rag.retriever import ResearchRetriever
from src.rag.adapter import RecommendationQueryAdapter
from src.rag.evidence import EvidencePackage, build_evidence_package


class TestRAGLayer(unittest.TestCase):

    def setUp(self):
        """Set up temporary directory and sample research document fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_dir = os.path.join(self.temp_dir, "chroma")

        self.sample_doc_1 = {
            "document_id": "pub_lu_2021_34365216",
            "title": "Effects of music therapy on anxiety: A meta-analysis of randomized controlled trials",
            "authors": ["Guangli Lu", "Ruiying Jia", "Dandan Liang"],
            "year": 2021,
            "pmid": "34365216",
            "doi": "10.1016/j.psychres.2021.114137",
            "source": "PubMed",
            "topic": "Music Therapy and Anxiety",
            "data_type": "real_research",
            "content_scope": "abstract",
            "text": "This meta-analysis evaluated the efficacy of music therapy in reducing anxiety across randomized controlled trials."
        }

        self.sample_doc_2 = {
            "document_id": "pub_vandentol_2022_35714120",
            "title": "Music listening and stress recovery in healthy individuals",
            "authors": ["Kirsten van den Tol", "Marieke van der Zalm"],
            "year": 2022,
            "pmid": "35714120",
            "doi": "10.1371/journal.pone.0270031",
            "source": "PubMed",
            "topic": "Music Listening and Stress Recovery",
            "data_type": "real_research",
            "content_scope": "abstract",
            "text": "A meta-analysis of laboratory studies found a non-significant cumulative effect of music listening on acute stress recovery."
        }

        # Write sample JSONL
        self.jsonl_path = os.path.join(self.temp_dir, "test_research.jsonl")
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.sample_doc_1) + "\n")
            f.write(json.dumps(self.sample_doc_2) + "\n")

    def tearDown(self):
        """Clean up temporary files and directories."""
        import gc
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)


    def test_1_research_loading(self):
        """Test loading documents from JSONL file."""
        pipeline = IngestionPipeline()
        docs = pipeline.load_jsonl(self.jsonl_path)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["document_id"], "pub_lu_2021_34365216")
        self.assertEqual(docs[1]["document_id"], "pub_vandentol_2022_35714120")

    def test_2_metadata_validation(self):
        """Test schema validator for valid and invalid research documents."""
        chunker = DocumentChunker()
        valid_doc = self.sample_doc_1
        invalid_doc = {"title": "Missing ID and text"}

        self.assertTrue(chunker.validate_document(valid_doc))
        self.assertFalse(chunker.validate_document(invalid_doc))

    def test_3_chunking(self):
        """Test document chunking logic."""
        chunker = DocumentChunker(max_chunk_words=10)
        chunks = chunker.chunk_document(self.sample_doc_1)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["document_id"], "pub_lu_2021_34365216")
        self.assertEqual(chunks[0]["pmid"], "34365216")

    def test_4_stable_chunk_ids(self):
        """Test that chunk IDs are deterministic and stable across calls."""
        chunker = DocumentChunker()
        chunks_1 = chunker.chunk_document(self.sample_doc_1)
        chunks_2 = chunker.chunk_document(self.sample_doc_1)
        self.assertEqual(chunks_1[0]["chunk_id"], "pub_lu_2021_34365216_chunk_0")
        self.assertEqual(chunks_1[0]["chunk_id"], chunks_2[0]["chunk_id"])

    def test_5_embedding_generation(self):
        """Test embedding generation produces 384-dimensional numerical vectors."""
        embedder = EmbeddingModel(use_fallback=True)
        vector = embedder.encode("Music therapy reduces anxiety")
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 384)
        self.assertTrue(all(isinstance(val, float) for val in vector))

    def test_6_vector_insertion(self):
        """Test vector insertion into VectorStore."""
        store = VectorStore(persist_directory=self.chroma_dir, collection_name="test_col_1")
        chunker = DocumentChunker()
        embedder = EmbeddingModel(use_fallback=True)

        chunks = chunker.chunk_document(self.sample_doc_1)
        embeddings = embedder.encode_batch([c["text"] for c in chunks])

        count = store.upsert_chunks(chunks, embeddings)
        self.assertEqual(count, len(chunks))
        self.assertEqual(store.count(), len(chunks))

    def test_7_persistence(self):
        """Test that vector store directory is persisted locally."""
        store = VectorStore(persist_directory=self.chroma_dir, collection_name="test_col_persist")
        store.upsert_chunks(
            [{"chunk_id": "c1", "text": "sample text", "document_id": "d1"}],
            [[0.1] * 384]
        )
        self.assertTrue(os.path.exists(self.chroma_dir))

    def test_8_semantic_retrieval(self):
        """Test semantic search returns relevant matching chunk."""
        store = VectorStore(persist_directory=self.chroma_dir, collection_name="test_col_ret")
        chunker = DocumentChunker()
        embedder = EmbeddingModel(use_fallback=True)

        chunks = chunker.chunk_documents([self.sample_doc_1, self.sample_doc_2])
        embeddings = embedder.encode_batch([c["text"] for c in chunks])
        store.upsert_chunks(chunks, embeddings)

        retriever = ResearchRetriever(embedder=embedder, vector_store=store)
        results = retriever.retrieve("anxiety reduction in RCTs", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("text", results[0])

    def test_9_metadata_preservation(self):
        """Test that title, authors, year, PMID, DOI are preserved in retrieval output."""
        store = VectorStore(persist_directory=self.chroma_dir, collection_name="test_col_meta")
        chunker = DocumentChunker()
        embedder = EmbeddingModel(use_fallback=True)

        chunks = chunker.chunk_document(self.sample_doc_1)
        embeddings = embedder.encode_batch([c["text"] for c in chunks])
        store.upsert_chunks(chunks, embeddings)

        retriever = ResearchRetriever(embedder=embedder, vector_store=store)
        results = retriever.retrieve("music therapy anxiety", top_k=1)
        meta = results[0]["metadata"]
        self.assertEqual(meta["pmid"], "34365216")
        self.assertEqual(meta["doi"], "10.1016/j.psychres.2021.114137")
        self.assertEqual(meta["year"], 2021)

    def test_10_top_k_retrieval(self):
        """Test that top_k parameter restricts the number of retrieved items."""
        store = VectorStore(persist_directory=self.chroma_dir, collection_name="test_col_topk")
        chunker = DocumentChunker()
        embedder = EmbeddingModel(use_fallback=True)

        chunks = chunker.chunk_documents([self.sample_doc_1, self.sample_doc_2])
        embeddings = embedder.encode_batch([c["text"] for c in chunks])
        store.upsert_chunks(chunks, embeddings)

        retriever = ResearchRetriever(embedder=embedder, vector_store=store)
        results_k1 = retriever.retrieve("stress music", top_k=1)
        results_k2 = retriever.retrieve("stress music", top_k=2)
        self.assertEqual(len(results_k1), 1)
        self.assertEqual(len(results_k2), 2)

    def test_11_empty_corpus_handling(self):
        """Test retriever when vector store is empty."""
        store = VectorStore(persist_directory=self.chroma_dir, collection_name="test_col_empty")
        retriever = ResearchRetriever(vector_store=store)
        results = retriever.retrieve("any query", top_k=3)
        self.assertEqual(results, [])

    def test_12_evidence_package_structure(self):
        """Test EvidencePackage data contract schema."""
        retrieved_chunks = [
            {
                "chunk_id": "pub_lu_2021_34365216_chunk_0",
                "text": "Sample text",
                "metadata": {
                    "document_id": "pub_lu_2021_34365216",
                    "title": "Effects of music therapy",
                    "authors_str": "Guangli Lu, Ruiying Jia",
                    "year": 2021,
                    "pmid": "34365216",
                    "doi": "10.1016/j.psychres.2021.114137",
                    "source": "PubMed",
                    "topic": "Music Therapy and Anxiety"
                },
                "distance": 0.15,
                "similarity_score": 0.8696
            }
        ]
        pkg = build_evidence_package(query="anxiety therapy", retrieved_chunks=retrieved_chunks)

        self.assertIn("query", pkg)
        self.assertIn("retrieved_chunks", pkg)
        self.assertIn("sources", pkg)
        self.assertIn("retrieval_metadata", pkg)
        self.assertEqual(pkg["query"], "anxiety therapy")
        self.assertEqual(len(pkg["sources"]), 1)
        self.assertEqual(pkg["sources"][0]["pmid"], "34365216")

    def test_13_duplicate_ingestion_prevention(self):
        """Test that running ingestion twice does not create duplicate chunks in ChromaDB."""
        store = VectorStore(persist_directory=self.chroma_dir, collection_name="test_col_dup")
        embedder = EmbeddingModel(use_fallback=True)
        chunker = DocumentChunker()
        pipeline = IngestionPipeline(chunker=chunker, embedder=embedder, vector_store=store)

        res1 = pipeline.run(self.jsonl_path)
        count_first_run = store.count()

        res2 = pipeline.run(self.jsonl_path)
        count_second_run = store.count()

        self.assertEqual(count_first_run, count_second_run)


if __name__ == "__main__":
    unittest.main()
