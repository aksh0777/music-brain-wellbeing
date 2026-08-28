# Phase 4 Technical Foundation: Research-Grounded RAG Layer

## 1. Phase Objective
The objective of Phase 4 is to build the **Research Retrieval Foundation** for the Music Brain Wellbeing Intelligence System. The RAG layer indexes verified academic research from PubMed regarding music therapy, stress reduction, autonomic arousal, and emotion regulation into a local persistent vector database (ChromaDB) using `sentence-transformers/all-MiniLM-L6-v2`. When a user receives a music recommendation, an adapter converts their acoustic profile into a research query, retrieves the top-K relevant scientific chunks, and packages them into a structured `EvidencePackage`.

$$\text{Spotify Listening} \rightarrow \text{Music Profile} \rightarrow \text{Recommendation Engine} \rightarrow \text{Research Query} \rightarrow \text{Embeddings} \rightarrow \text{ChromaDB} \rightarrow \text{Semantic Retrieval} \rightarrow \text{Evidence Package}$$

> **System Boundary:** Phase 4 strictly ends at **Research Retrieval $\rightarrow$ Evidence Package**. Phase 5 will consume the `EvidencePackage` to generate grounded LLM natural language explanations.

---

## 2. First-Principles RAG Explanation

### What is RAG?
Retrieval-Augmented Generation (RAG) is an architectural pattern that combines an information retrieval system (vector database) with a generative model (LLM). Instead of relying solely on the parametric memory of an LLM (which can hallucinate or contain outdated knowledge), RAG retrieves relevant factual document chunks from an authoritative corpus and passes them as explicit context to the model.

### Why Do We Need Retrieval?
Large Language Models (LLMs) do not possess real-time or verified domain databases out of the box. Simply asking an LLM about music psychology risks generating plausible-sounding but completely fabricated citations, non-existent DOIs, or exaggerated clinical claims. Retrieval grounds the AI in verified, peer-reviewed scientific literature.

### What is an Embedding?
An embedding is a dense numerical vector representation of text in a continuous vector space (e.g., $R^{384}$). Words and sentences with similar semantic meanings are mapped to nearby geometric coordinates within this space.

### How Does Text Become an Embedding Vector?
Text is processed through a pretrained Transformer neural network (such as `all-MiniLM-L6-v2`). The model tokenizes the input text, passes it through self-attention layers, and aggregates hidden states (using mean pooling) to produce a fixed-length vector of floating-point numbers.

### What Does the Vector Represent?
The vector represents the statistical and semantic geometry of the text learned during contrastive pretraining. It captures conceptual relationships (e.g., the proximity between "anxiety reduction" and "stress recovery"), but individual vector dimensions do not correspond to manual human labels.

### What Does ChromaDB Store?
ChromaDB stores:
1. Unique Chunk Identifiers (`ids`)
2. Original Raw Text Chunks (`documents`)
3. 384-dimensional Embedding Vectors (`embeddings`)
4. Associated Provenance Metadata (`metadatas`: title, authors, year, PMID, DOI, topic)

### How Does Semantic Similarity Work?
Semantic similarity is computed using distance metrics between the query embedding vector $\vec{q}$ and document chunk vectors $\vec{d}$. ChromaDB calculates cosine distance:

$$\text{Cosine Distance} = 1 - \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$

Chunks with smaller cosine distance (or higher cosine similarity) are returned as top-K nearest neighbors.

### Why Do We Store Original Text Together with Embeddings?
Embeddings are mathematical vectors optimized for spatial indexing, not human or LLM reading. We store the original text alongside the embedding so that once top-K vectors are retrieved, their exact source text can be injected into the LLM prompt.

### Why Retrieval Happens Before Generation?
Retrieval must precede generation because the generative model requires the retrieved factual context to constrain its output, eliminating hallucination and guaranteeing attribution to specific scientific papers.

### Why RAG is Better Than Asking an LLM Directly
1. **Zero Hallucination of Citations:** Every source citation originates from a verified PubMed ID in ChromaDB.
2. **Auditability & Data Provenance:** Every claim in the evidence package traces back to exact raw JSONL records.
3. **Domain Customization:** We can update or expand the research corpus without expensive model retraining or fine-tuning.

---

## 3. Architecture & Data Flow

```
Spotify Listening Data
        ↓
Music Intelligence (Phase 1)
        ↓
User Music Profile (Phase 1)
        ↓
Recommendation Engine (Phase 2 & 3)
        ↓
Recommended Track + Acoustic Profile
        ↓
RecommendationQueryAdapter (Phase 4)
        ↓
Research Query String
        ↓
EmbeddingModel (sentence-transformers/all-MiniLM-L6-v2)
        ↓
ChromaDB Local VectorStore (HNSW Cosine Index)
        ↓
ResearchRetriever (Semantic k-NN Search)
        ↓
Top-K Structured Chunks + Distance Metrics
        ↓
EvidencePackage (Data Contract for Phase 5 LLM)
```

---

## 4. Research Corpus Provenance

The research corpus is stored in `data/raw/research/music_wellbeing_research.jsonl` and contains 10 verified PubMed sources:

| Document ID | Title | Authors | Year | PMID | DOI | Topic |
|---|---|---|---|---|---|---|
| `pub_lu_2021_34365216` | Effects of music therapy on anxiety: A meta-analysis | Lu et al. | 2021 | 34365216 | 10.1016/j.psychres.2021.114137 | Music Therapy & Anxiety |
| `pub_dewitte_2025_40547443` | Music therapy for the treatment of anxiety: systematic review | de Witte et al. | 2025 | 40547443 | 10.1016/j.eclinm.2025.103293 | Music Therapy & Anxiety |
| `pub_dewitte_2022_33176590` | Music therapy for stress reduction: systematic review | de Witte et al. | 2022 | 33176590 | 10.1080/17437199.2020.1846580 | Stress Reduction |
| `pub_dewitte_2020_31167611` | Effects of music interventions on stress-related outcomes | de Witte et al. | 2020 | 31167611 | 10.1080/17437199.2019.1627897 | Stress Outcomes |
| `pub_vandentol_2022_35714120` | Music listening and stress recovery in healthy individuals | van den Tol et al. | 2022 | 35714120 | 10.1371/journal.pone.0270031 | Stress Recovery (Mixed) |
| `pub_ijpsycho_2024_38458383` | Effects of music and auditory stimulation on autonomic arousal | IJ Psychophysiol | 2024 | 38458383 | 10.1016/j.ijpsycho.2024.108420 | Autonomic Arousal |
| `pub_chong_2024_39336008` | Scoping Review on the Use of Music for Emotion Regulation | Chong et al. | 2024 | 39336008 | 10.3390/bs14090793 | Emotion Regulation |
| `pub_chamorro_2007_17456267` | Personality and music: can traits explain music use? | Chamorro-Premuzic et al.| 2007 | 17456267 | 10.1348/000712606X111177 | Individual Differences |
| `pub_morawetz_2024_38759742` | Neural underpinnings of individual differences in emotion regulation | Morawetz & Basten | 2024 | 38759742 | 10.1016/j.neubiorev.2024.105727 | Neural Underpinnings |
| `pub_chu_2026_42339210` | Music's context-dependent influence on oxytocin | Chu & Tsai | 2026 | 42339210 | 10.3390/fcogn.2025.1678665 | Context-Dependent Effects |

---

## 5. Module Overview (`src/rag/`)

- [`chunker.py`](file:///c:/Users/aksha/OneDrive/Desktop/music-brain-wellbeing/src/rag/chunker.py): `DocumentChunker` splits documents into structured chunks with stable IDs (`{doc_id}_chunk_{i}`) and preserves metadata.
- [`embeddings.py`](file:///c:/Users/aksha/OneDrive/Desktop/music-brain-wellbeing/src/rag/embeddings.py): `EmbeddingModel` wraps `sentence-transformers/all-MiniLM-L6-v2` generating 384-dim dense float vectors.
- [`vector_store.py`](file:///c:/Users/aksha/OneDrive/Desktop/music-brain-wellbeing/src/rag/vector_store.py): `VectorStore` manages local persistent ChromaDB at `data/vector_store/chroma/`.
- [`ingest.py`](file:///c:/Users/aksha/OneDrive/Desktop/music-brain-wellbeing/src/rag/ingest.py): `IngestionPipeline` orchestrates repeatable, idempotent corpus loading.
- [`retriever.py`](file:///c:/Users/aksha/OneDrive/Desktop/music-brain-wellbeing/src/rag/retriever.py): `ResearchRetriever` executes k-NN semantic similarity search against ChromaDB.
- [`adapter.py`](file:///c:/Users/aksha/OneDrive/Desktop/music-brain-wellbeing/src/rag/adapter.py): `RecommendationQueryAdapter` converts user profiles / track acoustic features into academic research queries.
- [`evidence.py`](file:///c:/Users/aksha/OneDrive/Desktop/music-brain-wellbeing/src/rag/evidence.py): `EvidencePackage` formats retrieved chunks and sources into a JSON-serializable data contract.

---

## 6. Scientific Boundary Rules
1. **NEVER Claim Clinical Treatment:** Music recommendations are not clinical anxiety treatments.
2. **Preserve Nuance:** Include studies showing mixed or non-significant results (e.g. van den Tol et al. 2022, PMID 35714120).
3. **Population Separation:** MXMH survey dataset $\neq$ Spotify user dataset. No row-level joins are performed.
4. **Nuanced Phrasing:** Use terms like "Research has investigated...", "Evidence suggests...", "Findings indicate context-dependence...".

---

## 7. Interview Revision Questions & Answers

### Q1: What is RAG and why is it preferred over fine-tuning for domain knowledge?
**Answer:** RAG combines a vector database retrieval system with an LLM. It is preferred over fine-tuning for domain knowledge because: (1) RAG eliminates hallucinations by grounding answers in explicit retrieved context, (2) updating domain knowledge requires simply updating vector store documents rather than costly model training, and (3) RAG provides exact citation provenance (PMIDs, DOIs) for auditability.

### Q2: Why did you choose `sentence-transformers/all-MiniLM-L6-v2`?
**Answer:** `all-MiniLM-L6-v2` is a lightweight, highly efficient 6-layer Transformer that maps sentences to 384-dimensional dense vectors. It balances state-of-the-art semantic embedding quality with ultra-fast inference speed and minimal memory footprint, making it ideal for local CPU/GPU execution.

### Q3: How does ChromaDB handle vector storage and retrieval?
**Answer:** ChromaDB uses HNSW (Hierarchical Navigable Small World) graph indexing for fast approximate nearest neighbor (ANN) vector search. It persists document text, 384-dim dense float embeddings, metadata, and chunk IDs to disk, allowing fast cosine similarity queries.

### Q4: Why are stable chunk IDs important in RAG ingestion?
**Answer:** Stable chunk IDs (e.g., `{doc_id}_chunk_{i}`) make the ingestion pipeline idempotent. When re-running ingestion, ChromaDB upserts existing chunk IDs rather than duplicating documents, preventing duplicate vector entries.

### Q5: How do you bridge music recommendations with scientific literature?
**Answer:** We built `RecommendationQueryAdapter`, which translates quantitative user acoustic features (e.g. energy 0.32, tempo 82 BPM, acousticness 0.78) into academic search concepts (e.g. "low energy soothing acoustics slow tempo stress recovery"). This retrieves relevant scientific literature while explicitly separating observed user behavior from clinical evidence.

### Q6: What is the structure of your Evidence Package?
**Answer:** The `EvidencePackage` is a structured JSON payload containing: `query`, `retrieved_chunks` (with text, distance, and similarity score), `sources` (distinct titles, authors, PMIDs, DOIs), and `retrieval_metadata` (timestamp, model name, collection count).

### Q7: Why not use LangChain or LangGraph?
**Answer:** Avoiding abstraction frameworks allowed us to build a transparent, first-principles RAG pipeline without hidden prompt wrappers, unexpected overhead, or black-box dependencies.

### Q8: How do you handle non-significant or mixed scientific findings in RAG?
**Answer:** We explicitly curated literature reporting non-significant or mixed effects (such as van den Tol et al. 2022, PMID 35714120). Storing these in ChromaDB ensures semantic search retrieves nuanced evidence rather than confirmation-biased claims.

### Q9: How do you prevent medical overclaims in the AI system?
**Answer:** We enforce strict scientific boundaries: music recommendations are framed as non-clinical acoustic context, and prompts/adapters use non-directive language ("evidence suggests", "research has investigated").

### Q10: How do you verify RAG system performance without external API calls?
**Answer:** We created an offline unit test suite (`tests/test_rag.py`) covering 13 test scenarios including JSONL schema validation, chunking, stable IDs, embedding dimensions, vector insertion, persistence, top-k retrieval, empty corpus handling, evidence schema, and idempotent deduplication.
