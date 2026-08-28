# Music, Brain & Wellbeing Intelligence System

> Investigating self-reported mental wellbeing through behavioral music data, audio feature engineering, personalized recommendations, and secure Spotify Web API stream integration.

---

## 1. System Architecture

```text
[Completed Layers]
1. MXMH Survey Dataset & Machine Learning Baseline
   └── Anxiety Prediction Model (Tuned Decision Tree, Test R² = 0.0636, RMSE = 2.7499)

2. Music Data + Listening Intelligence Foundation (Phase 1)
   ├── Spotify Track Catalog & Synthetic Stream Ingestion (data/raw/spotify/, data/processed/music/)
   ├── Audio Feature Scaling, Tempo Normalization & Range Checks (src/features/music_features.py)
   ├── 30-Minute Inactivity Gap Sessionization (src/features/sessions.py)
   ├── Cyclical Sin/Cos Temporal Encodings (src/features/temporal.py)
   ├── K-Means Acoustic Profile Clustering (src/features/clustering.py)
   └── Quantitative User Music Profile Aggregation (src/features/user_profile.py)

3. Personalized Music Recommendation Engine (Phase 2)
   ├── Candidate Pre-Filtering (src/recommendation/candidate_retrieval.py)
   ├── Standardized Vector Feature Alignment & Euclidean Similarity (src/recommendation/similarity.py)
   ├── Acoustic Profile Compatibility & Weighted Score Ranking (src/recommendation/ranking.py)
   ├── Cluster Diversity Filtering & Deterministic Explanation Generation (src/recommendation/recommender.py)
   └── Executable Notebook (notebooks/13_recommendation_engine.ipynb)

4. Spotify API Integration Layer & Adapter Architecture (Phase 3)
   ├── OAuth 2.0 Credential Validation & Token Management (src/spotify/spotify_auth.py)
   ├── Web API HTTP Client with 401 Refresh & 429 Retry Logic (src/spotify/spotify_client.py)
   ├── Spotify JSON to Internal Schema Adapter (src/spotify/spotify_mapper.py)
   ├── End-to-End Pipeline Coordinator (src/spotify/spotify_pipeline.py)
   └── Executable Demo Notebook & Test Suite (notebooks/14_spotify_integration_demo.ipynb)

5. Research-Grounded RAG Layer (Phase 4)
   ├── Curated PubMed Research Corpus & Provenance (data/raw/research/music_wellbeing_research.jsonl)
   ├── Document Chunker with Stable Chunk IDs & Metadata Preservation (src/rag/chunker.py)
   ├── 384-Dim Dense Embedding Layer (sentence-transformers/all-MiniLM-L6-v2, src/rag/embeddings.py)
   ├── Local Persistent ChromaDB Vector Store & HNSW Cosine Index (src/rag/vector_store.py)
   ├── Idempotent Ingestion Pipeline (src/rag/ingest.py)
   ├── Semantic Research Retriever (src/rag/retriever.py)
   ├── Recommendation to Research Query Adapter (src/rag/adapter.py)
   ├── Evidence Package Data Contract (src/rag/evidence.py)
   └── Executable Demo Notebook & Test Suite (notebooks/15_rag_retrieval_demo.ipynb)

6. Grounded Non-Clinical LLM Explanation Layer (Phase 5)
   ├── Input & Output Data Contracts (src/explanation/schemas.py)
   ├── Modular LLM Provider Abstraction & DEMO Mode (src/explanation/llm_provider.py)
   ├── Safety Rules & Grounding Prompt Engineering (src/explanation/prompt.py)
   ├── Citation & Track ID Grounding Validator (src/explanation/validation.py)
   ├── End-to-End Pipeline Coordinator (src/explanation/explanation_generator.py)
   └── Executable Demo Notebook & Test Suite (notebooks/16_llm_explanation_demo.ipynb)

7. Systematic Evaluation Framework (Phase 6)
   ├── Controlled Evaluation Benchmarks (data/evaluation/controlled_profiles.json, rag_eval_queries.json)
   ├── Evaluation Metric Utilities (src/evaluation/metrics.py)
   ├── Recommendation Evaluator & Baseline Comparison (src/evaluation/recommendation_eval.py)
   ├── RAG Research Retrieval Evaluator (src/evaluation/rag_eval.py)
   ├── LLM Explanation & Grounding Evaluator (src/evaluation/llm_eval.py)
   ├── End-to-End Pipeline Evaluator (src/evaluation/end_to_end_eval.py)
   └── Executable Evaluation Notebook & Test Suite (notebooks/17_system_evaluation.ipynb)

[Future Layers]
8. FastAPI Serving Backend & Interactive Web UI (Phase 7 — Future)
```

---

## 2. Research Origin, Key Findings & Scientific Limitations

- **Research Origin (MXMH Survey)**: The project originated with a supervised regression question: *Can music listening habits predict self-reported anxiety ($0-10$ scale)?* Using the MXMH dataset (736 records), we tuned a Decision Tree to `max_depth=3` (5-fold CV), achieving a modest Test $R^2 = 0.0636$ and Test RMSE = `2.7499`.
- **Scientific Significance of Low $R^2$**: Explaining ~6.4% of the variance proved empirically that music habits alone cannot reliably predict or diagnose anxiety. Rather than a failure, this finding served as an essential scientific boundary that guided our pivot away from clinical prediction.
- **Architectural Pivot to Music Intelligence**: We evolved the project toward understanding individual listening behavior via Spotify streams (30-min sessions, cyclical temporal features, K-Means acoustic profiles), explainable recommendation ranking, scientific RAG evidence retrieval, grounded non-clinical LLM explanation generation, and systematic evaluation benchmarking.
- **Data Boundary (No Row-Level Join)**: MXMH survey respondents, Spotify users, and PubMed research records are **completely separate data sources**. There is **no row-level join** across them.
- **Acoustic Descriptors $\neq$ Clinical Moods**: Audio clusters (K-Means, $K=4$) measure physical sound properties (tempo, energy, acousticness). They are strictly labeled **"Acoustic Profiles"**, avoiding clinical diagnoses.
- **Non-Causal Boundary**: Recommendations match user musical preferences and acoustic context. We **NEVER** claim *"Song X cures anxiety"*. Any anxiety relationship remains observational.

---

## 3. Repository Structure

```text
.env.example                             # Template environment variable configuration
data/
  ├── raw/
  │   ├── mxmh_survey_results.csv        # Untouched original survey dataset
  │   ├── spotify/tracks.csv             # Spotify track catalog
  │   └── research/                      # Verified PubMed research JSONL records
  ├── processed/
  │   ├── mxmh_cleaned.csv               # Untouched cleaned survey dataset
  │   └── music/listening_history.csv    # Synthetic timestamped stream logs (data_type="synthetic/demo")
  ├── evaluation/                        # Version-controlled benchmark profiles & queries
  └── vector_store/
      └── chroma/                        # Local persistent ChromaDB HNSW vector index
docs/                                    # Technical guides, analysis docs, decision & learning logs
docs/figures/                            # Execution plots (evaluation, complexity, cluster & rec summaries)
notebooks/                               # Executable Jupyter notebooks (01 to 17)
src/
  ├── data/                              # Data loading and schema validation
  ├── features/                          # Scaling, sessionization, temporal encodings, clustering, user profiling
  ├── recommendation/                    # Candidate retrieval, similarity, ranking, top-N recommendation engine
  ├── spotify/                           # Spotify OAuth, client, adapter mapping, pipeline coordinator
  ├── rag/                               # Document chunker, embeddings, ChromaDB, retriever, adapter, evidence
  ├── explanation/                       # Schemas, LLM provider, prompt engineering, grounding validator, coordinator
  ├── evaluation/                        # Metrics utilities, recommendation, RAG, LLM, and end-to-end evaluators
  └── models/                            # Regression modeling pipelines
tests/                                   # Unit test suite (64 unit tests passing cleanly)
```

---

## 4. How to Reproduce & Run Tests

1. Activate virtual environment and verify dependencies:
   ```bash
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Configure environment variables (optional for live Spotify or LLM APIs):
   ```bash
   cp .env.example .env
   # Edit .env with your SPOTIFY_CLIENT_ID and LLM_API_KEY
   ```
3. Run the complete unit test suite (64 unit tests across Phase 1, 2, 3, 4, 5, and 6):
   ```bash
   .venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
   ```
4. Execute Phase 6 systematic evaluation demonstration notebook:
   ```bash
   .venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/17_system_evaluation.ipynb --output 17_system_evaluation.ipynb
   ```



