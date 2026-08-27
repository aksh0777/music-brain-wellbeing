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

[Future Layers]
5. Research-Grounded RAG Layer (Phase 4 — Future)
6. Non-Clinical LLM Explanation Layer (Phase 5 — Future)
7. FastAPI Serving Backend & Interactive Web UI (Phase 6 — Future)
```

---

## 2. Research Origin, Key Findings & Scientific Limitations

- **Research Origin (MXMH Survey)**: The project originated with a supervised regression question: *Can music listening habits predict self-reported anxiety ($0-10$ scale)?* Using the MXMH dataset (736 records), we tuned a Decision Tree to `max_depth=3` (5-fold CV), achieving a modest Test $R^2 = 0.0636$ and Test RMSE = `2.7499`.
- **Scientific Significance of Low $R^2$**: Explaining ~6.4% of the variance proved empirically that music habits alone cannot reliably predict or diagnose anxiety. Rather than a failure, this finding served as an essential scientific boundary that guided our pivot away from clinical prediction.
- **Architectural Pivot to Music Intelligence**: We evolved the project toward understanding individual listening behavior via Spotify streams (30-min sessions, cyclical temporal features, K-Means acoustic profiles) and explainable recommendation ranking.
- **Data Boundary (No Row-Level Join)**: MXMH survey respondents and Spotify users are **completely separate populations**. There is **no row-level join** between datasets, and survey anxiety scores are never transferred to Spotify users.
- **Acoustic Descriptors $\neq$ Clinical Moods**: Audio clusters (K-Means, $K=4$) measure physical sound properties (tempo, energy, acousticness). They are strictly labeled **"Acoustic Profiles"**, avoiding clinical diagnoses.
- **Non-Causal Boundary**: Recommendations match user musical preferences and acoustic context. We **NEVER** claim *"Song X cures anxiety"*. Any anxiety relationship remains observational.

---

## 3. Repository Structure

```text
.env.example                             # Template environment variable configuration
data/
  ├── raw/
  │   ├── mxmh_survey_results.csv        # Untouched original survey dataset
  │   └── spotify/tracks.csv             # Spotify track catalog
  └── processed/
      ├── mxmh_cleaned.csv               # Untouched cleaned survey dataset
      └── music/listening_history.csv    # Synthetic timestamped stream logs (data_type="synthetic/demo")
docs/                                    # Technical guides, analysis docs, decision & learning logs
docs/figures/                            # Execution plots (evaluation, complexity, cluster & rec summaries)
notebooks/                               # Executable Jupyter notebooks (01 to 14)
src/
  ├── data/                              # Data loading and schema validation
  ├── features/                          # Scaling, sessionization, temporal encodings, clustering, user profiling
  ├── recommendation/                    # Candidate retrieval, similarity, ranking, top-N recommendation engine
  ├── spotify/                           # Spotify OAuth, client, adapter mapping, pipeline coordinator
  ├── models/                            # Regression modeling pipelines
  └── evaluation/                        # Evaluation metrics and residual diagnostic utilities
tests/                                   # Unit test suite (28 unit tests passing cleanly)
```

---

## 4. How to Reproduce & Run Tests

1. Activate virtual environment and verify dependencies:
   ```bash
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Configure environment variables (optional for live Spotify API access):
   ```bash
   cp .env.example .env
   # Edit .env with your SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
   ```
3. Run the complete unit test suite (28 unit tests across Phase 1, 2, and 3):
   ```bash
   .venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
   ```
4. Execute Spotify integration demonstration notebook:
   ```bash
   .venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/14_spotify_integration_demo.ipynb --output 14_spotify_integration_demo.ipynb
   ```
