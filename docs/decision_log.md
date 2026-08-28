# Architecture & Technical Decision Log

Every major technical decision made in the **Music, Brain & Wellbeing** project is recorded here with rationale, alternatives, tradeoffs, and evidence to prepare for interview defense.

---

### Decision 001: Adopt a Progressive Documentation-First Architecture
* **Date**: 2026-08-24
* **Decision**: Create a structured chapter-by-chapter documentation system under `docs/` before executing full ML modeling.
* **Why**: Ensures complete first-principles understanding, prevents black-box AI code generation, and builds interview readiness for every component.
* **Alternative**: Jumping directly into exploratory notebooks and writing unstructured scripts.
* **Why Not Alternative**: Unstructured notebooks lead to hidden state bugs, lack of production modularity, and inability to defend decisions in a Citi interview.
* **Tradeoff**: Takes more initial setup time before training ML models, but guarantees 100% auditability and code ownership.
* **Evidence**: Clean, modular structure in `docs/` and `src/`.

---

### Decision 002: Frame Project Problem as Observational Prediction & Association (Not Causation)
* **Date**: 2026-08-24
* **Decision**: Explicitly distinguish prediction, association, and causation. Avoid claiming music *causes* wellbeing changes.
* **Why**: Maintains scientific integrity. Survey and listening logs are observational.
* **Alternative**: Claiming music intervention directly increases wellbeing score.
* **Why Not Alternative**: Scientifically invalid without a randomized controlled trial (RCT) design.
* **Tradeoff**: Restricts claims to correlation/prediction metrics ($R^2$, ROC-AUC), but protects model credibility during technical audit/interview.
* **Evidence**: Chapter 01 research framework.

---

### Decision 003: Control Decision Tree Depth and Leaf Constraints
* **Date**: 2026-08-26
* **Decision**: Restrict Decision Tree Regressor complexity using hyperparameters like `max_depth` and `min_samples_leaf` rather than deploying default configurations.
* **Why**: An unconstrained tree splits nodes recursively until all leaf nodes are pure, resulting in perfect memorization of training sample noise (Train $R^2 = 1.0$) but failing to generalize to unseen test observations (Test $R^2 = -1.1078$). Controlling depth reduces model variance and stabilizes predictions.
* **Alternative**: Allow the tree to grow to full depth and apply post-pruning algorithms.
* **Why Not Alternative**: Post-pruning is computationally expensive and less directly interpretable during initial validation compared to pre-pruning limits.
* **Tradeoff**: Restricting tree depth introduces some bias (slightly underfitting the training data), but significantly reduces variance, raising the Test $R^2$ from `-1.1078` to `+0.0636`.

---

### Decision 004: Compare Models Against Simple Baselines (Naive Mean and Linear Regression)
* **Date**: 2026-08-26
* **Decision**: Evaluate all complex non-linear models against a naive mean predictor and a simple OLS Linear Regression baseline.
* **Why**: Establishes a reference threshold. A complex model is only useful if it outperforms simpler models.
* **Alternative**: Evaluate complex models in isolation.
* **Why Not Alternative**: Fails to demonstrate whether the model is learning anything useful beyond trivial prediction strategies.
* **Tradeoff**: Requires maintaining multiple parallel evaluation pipelines, but guarantees 100% rigor in defending model value during quant interviews.

---

### Decision 005: Implement K-Fold Cross-Validation on the Training Set
* **Date**: 2026-08-26
* **Decision**: Implement a 5-fold cross-validation strategy strictly on the training partition (`X_train`) for validation evaluation.
* **Why**: Evaluating model performance on a single train/validation split introduces noise due to partition variance (e.g. an unusually easy or hard validation set). CV averages metrics over 5 folds to provide a stable, low-variance estimate.
* **Alternative**: Rely on simple train/validation split.
* **Why Not Alternative**: Unreliable validation performance estimation leads to poor hyperparameter selection.
* **Tradeoff**: Increases training computation time by 5x, but prevents choosing a model that only fits a specific validation partition.

---

### Decision 006: Keep Test Set Untouched During Model Selection
* **Date**: 2026-08-26
* **Decision**: Hold out the test set (`X_test`, `y_test`) entirely, and evaluate on it only once at the end of the modeling stage.
* **Why**: To prevent data leakage. If we select models or tune hyperparameters based on test set scores, we implicitly leak test information into model selection, inflating our generalization estimate.
* **Alternative**: Use the test set score as a feedback loop for tuning hyperparameters.
* **Why Not Alternative**: Violates the definition of out-of-sample validation, yielding overly optimistic results that do not translate to real-world performance.
* **Tradeoff**: Limits our ability to double-check test results during tuning, but guarantees that final metrics are completely unbiased.

---

### Decision 007: Select Tuned Decision Tree (max_depth=3) as Candidate Model
* **Date**: 2026-08-26
* **Decision**: Select the tree model regularized to `max_depth=3` as our current best candidate.
* **Why**: It achieved the highest mean R² on cross-validation (-0.1027) and generalized successfully to the untouched test set, achieving a positive R² of `0.0636` (beating both Linear Regression and Naive baselines).
* **Alternative**: Select the default unregularized tree.
* **Why Not Alternative**: The unregularized model overfit severely (Test R² = -1.1078), memorizing training noise.
* **Tradeoff**: The shallow tree limits prediction variance, resulting in highly compressed predictions, but generalizes better than any other model tested.

---

### Decision 008: Restrict the Addition of Unnecessary Complex Algorithms
* **Date**: 2026-08-26
* **Decision**: Conclude the baseline modeling phase without immediately adding advanced models like Neural Networks, XGBoost, or SVM.
* **Why**: To prevent unnecessary complexity. Baseline and tuned tree models show that the predictive signal in these attributes is very weak ($R^2 \approx 0.06$). Adding complex black-box models would likely increase overfitting rather than extract a signal that is fundamentally absent from the feature set.
* **Alternative**: Train highly complex ensemble or deep learning models immediately.
* **Why Not Alternative**: Violates the principle of parsimony and risks overfitting to statistical noise, while adding training and maintenance complexity.
* **Tradeoff**: We report a simple, shallow tree rather than a complex model, but our findings are mathematically and scientifically honest.

---

### Decision 009: Report Weak Predictive Performance Honestly
* **Date**: 2026-08-26
* **Decision**: Explicitly report that the best model explains only 6.36% of the variance on unseen data, rather than claiming a "highly accurate predictor."
* **Why**: Maintains scientific integrity. Acknowledging that 93.64% of the variance in self-reported anxiety remains unexplained by daily music habits prevents overclaiming.
* **Alternative**: Use classification binning to report high training accuracy or frame metrics deceptively.
* **Why Not Alternative**: Dishonest framing fails technical audits and quant interviews, where rigour and self-skepticism are highly valued.
* **Tradeoff**: The project outcome is framed as having weak predictive power, but the methodology is robust and mathematically sound.

---

### Decision 010: Avoid Causal Claims and Focus Future Work on Data Quality
* **Date**: 2026-08-26
* **Decision**: Avoid stating that music-listening habits cause mental wellbeing changes, and focus future improvement recommendations on acquiring longitudinal data.
* **Why**: The survey data is observational and cross-sectional, which restricts us to modeling predictive association. Establishing causality is impossible without longitudinal monitoring or randomized interventions.
* **Alternative**: Frame results as "listening to Rock causes anxiety" or "music improves symptoms."
* **Why Not Alternative**: Scientifically invalid and clinically irresponsible.
* **Tradeoff**: Focuses recommendations on data collection adjustments rather than simple parameter tweaks, recognizing that the primary bottleneck is data quality, not the algorithm.

---

### Decision 011: Use Gini Impurity Reduction for Feature Importance Extraction
* **Date**: 2026-08-26
* **Decision**: Extract feature importances using the model's Gini impurity reduction (variance decrease) and report active columns strictly under predictive association guidelines.
* **Why**: Provides an internal look at which features the model uses to partition the data, mapping features directly to their predictive contributions.
* **Alternative**: Use permutation importance or SHAP values.
* **Why Not Alternative**: Gini importance is natively computed during fitting and is highly interpretable for shallow trees, making SHAP or permutation importances unnecessary at this baseline stage.
* **Tradeoff**: Gini importance can be biased toward high-cardinality continuous variables, but for a shallow tree with only 3 splits, this risk is easily audited.

---

### Decision 012: Formulate Model Selection Rationale Around MAE/RMSE and Generalization
* **Date**: 2026-08-26
* **Decision**: Frame model selection reasoning by balancing average absolute deviation (MAE), quadratic error penalties (RMSE), and the generalization gap (difference between train and test RMSE).
* **Why**: Selecting a model based solely on raw test set R² is a single-point estimate that ignores overfitting risks and error distributions.
* **Alternative**: Select model purely based on the highest raw training score.
* **Why Not Alternative**: Leads to choosing highly overfit models (like the unconstrained tree) that fail to generalize.
* **Tradeoff**: Requires running multiple parallel metrics calculations, but ensures a mathematically and scientifically robust selection process.

---

### Decision 013: Select Meaningful Relationships Based on Domain Knowledge
* **Date**: 2026-08-26
* **Decision**: Focus exploratory relationship analysis strictly on domain-relevant variables (`Age`, `Hours per day`, and `Music effects` vs. `Anxiety`) rather than blindly computing correlations on all 54 one-hot features.
* **Why**: To prevent data-dredging and keep the research insights interpretable and aligned with our model's top feature importances.
* **Alternative**: Compute and plot correlation matrices for all 54 feature combinations.
* **Why Not Alternative**: Yields overwhelming, uninterpretable plots containing statistical noise and meaningless associations.
* **Tradeoff**: Restricts the analysis to a small set of variables, but ensures high clarity and alignment with our modeling goals.

---

### Decision 014: Perform Targeted Robustness Checks (Outliers and Imputation Audit)
* **Date**: 2026-08-26
* **Decision**: Validate the stability of our main correlation coefficients by excluding extreme listening hours (>12 hours), excluding age leverage points (>70 years), and checking BPM median imputation impact against row deletion.
* **Why**: To verify that our findings are stable across population subsets and that our preprocessing decisions did not introduce covariance bias.
* **Alternative**: Rely on simple overall sample estimates without validation.
* **Why Not Alternative**: Exposes the project to criticisms that outliers or imputation choices dominated our final metrics.
* **Tradeoff**: Requires writing additional data filtering logic, but guarantees 100% confidence in the stability of our findings.

---

### Decision 015: Keep Music Catalog and Listening History Separate from MXMH Survey Data
* **Date**: 2026-08-27
* **Decision**: Store music track catalogs in `data/raw/spotify/tracks.csv` and listening events in `data/processed/music/listening_history.csv`, leaving `mxmh_survey_results.csv` and `mxmh_cleaned.csv` untouched.
* **Why**: The MXMH survey represents participant-level cross-sectional research data ($N=736$), while music catalog streams represent track-level audio descriptors and timestamped listening events. Merging them into a single raw file would corrupt observational units and create data lineage confusion.
* **Alternative**: Append music catalog features directly as new columns in the MXMH survey CSV.
* **Why Not Alternative**: Mixing participant survey rows with track catalog rows violates data normalization principles and breaks existing pipeline scripts.
* **Tradeoff**: Requires managing two data subtrees, but guarantees strict separation of concerns.

---

### Decision 016: Use K-Means for Acoustic Profile Clustering
* **Date**: 2026-08-27
* **Decision**: Adopt K-Means as the baseline unsupervised clustering algorithm for grouping audio features, evaluating $K \in [2, 8]$ using Silhouette Scores and Elbow plots.
* **Why**: K-Means is fast, interpretable, scalable, and produces explicit centroid vectors that can be directly mapped to human-readable audio structural descriptors.
* **Alternative**: Use DBSCAN or Hierarchical Clustering.
* **Why Not Alternative**: DBSCAN struggles with varying density across standardized audio features, and hierarchical clustering scales poorly ($O(N^3)$) for large track catalogs.
* **Tradeoff**: K-Means assumes spherical clusters, requiring proper feature standardization beforehand.

---

### Decision 017: Label Clusters as "Acoustic Profiles" Strictly Excluding Clinical Mood Claims
* **Date**: 2026-08-27
* **Decision**: Name K-Means clusters based strictly on physical audio properties (e.g. *"High Energy, High Danceability"*, *"Low Energy, Acoustic"*) and explicitly reject labeling them as "anxiety", "depression", or clinical mood diagnoses.
* **Why**: Audio descriptors measure physical signal properties (tempo, spectral energy, acousticness). Labeling a low-energy cluster as "depressive" is scientifically unsupported and clinically irresponsible.
* **Alternative**: Follow `spotify-brain`'s naming convention of calling clusters "Mood Clusters".
* **Why Not Alternative**: Conflates physical audio descriptors with human psychological states.
* **Tradeoff**: Requires clearer documentation boundaries, but prevents scientific overreach.

---

### Decision 018: Use 30-Minute Inactivity Gap Threshold for Sessionization
* **Date**: 2026-08-27
* **Decision**: Define a listening session as a contiguous sequence of plays where consecutive events are separated by less than 30 minutes ($\Delta t \le 30\text{ mins}$).
* **Why**: 30 minutes represents a standard behavioral heuristic for partitioning continuous listening streams into distinct temporal episodes (e.g., commute, study session, workout).
* **Alternative**: Treat each track as an isolated event or use a fixed 24-hour daily window.
* **Why Not Alternative**: Isolated track events ignore sequential continuity, while 24-hour daily windows fail to isolate distinct morning/evening activities.
* **Tradeoff**: 30 minutes is an engineering threshold rather than a psychological boundary, but it provides a clean, reproducible grouping logic.

---

### Decision 019: Use Content-Based Filtering First for Personalized Recommendations
* **Date**: 2026-08-27
* **Decision**: Implement content-based recommendation matching quantitative user feature profile vectors against standardized track audio features, rather than starting with collaborative filtering.
* **Why**: Content-based filtering does not require a dense multi-user interaction matrix and works effectively with single-user listening streams and track audio feature catalogs.
* **Alternative**: Start with collaborative filtering (Matrix Factorization / SVD).
* **Why Not Alternative**: Collaborative filtering requires a large multi-user rating/interaction matrix, which is unavailable in single-user listening history logs.
* **Tradeoff**: Content-based filtering cannot recommend tracks outside the user's historical feature boundary (filter bubble risk), but it is transparent, robust, and fully explainable.

---

### Decision 020: Use Euclidean Distance over Cosine Similarity in Standardized Feature Space
* **Date**: 2026-08-27
* **Decision**: Select Euclidean distance ($d = \sqrt{\sum (u_k - c_{jk})^2}$) transformed into bounded similarity ($S = 1 / (1 + d)$) as the primary similarity metric.
* **Why**: In standardized audio feature space ($\mu=0, \sigma=1$), feature magnitudes carry meaningful physical audio intensity properties (e.g. energy 0.9 vs 0.3 is an absolute acoustic difference that cosine similarity dampens).
* **Alternative**: Use Cosine Similarity ($\cos \theta$).
* **Why Not Alternative**: Cosine similarity measures angular orientation and ignores vector magnitude differences.
* **Tradeoff**: Euclidean distance requires strict feature standardization (`StandardScaler`) prior to calculation, but correctly captures absolute geometric proximity.

---

### Decision 021: Combine Feature Similarity (70%) and Acoustic Profile Compatibility (30%)
* **Date**: 2026-08-27
* **Decision**: Calculate final recommendation score as `0.7 * similarity_score + 0.3 * profile_compatibility_score`.
* **Why**: Combines continuous audio feature similarity with discrete behavioral habit distribution shares (cluster preference shares).
* **Alternative**: Use similarity score alone.
* **Why Not Alternative**: Similarity score alone ignores user habit distributions across discrete K-Means acoustic clusters.
* **Tradeoff**: Adds a weighting hyperparameter, but improves alignment with user cluster listening preferences.

---

### Decision 022: Use Deterministic Machine-Readable Explanations Before LLM Integration
* **Date**: 2026-08-27
* **Decision**: Generate recommendation reasons using deterministic Python string formatting rather than calling an LLM.
* **Why**: Decouples deterministic recommendation math from natural language LLM text generation, ensuring high execution speed and zero LLM hallucination risks.
* **Alternative**: Pass recommended tracks directly to an LLM to generate explanation text.
* **Why Not Alternative**: Introduces latency, API cost, and hallucination risks before establishing the core recommendation engine.
* **Tradeoff**: Deterministic explanations are template-based, but 100% reliable and fast.

---

### Decision 023: Use Server-Side OAuth 2.0 Credentials in Environment Variables
* **Date**: 2026-08-27
* **Decision**: Store `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, and `SPOTIFY_REDIRECT_URI` exclusively in server-side environment variables (`os.getenv`), ignoring `.env` in Git and providing `.env.example` as a placeholder template.
* **Why**: Prevents secret leaks, credential revocation, and unauthorized API usage.
* **Alternative**: Hardcode developer credentials or commit local `.env` files.
* **Why Not Alternative**: Exposes credentials publicly on GitHub.
* **Tradeoff**: Requires developers to configure local environment variables, but guarantees 100% security compliance.

---

### Decision 024: Create an Adapter Data Mapping Layer (`spotify_mapper.py`)
* **Date**: 2026-08-27
* **Decision**: Isolate external Spotify JSON payload parsing inside `map_recently_played_to_internal` rather than scattering Spotify field names across machine learning modules.
* **Why**: External APIs change field names and structures independently. Isolating API logic behind an adapter pattern protects downstream Phase 1 feature engineering and Phase 2 recommendation modules.
* **Alternative**: Use raw Spotify API field names (`items[].track.name`) directly across all feature and recommendation scripts.
* **Why Not Alternative**: Tight coupling causes downstream scripts to break whenever Spotify modifies API structures.
* **Tradeoff**: Requires writing an extra mapping transformation, but preserves modular system independence.

---

### Decision 025: Implement Dual `REAL` vs `DEMO` Execution Modes
* **Date**: 2026-08-27
* **Decision**: Provide a `mode` parameter (`REAL` or `DEMO`) in `run_spotify_pipeline` allowing full offline testing with mock API JSON fixtures without requiring live OAuth tokens.
* **Why**: Allows unit test suites, CI/CD pipelines, and demonstration notebooks to run deterministically offline without dependency on live network connectivity or user authorization.
* **Alternative**: Require live user OAuth sign-in for all tests and notebooks.
* **Why Not Alternative**: Fails CI/CD automated test suites whenever live tokens expire.
* **Tradeoff**: Requires maintaining a mock payload generator, but guarantees 100% test reproducibility.

---

### Decision 026: Local Persistent Vector Store (ChromaDB) over Cloud Vector DB
* **Date**: 2026-08-28
* **Decision**: Deploy ChromaDB as a local persistent vector database at `data/vector_store/chroma/` using HNSW cosine distance indexing rather than managed cloud vector databases (Pinecone, Qdrant Cloud).
* **Why**: Keeps the RAG retrieval layer fully self-contained, offline-capable, and zero-cost, while ensuring full developer control over index persistence.
* **Alternative**: Use cloud vector database APIs (Pinecone).
* **Why Not Alternative**: Introduces network latency, API key management overhead, and external cloud service dependencies for a research demonstration system.
* **Tradeoff**: Local vector store depends on local disk space, but provides instant offline execution and zero subscription cost.

---

### Decision 027: Use `sentence-transformers/all-MiniLM-L6-v2` Dense Embedding Model
* **Date**: 2026-08-28
* **Decision**: Standardize on `sentence-transformers/all-MiniLM-L6-v2` for generating 384-dimensional dense numerical vectors.
* **Why**: Provides an optimal trade-off between semantic search accuracy, CPU inference speed, and small model size (80MB footprint).
* **Alternative**: Use OpenAI `text-embedding-3-small` or large transformer models (`bge-large-en`).
* **Why Not Alternative**: OpenAI API requires paid cloud access and external network calls; larger models consume excessive local RAM without significant gains for short abstracts.
* **Tradeoff**: 384-dim vector space is slightly smaller than 1536-dim models, but handles 200-word research abstracts with exceptional semantic retrieval performance.

---

### Decision 028: Exclude High-Level Abstraction Frameworks (LangChain / LangGraph) in Phase 4
* **Date**: 2026-08-28
* **Decision**: Implement custom, first-principles RAG modules (`chunker.py`, `embeddings.py`, `vector_store.py`, `retriever.py`, `adapter.py`, `evidence.py`) without using LangChain or LangGraph.
* **Why**: Builds deep, transparent understanding of underlying RAG mechanics (vector space geometry, cosine distance, metadata filtering, chunking strategies) without black-box framework abstractions.
* **Alternative**: Use LangChain's `VectorStoreIndexWrapper` or `RetrievalQA`.
* **Why Not Alternative**: High-level wrappers hide vector store operations and complicate custom metadata validation and scientific evidence packaging.
* **Tradeoff**: Requires writing modular boilerplate code, but guarantees 100% code transparency and interview readiness.

---

### Decision 029: Stable Idempotent Chunk IDs (`{doc_id}_chunk_{i}`)
* **Date**: 2026-08-28
* **Decision**: Generate deterministic chunk IDs by combining document accession IDs with sequential chunk indices.
* **Why**: Enables idempotent upserts into ChromaDB. Repeated ingestion pipeline runs update existing records in-place rather than inserting duplicate vector entries.
* **Alternative**: Generate random UUIDs for each chunk during chunking.
* **Why Not Alternative**: Random UUIDs create duplicate vector records every time the ingestion script is executed.
* **Tradeoff**: Requires strictly unique document accession IDs in source JSONL, but prevents vector store corruption.

---

### Decision 030: Establish `LLMProvider` Abstract Interface for Vendor Decoupling
* **Date**: 2026-08-28
* **Decision**: Create an abstract base class `LLMProvider` defining `generate_explanation(system_prompt, user_prompt)` implemented by `DemoLLMProvider` and `GenericOpenAILLMProvider`.
* **Why**: Decouples application explanation logic from specific LLM providers (OpenAI, Groq, Ollama, local models).
* **Alternative**: Hardcode direct OpenAI Python SDK calls inside the explanation module.
* **Why Not Alternative**: Tight coupling breaks offline unit testing, prevents swapping LLM backends, and introduces hard external library dependencies.
* **Tradeoff**: Requires defining abstract base classes, but provides maximum architectural flexibility.

---

### Decision 031: Dual `DEMO` vs `REAL` LLM Execution Modes
* **Date**: 2026-08-28
* **Decision**: Implement a deterministic `DemoLLMProvider` that generates structured JSON explanations offline without API keys, alongside `GenericOpenAILLMProvider`.
* **Why**: Allows unit test suites, CI/CD pipelines, and local demonstration notebooks to run deterministically without requiring paid API keys or external network connectivity.
* **Alternative**: Require active API keys for all test runs.
* **Why Not Alternative**: Causes automated unit test failures when network connectivity is unavailable or API tokens expire.
* **Tradeoff**: Requires maintaining a mock provider implementation, but guarantees 100% test suite reliability.

---

### Decision 032: Programmatic Citation & Track ID Grounding Validator
* **Date**: 2026-08-28
* **Decision**: Build a deterministic `GroundingValidator` in `src/explanation/validation.py` that cross-checks cited PMIDs/DOIs against `EvidencePackage.sources` and track IDs against `recommendations`.
* **Why**: Generative LLMs generate text stochastically and may hallucinate citations despite strict prompt instructions. Post-generation validation ensures ungrounded assertions are flagged before reaching users.
* **Alternative**: Rely solely on system prompt instructions to eliminate hallucinations.
* **Why Not Alternative**: System prompts reduce hallucination risk but cannot guarantee 100% citation accuracy across all LLM models.
* **Tradeoff**: Introduces a minor validation processing step, but guarantees robust citation auditability.

---

### Decision 033: Enforce Strict Non-Clinical & Non-Causal Safety Guardrails
* **Date**: 2026-08-28
* **Decision**: Instruct the prompt engine and validator to prohibit clinical psychiatric terms (`"diagnose anxiety"`, `"cures anxiety"`, `"clinical treatment"`) and enforce non-causal disclaimers.
* **Why**: Music recommendations provide preference compatibility and observational scientific context, NOT medical diagnoses or clinical therapies.
* **Alternative**: Allow free-form medical advice generation if requested by users.
* **Why Not Alternative**: Violates medical safety standards and misrepresents observational research as clinical intervention.
* **Tradeoff**: Limits LLM expression to preference and observational context, but ensures complete scientific and ethical compliance.





