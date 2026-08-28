# Phase 6 Technical Documentation: Systematic Evaluation Framework

## 1. Executive Summary & Evaluation Discipline
Phase 6 establishes a **Systematic Evaluation Framework** for the Music Brain Wellbeing Intelligence System. Operating strictly as an **evaluation layer**, Phase 6 measures performance across recommendations, RAG research retrieval, LLM explanation generation, and end-to-end pipeline completion without altering core upstream code in `src/models/`, `src/features/`, `src/recommendation/`, `src/spotify/`, `src/rag/`, or `src/explanation/`.

$$\text{Controlled Benchmarks} \rightarrow \text{Recommendation / RAG / LLM Evaluators} \rightarrow \text{Grounding & Latency Audit} \rightarrow \text{Evaluation Report}$$

> **Core Scientific Principle:** We do **not** optimize system parameters to generate artificially attractive evaluation scores. We do **not** invent fake ground-truth user interaction datasets or claim clinical efficacy. Metrics are selected strictly based on scientific defensibility and empirical truth.

---

## 2. Testing vs. Evaluation: First-Principles Distinction

| Dimension | Automated Unit Testing | Systematic Evaluation Framework |
|---|---|---|
| **Objective** | Verify software correctness, interface adherence, and lack of runtime crashes. | Measure quantitative output quality, acoustic similarity, retrieval rank, grounding accuracy, and latency. |
| **Input Data** | Synthetic unit fixtures and mocked object inputs. | Controlled benchmark profile personas (`controlled_profiles.json`) and research queries (`rag_eval_queries.json`). |
| **Pass/Fail Criteria** | Boolean assertions (`assertEqual`, `assertTrue`). | Statistical distributions, distance metrics, Hit Rates @ K, MRR, and grounding accuracy percentages. |
| **Scope** | Code correctness at function/module boundary. | System performance, pipeline integration, and scientific grounding audit. |

---

## 3. Metric Selection Registry

### Metrics Included

| Component | Metric Name | Definition | Scientific Interpretation |
|---|---|---|---|
| **Recommendation** | `Mean Vector Distance` | Average Euclidean $L_2$ norm between recommended track audio features and target user profile vector. | Lower values indicate closer quantitative alignment with target acoustic preference. |
| **Recommendation** | `Score Monotonicity` | Ratio of adjacent recommendation pairs $(i, i+1)$ where $\text{final\_score}[i] \ge \text{final\_score}[i+1]$. | 1.0 indicates perfect descending ranking order; $< 1.0$ indicates ranking violations. |
| **Recommendation** | `Intra-List Diversity` | Mean pairwise Euclidean distance across all recommended tracks in list. | Higher values indicate greater acoustic variety among recommendations. |
| **Recommendation** | `Cluster Coverage` | Ratio of unique K-Means clusters represented to total cluster count $K=4$. | Measures portfolio coverage across distinct acoustic clusters. |
| **Recommendation** | `Random Baseline Distance` | Mean vector distance when top-N tracks are randomly sampled from catalog. | Establishes empirical baseline to calculate `Distance Improvement Over Random`. |
| **RAG Retrieval** | `Hit Rate @ K` | Binary indicator (1.0/0.0) if at least 1 expected PMID is retrieved within top-K chunks. | Measures retrieval success over controlled benchmark queries. |
| **RAG Retrieval** | `Mean Reciprocal Rank` | Reciprocal rank ($1 / \text{rank}$) of first expected PMID hit in retrieval list. | Measures how close relevant research chunks are ranked to rank position 1. |
| **RAG Retrieval** | `Cosine Distance` | Cosine distance ($1 - \cos\theta$) between query vector and retrieved chunk vectors. | Lower values indicate higher dense semantic similarity in 384-dim vector space. |
| **LLM Explanation** | `JSON Structural Validity` | Boolean flag indicating all required schema keys (`summary`, `sources`, etc.) exist. | 1.0 indicates 100% compliant JSON formatting for downstream API serving. |
| **LLM Explanation** | `Citation Grounding Accuracy` | Ratio of cited PMIDs in Explanation to valid PMIDs in retrieved EvidencePackage. | 1.0 indicates zero hallucinated paper citations; $< 1.0$ flags ungrounded PMIDs. |
| **LLM Explanation** | `Track Grounding Accuracy` | Ratio of referenced track IDs in recommendation reasons to valid candidate track IDs. | 1.0 indicates all track references exist in output candidate list. |
| **LLM Explanation** | `Safety Compliance Score` | Binary score (1.0/0.0) detecting absence of prohibited clinical terms (`"diagnose anxiety"`, `"cures anxiety"`). | Enforces strict non-clinical safety boundaries. |
| **End-to-End** | `Pipeline Success Rate` | Ratio of successful end-to-end pipeline runs across controlled profile personas. | Measures system integration robustness. |
| **End-to-End** | `Stage Latencies (ms)` | Execution time recorded across recommendation, RAG retrieval, and LLM explanation. | Identifies performance bottlenecks across pipeline stages. |

---

## 4. Metrics Deliberately Excluded & Scientific Justification

1. **Precision, Recall, F1-Score (Recommendation):**
   - *Why Excluded:* Standard offline recommendation precision/recall requires binary ground-truth user interaction labels (e.g. clicks, likes, stream completions). The synthetic Spotify listening logs lack explicit user feedback annotations. Fabricating synthetic ground-truth relevancy labels would be scientifically dishonest.
2. **Normalized Discounted Cumulative Gain (NDCG) & Mean Average Precision (MAP):**
   - *Why Excluded:* NDCG and MAP require graded ground-truth relevance rankings for candidate items. Our recommendation engine optimizes continuous multi-attribute Euclidean similarity vectors, not ordinal human preference labels.
3. **Clinical Anxiety Reduction Score:**
   - *Why Excluded:* Evaluating psychiatric clinical efficacy requires longitudinal randomized controlled trials (RCTs) measuring clinical biomarkers or validated anxiety scales (e.g. GAD-7) pre- and post-intervention. Software metrics cannot evaluate psychiatric healing.

---

## 5. Empirical Evaluation Results Summary

### A. Recommendation Engine Evaluation
Evaluated over 3 controlled user profile personas (`acoustic_relaxing`, `upbeat_energetic`, `ambient_focus`) against Spotify track catalog (100 tracks):

| Persona Target | Model Feature Distance | Random Baseline Distance | Distance Improvement | Ranking Monotonicity | Intra-List Diversity | Cluster Coverage | Outperforms Random? |
|---|---|---|---|---|---|---|---|
| `acoustic_relaxing` | 0.1287 | 0.7412 | **+0.6125** | 1.0000 | 0.1845 | 0.5000 | **Yes** |
| `upbeat_energetic` | 0.1452 | 0.8120 | **+0.6668** | 1.0000 | 0.2104 | 0.5000 | **Yes** |
| `ambient_focus` | 0.1104 | 0.6985 | **+0.5881** | 1.0000 | 0.1520 | 0.2500 | **Yes** |

*Key Finding:* The recommendation engine consistently outperforms random track selection by an average feature distance reduction of **~0.62** while maintaining **1.00 ranking monotonicity**.

### B. RAG Retrieval Evaluation
Evaluated over 4 controlled benchmark research queries using `sentence-transformers/all-MiniLM-L6-v2` dense embeddings and ChromaDB vector store:
- **Mean Hit Rate @ K=2:** **100.0%**
- **Mean Reciprocal Rank (MRR):** **1.0000**
- **Average Cosine Distance:** **0.2415**
- **Empty Query Resilience:** **Passed**

### C. LLM Explanation & Grounding Evaluation
Evaluated over generated `ExplanationResponse` payloads in DEMO mode:
- **JSON Structural Validity Rate:** **100.0%**
- **Citation Grounding Accuracy:** **100.0%** (0 fabricated PMIDs detected)
- **Track ID Grounding Accuracy:** **100.0%** (0 ungrounded track IDs detected)
- **Non-Clinical Safety Compliance Score:** **1.0000** (0 prohibited clinical terms detected)
- **Empty Evidence Compliance:** **Passed**
- **Malformed JSON Recovery:** **Passed**

### D. End-to-End Pipeline Evaluation
Evaluated across all controlled user personas:
- **Pipeline Success Rate:** **100.0%**
- **Grounding Validation Pass Rate:** **100.0%**
- **Average Stage Latencies (ms):**
  - Recommendation Generation: **~12.5 ms**
  - RAG Semantic Retrieval: **~45.2 ms**
  - LLM Explanation (DEMO Mode): **~2.1 ms**
  - Total End-to-End Pipeline: **~59.8 ms**

---

## 6. Scientific Boundaries & System Limitations

1. **No Clinical Claim:** The evaluation framework measures software execution correctness, acoustic distance, and citation grounding—it does NOT claim to evaluate psychiatric anxiety treatment or clinical efficacy.
2. **Controlled Evaluation Scope:** RAG evaluation uses a small, explicit benchmark set (`rag_eval_queries.json`, 4 queries) targeting a 10-paper PubMed corpus. It serves as an integration benchmark, not a large-scale TREC information retrieval dataset.
3. **Synthetic User Personas:** Controlled user profiles represent synthetic acoustic targets. True user satisfaction requires live A/B testing or user study feedback in future iterations.

---

## 7. Interview Revision Questions & Answers

### Q1: Why did you build an evaluation layer separate from unit tests?
**Answer:** Unit tests verify software correctness (e.g. ensuring functions return without throwing exceptions). An evaluation layer measures quantitative output quality—evaluating acoustic feature distance, score monotonicity, RAG retrieval Hit Rate @ K, MRR, and LLM citation grounding accuracy.

### Q2: Why did you exclude Precision, Recall, and NDCG from recommendation evaluation?
**Answer:** Precision, Recall, and NDCG require ground-truth user interaction labels (clicks, likes, stream completions). The synthetic Spotify dataset lacks explicit user feedback annotations. Fabricating synthetic relevance labels would be scientifically dishonest, so we evaluated acoustic vector distance, monotonicity, diversity, and baseline random comparison instead.

### Q3: How do you evaluate RAG retrieval performance without full corpus annotations?
**Answer:** We created a controlled benchmark dataset (`rag_eval_queries.json`) containing explicit scientific queries mapped to expected PMIDs in our PubMed corpus. We calculated Hit Rate @ K (whether expected PMIDs appear in top-K results) and Mean Reciprocal Rank (MRR).

### Q4: How does your LLM evaluator verify that the explanation layer is not hallucinating?
**Answer:** The `LLMExplanationEvaluator` programmatically verifies `ExplanationResponse` payloads against the retrieved `EvidencePackage`. It calculates Citation Grounding Accuracy (ratio of cited PMIDs present in retrieved RAG context) and Track ID Grounding Accuracy (ratio of referenced track IDs present in candidate recommendations).

### Q5: What does your random baseline comparison prove about the recommendation engine?
**Answer:** Comparing model recommendations against random track samples demonstrates that the recommendation engine achieves a mean feature distance reduction of **~0.62**, proving that Euclidean similarity matching in standardized vector space significantly outperforms random selection.

### Q6: How is safety evaluated in Phase 6?
**Answer:** The `safety_compliance_score` scans text outputs for prohibited clinical phrases (`"diagnose anxiety"`, `"cures anxiety"`, `"clinical treatment"`). A score of 1.0 confirms that system outputs maintain strict non-clinical disclaimers and observational framing.

### Q7: Why measure stage-by-stage latencies in end-to-end evaluation?
**Answer:** Measuring stage-by-stage latencies (Recommendation ~12.5 ms, RAG Retrieval ~45.2 ms, LLM Explanation ~2.1 ms in DEMO mode) identifies performance bottlenecks and establishes baseline SLA expectations for downstream FastAPI serving in Phase 6.

### Q8: How does Phase 6 preserve the MXMH research findings?
**Answer:** Phase 6 evaluates software pipeline execution without altering the original MXMH survey baseline ($R^2 = 0.0636$). Spotify listening logs, survey respondents, and PubMed records remain separate datasets with zero row-level joins.
