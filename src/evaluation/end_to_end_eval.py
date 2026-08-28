"""
End-to-End Pipeline Evaluator Module

WHAT: Systematically evaluates the full end-to-end Music Brain Wellbeing System pipeline:
      User Profile → Recommendation Engine → RAG Retrieval → EvidencePackage → ExplanationGenerator → GroundingValidator.

WHY: Measures end-to-end pipeline completion rate, stage-by-stage latency breakdowns, validation pass rates,
and edge-case resilience across controlled user profile personas.
"""

import time
import pandas as pd
from typing import List, Dict, Any
from src.recommendation.recommender import recommend_tracks
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import ResearchRetriever
from src.rag.adapter import RecommendationQueryAdapter
from src.rag.evidence import build_evidence_package
from src.explanation.schemas import ExplanationRequest, SafetyConstraints
from src.explanation.explanation_generator import ExplanationGenerator


class EndToEndEvaluator:
    """
    Evaluator executing end-to-end pipeline benchmark runs across controlled user profiles.
    """

    def __init__(
        self,
        catalog_df: pd.DataFrame,
        vector_store: VectorStore,
        embedder: EmbeddingModel
    ):
        self.catalog_df = catalog_df
        self.vector_store = vector_store
        self.embedder = embedder
        self.retriever = ResearchRetriever(embedder=self.embedder, vector_store=self.vector_store)
        self.adapter = RecommendationQueryAdapter()
        self.explanation_generator = ExplanationGenerator(mode="DEMO")

    def evaluate_pipeline(
        self,
        controlled_profiles: List[Dict[str, Any]],
        top_n_recs: int = 3,
        top_k_rag: int = 2
    ) -> Dict[str, Any]:
        """
        Execute end-to-end pipeline benchmark over controlled user profiles.

        Args:
            controlled_profiles: List of controlled user profile dictionaries.
            top_n_recs: Top-N recommendations to generate per profile.
            top_k_rag: Top-K research chunks to retrieve per profile.

        Returns:
            Dictionary containing success rates, latency metrics, and per-profile execution details.
        """
        if not controlled_profiles:
            return {"total_runs": 0, "success_rate": 0.0}

        successful_runs = 0
        validated_runs = 0
        profile_results = []

        rec_latencies = []
        rag_latencies = []
        exp_latencies = []
        total_latencies = []

        for prof in controlled_profiles:
            persona = prof.get("persona", "default")
            t_start = time.perf_counter()

            try:
                # Stage 1: Recommendation Generation
                t_rec_start = time.perf_counter()
                recs_df = recommend_tracks(prof, self.catalog_df, top_n=top_n_recs)
                t_rec = (time.perf_counter() - t_rec_start) * 1000.0
                rec_latencies.append(t_rec)

                recs_list = recs_df.to_dict(orient="records")

                # Stage 2: RAG Research Retrieval
                t_rag_start = time.perf_counter()
                query = self.adapter.construct_query_from_profile(prof, target_topic="relaxing acoustic stress recovery")
                chunks = self.retriever.retrieve(query, top_k=top_k_rag)
                evidence_pkg = build_evidence_package(query, chunks)
                t_rag = (time.perf_counter() - t_rag_start) * 1000.0
                rag_latencies.append(t_rag)

                # Stage 3: Explanation Request & Generation
                t_exp_start = time.perf_counter()
                req = ExplanationRequest(
                    user_profile=prof,
                    recommendations=recs_list,
                    acoustic_profiles={"audio_feature_summary": prof.get("audio_feature_summary", {})},
                    evidence_package=evidence_pkg,
                    safety_constraints=SafetyConstraints()
                )
                response = self.explanation_generator.generate(req)
                t_exp = (time.perf_counter() - t_exp_start) * 1000.0
                exp_latencies.append(t_exp)

                t_total = (time.perf_counter() - t_start) * 1000.0
                total_latencies.append(t_total)

                successful_runs += 1
                if response.is_validated:
                    validated_runs += 1

                profile_results.append({
                    "persona": persona,
                    "user_id": prof.get("user_id"),
                    "status": "SUCCESS",
                    "recommendation_count": len(recs_list),
                    "retrieved_chunks_count": len(chunks),
                    "is_validated": response.is_validated,
                    "grounding_score": response.grounding_score,
                    "rec_latency_ms": round(t_rec, 2),
                    "rag_latency_ms": round(t_rag, 2),
                    "exp_latency_ms": round(t_exp, 2),
                    "total_latency_ms": round(t_total, 2)
                })

            except Exception as e:
                profile_results.append({
                    "persona": persona,
                    "user_id": prof.get("user_id"),
                    "status": "FAILED",
                    "error": str(e)
                })

        total_runs = len(controlled_profiles)
        success_rate = round(successful_runs / float(total_runs), 4) if total_runs > 0 else 0.0
        val_pass_rate = round(validated_runs / float(successful_runs), 4) if successful_runs > 0 else 0.0

        avg_rec_lat = round(sum(rec_latencies) / len(rec_latencies), 2) if rec_latencies else 0.0
        avg_rag_lat = round(sum(rag_latencies) / len(rag_latencies), 2) if rag_latencies else 0.0
        avg_exp_lat = round(sum(exp_latencies) / len(exp_latencies), 2) if exp_latencies else 0.0
        avg_total_lat = round(sum(total_latencies) / len(total_latencies), 2) if total_latencies else 0.0

        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "pipeline_success_rate": success_rate,
            "grounding_validation_pass_rate": val_pass_rate,
            "latency_breakdown_ms": {
                "avg_recommendation_ms": avg_rec_lat,
                "avg_rag_retrieval_ms": avg_rag_lat,
                "avg_explanation_generation_ms": avg_exp_lat,
                "avg_total_pipeline_ms": avg_total_lat
            },
            "profile_execution_details": profile_results
        }
