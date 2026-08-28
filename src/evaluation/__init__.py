"""
Systematic Evaluation Layer Package

WHAT: Exports metrics utilities and evaluation runner classes for recommendations, RAG retrieval,
LLM explanations, and end-to-end pipeline benchmark execution.
"""

from .metrics import (
    mean_vector_distance,
    ranking_monotonicity_score,
    intra_list_diversity,
    cluster_coverage,
    retrieval_hit_rate_at_k,
    mean_reciprocal_rank,
    citation_grounding_accuracy,
    track_grounding_accuracy,
    safety_compliance_score
)
from .recommendation_eval import RecommendationEvaluator
from .rag_eval import RAGRetrieverEvaluator
from .llm_eval import LLMExplanationEvaluator
from .end_to_end_eval import EndToEndEvaluator

__all__ = [
    "mean_vector_distance",
    "ranking_monotonicity_score",
    "intra_list_diversity",
    "cluster_coverage",
    "retrieval_hit_rate_at_k",
    "mean_reciprocal_rank",
    "citation_grounding_accuracy",
    "track_grounding_accuracy",
    "safety_compliance_score",
    "RecommendationEvaluator",
    "RAGRetrieverEvaluator",
    "LLMExplanationEvaluator",
    "EndToEndEvaluator",
]
