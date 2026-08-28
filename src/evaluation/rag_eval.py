"""
RAG Retriever Evaluator Module

WHAT: Systematically evaluates Phase 4 Research-Grounded RAG semantic retrieval layer performance.

WHY: Measures retrieval Hit Rate @ K, Mean Reciprocal Rank (MRR), cosine distance distribution,
and empty query resilience over a controlled, version-controlled benchmark query dataset.

EXCLUDED METRICS & JUSTIFICATION:
    - Full Corpus Recall: Excluded because PubMed indexing uses a small, curated 10-paper research corpus.
    - Automated Ground-Truth Relevance Scoring: Excluded to prevent fabricating synthetic relevancy labels.
"""

from typing import List, Dict, Any
from src.rag.retriever import ResearchRetriever
from .metrics import retrieval_hit_rate_at_k, mean_reciprocal_rank


class RAGRetrieverEvaluator:
    """
    Evaluator executing semantic retrieval benchmarks over controlled evaluation queries.
    """

    def evaluate_retriever(
        self,
        retriever: ResearchRetriever,
        benchmark_queries: List[Dict[str, Any]],
        top_k: int = 2
    ) -> Dict[str, Any]:
        """
        Evaluate RAG retriever performance over a list of benchmark query dictionaries.

        Args:
            retriever: Initialized ResearchRetriever instance.
            benchmark_queries: List of query dictionaries with query_text and expected_pmids.
            top_k: Top-K chunks to retrieve per query.

        Returns:
            Dictionary containing aggregate hit rate @ K, MRR, mean distance, and query breakdown.
        """
        if not benchmark_queries:
            return {"total_queries": 0, "mean_hit_rate_at_k": 0.0, "mean_mrr": 0.0}

        hit_rates = []
        mrrs = []
        distances = []
        query_breakdowns = []

        for q in benchmark_queries:
            q_text = q.get("query_text", "")
            expected_pmids = q.get("expected_pmids", [])

            retrieved = retriever.retrieve(q_text, top_k=top_k)

            hr = retrieval_hit_rate_at_k(retrieved, expected_pmids, k=top_k)
            mrr = mean_reciprocal_rank(retrieved, expected_pmids)

            hit_rates.append(hr)
            mrrs.append(mrr)

            q_distances = [float(c.get("distance", 0.0)) for c in retrieved if "distance" in c]
            if q_distances:
                distances.extend(q_distances)

            query_breakdowns.append({
                "query_id": q.get("query_id"),
                "query_text": q_text,
                "retrieved_count": len(retrieved),
                "hit_rate_at_k": hr,
                "mrr": mrr,
                "top_retrieved_pmid": retrieved[0].get("metadata", {}).get("pmid") if retrieved else None
            })

        # Test empty query resilience
        empty_retrieved = retriever.retrieve("", top_k=top_k)
        empty_resilience_pass = (isinstance(empty_retrieved, list))

        mean_hr = round(sum(hit_rates) / float(len(hit_rates)), 4) if hit_rates else 0.0
        mean_mrr_val = round(sum(mrrs) / float(len(mrrs)), 4) if mrrs else 0.0
        avg_dist = round(sum(distances) / float(len(distances)), 4) if distances else 0.0

        return {
            "total_queries": len(benchmark_queries),
            "top_k": top_k,
            "mean_hit_rate_at_k": mean_hr,
            "mean_mrr": mean_mrr_val,
            "average_cosine_distance": avg_dist,
            "empty_query_resilience_pass": empty_resilience_pass,
            "query_breakdowns": query_breakdowns
        }
