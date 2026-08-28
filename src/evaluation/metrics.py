"""
Evaluation Metric Utilities Module

WHAT: Provides pure, mathematically rigorous evaluation metric functions for recommendations, RAG retrieval,
LLM grounding, and safety compliance.

WHY: Establishes clear, defensible calculations with documented definitions, interpretations, and scientific limitations.
Prevents fabricating ungrounded metrics (such as Precision/Recall/NDCG without user ground truth).

METRIC REGISTRY:
    - mean_vector_distance: Mean Euclidean distance between recommended tracks and target user profile.
    - ranking_monotonicity_score: Proportion of consecutive recommendation pairs ordered by descending score.
    - intra_list_diversity: Average pairwise Euclidean distance among recommended tracks.
    - cluster_coverage: Proportion of acoustic clusters represented in recommendation set.
    - retrieval_hit_rate_at_k: Binary hit score (1.0 if expected PMID is in top-K retrieved chunks).
    - mean_reciprocal_rank: Reciprocal rank (1/rank) of first expected PMID in retrieval list.
    - citation_grounding_accuracy: Proportion of cited PMIDs that exist in retrieved evidence package.
    - track_grounding_accuracy: Proportion of referenced track IDs that exist in candidate set.
    - safety_compliance_score: Binary compliance score (1.0 if no prohibited clinical phrases detected).
"""

import math
from typing import List, Dict, Any, Optional


def mean_vector_distance(
    recommendations: List[Dict[str, Any]],
    target_features: Dict[str, float],
    feature_keys: Optional[List[str]] = None
) -> float:
    """
    Calculate mean Euclidean distance between recommended tracks and target audio feature profile.

    Definition: Average $L_2$ norm between track feature vectors and target profile feature vector.
    Interpretation: Lower values indicate closer alignment with target acoustic preferences.
    Limitation: Assumes Euclidean distance is meaningful across unscaled or standardized feature dimensions.
    """
    if not recommendations or not target_features:
        return 0.0

    keys = feature_keys or ["energy", "tempo", "acousticness", "danceability", "valence", "instrumentalness"]
    total_dist = 0.0
    valid_tracks = 0

    for track in recommendations:
        dist_sq = 0.0
        match_count = 0
        for k in keys:
            if k in track and k in target_features:
                diff = float(track[k]) - float(target_features[k])
                dist_sq += diff * diff
                match_count += 1
        if match_count > 0:
            total_dist += math.sqrt(dist_sq)
            valid_tracks += 1

    return round(total_dist / valid_tracks, 4) if valid_tracks > 0 else 0.0


def ranking_monotonicity_score(
    recommendations: List[Dict[str, Any]],
    score_key: str = "final_score"
) -> float:
    """
    Measure proportion of consecutive recommendation pairs ordered by descending score.

    Definition: Ratio of pairs (i, i+1) where score[i] >= score[i+1].
    Interpretation: 1.0 indicates perfect descending score ordering; < 1.0 indicates ranking violations.
    Limitation: Only evaluates pairwise monotonicity, not absolute score calibration.
    """
    if len(recommendations) <= 1:
        return 1.0

    valid_pairs = 0
    total_pairs = len(recommendations) - 1

    for i in range(total_pairs):
        s1 = float(recommendations[i].get(score_key, 0.0))
        s2 = float(recommendations[i + 1].get(score_key, 0.0))
        if s1 >= s2 - 1e-9:
            valid_pairs += 1

    return round(valid_pairs / total_pairs, 4)


def intra_list_diversity(
    recommendations: List[Dict[str, Any]],
    feature_keys: Optional[List[str]] = None
) -> float:
    """
    Calculate average pairwise Euclidean distance among recommended tracks.

    Definition: Mean distance between all unique pairs of tracks in recommendation list.
    Interpretation: Higher values indicate greater acoustic diversity among recommendations.
    Limitation: High diversity may conflict with strict similarity targeting.
    """
    if len(recommendations) <= 1:
        return 0.0

    keys = feature_keys or ["energy", "tempo", "acousticness", "danceability", "valence", "instrumentalness"]
    total_dist = 0.0
    pair_count = 0

    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            t1 = recommendations[i]
            t2 = recommendations[j]
            dist_sq = 0.0
            for k in keys:
                if k in t1 and k in t2:
                    diff = float(t1[k]) - float(t2[k])
                    dist_sq += diff * diff
            total_dist += math.sqrt(dist_sq)
            pair_count += 1

    return round(total_dist / pair_count, 4) if pair_count > 0 else 0.0


def cluster_coverage(
    recommendations: List[Dict[str, Any]],
    cluster_key: str = "cluster_label",
    n_total_clusters: int = 4
) -> float:
    """
    Calculate ratio of distinct acoustic clusters represented in recommendation list.

    Definition: Count of unique cluster IDs present in top-N recommendations divided by total cluster count K.
    Interpretation: Measures portfolio coverage across K-Means acoustic clusters.
    Limitation: User preference profiles may intentionally target a single cluster.
    """
    if not recommendations or n_total_clusters <= 0:
        return 0.0

    observed_clusters = set()
    for trk in recommendations:
        c_id = trk.get(cluster_key)
        if c_id is not None:
            observed_clusters.add(c_id)

    return round(len(observed_clusters) / float(n_total_clusters), 4)


def retrieval_hit_rate_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_pmids: List[str],
    k: int = 2
) -> float:
    """
    Calculate binary Hit Rate @ K for RAG retrieval over controlled benchmark queries.

    Definition: 1.0 if at least one expected PMID appears in top-K retrieved chunks; 0.0 otherwise.
    Interpretation: Measures whether relevant research is successfully retrieved in top-K results.
    Limitation: Evaluated on controlled benchmark query set; not a full corpus annotation.
    """
    if not retrieved_chunks or not expected_pmids:
        return 0.0

    top_k_chunks = retrieved_chunks[:k]
    expected_set = set(str(p).strip() for p in expected_pmids)

    for chunk in top_k_chunks:
        meta = chunk.get("metadata", {})
        pmid = str(meta.get("pmid") or chunk.get("pmid", "")).strip()
        if pmid in expected_set:
            return 1.0

    return 0.0


def mean_reciprocal_rank(
    retrieved_chunks: List[Dict[str, Any]],
    expected_pmids: List[str]
) -> float:
    """
    Calculate Reciprocal Rank (1 / rank) of first expected PMID in retrieval results.

    Definition: 1 / rank of first relevant chunk (1-indexed); 0.0 if no expected PMID is retrieved.
    Interpretation: Higher values indicate relevant research chunks are ranked closer to rank 1.
    Limitation: Only evaluates first relevant hit position.
    """
    if not retrieved_chunks or not expected_pmids:
        return 0.0

    expected_set = set(str(p).strip() for p in expected_pmids)

    for rank_idx, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})
        pmid = str(meta.get("pmid") or chunk.get("pmid", "")).strip()
        if pmid in expected_set:
            return round(1.0 / float(rank_idx), 4)

    return 0.0


def citation_grounding_accuracy(
    explanation_sources: List[Dict[str, Any]],
    evidence_sources: List[Dict[str, Any]]
) -> float:
    """
    Calculate proportion of cited PMIDs in ExplanationResponse that exist in retrieved EvidencePackage.

    Definition: (Count of cited PMIDs present in EvidencePackage) / (Total cited PMIDs in Explanation).
    Interpretation: 1.0 indicates 100% citation grounding; < 1.0 indicates fabricated citations.
    Limitation: Evaluates PMID match; does not analyze semantic alignment of text.
    """
    if not explanation_sources:
        return 1.0  # Zero citations made, zero ungrounded citations

    valid_pmids = set(str(s.get("pmid", "")).strip() for s in evidence_sources if s.get("pmid"))
    if not valid_pmids:
        # If evidence package has no sources, any citation is ungrounded
        return 0.0

    grounded_count = 0
    total_cited = 0

    for src in explanation_sources:
        pmid = str(src.get("pmid", "")).strip()
        if pmid:
            total_cited += 1
            if pmid in valid_pmids:
                grounded_count += 1

    return round(grounded_count / total_cited, 4) if total_cited > 0 else 1.0


def track_grounding_accuracy(
    recommendation_reasons: List[str],
    valid_track_ids: List[str]
) -> float:
    """
    Calculate proportion of track IDs referenced in recommendation reasons that exist in candidate set.

    Definition: (Count of valid track ID references) / (Total track ID references).
    Interpretation: 1.0 indicates all track references exist in output recommendations.
    Limitation: Regex-based track ID extraction.
    """
    import re
    if not recommendation_reasons:
        return 1.0

    valid_set = set(str(t).strip() for t in valid_track_ids)
    total_refs = 0
    valid_refs = 0

    for reason in recommendation_reasons:
        matches = re.findall(r"ID:\s*([A-Za-z0-9_\-]+)", reason)
        for trk_id in matches:
            total_refs += 1
            if trk_id in valid_set:
                valid_refs += 1

    return round(valid_refs / total_refs, 4) if total_refs > 0 else 1.0


def safety_compliance_score(
    full_text: str,
    prohibited_terms: Optional[List[str]] = None
) -> float:
    """
    Calculate non-clinical safety compliance score.

    Definition: 1.0 if zero prohibited clinical phrases are detected in text; 0.0 if any are present.
    Interpretation: Binary safety enforcement indicator.
    Limitation: Keyword string matching does not perform deep semantic safety analysis.
    """
    terms = prohibited_terms or [
        "diagnose anxiety",
        "cures anxiety",
        "treats anxiety",
        "clinical treatment",
        "medical cure",
        "prescribes music",
        "psychiatric therapy"
    ]
    lower_text = full_text.lower()
    for t in terms:
        if t in lower_text:
            return 0.0
    return 1.0
