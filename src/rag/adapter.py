"""
Recommendation Engine to Research Query Adapter Module

WHAT: Converts quantitative music profiles and recommended track acoustic features into academic research queries.

WHY: Bridges the Recommendation Engine (Phase 2 & 3) with the Research Retrieval Layer (Phase 4).
Maps track descriptors (energy, valence, tempo, acousticness) into domain search terms.

IMPORTANT SCIENTIFIC BOUNDARY:
    - This adapter maps acoustic attributes to literature search concepts.
    - It does NOT make clinical predictions or claim that playing a track will cure or treat anxiety.
    - Maintains a clear operational boundary between OBSERVED USER BEHAVIOR and SCIENTIFIC LITERATURE EVIDENCE.
"""

from typing import Dict, Any, Optional


class RecommendationQueryAdapter:
    """
    Translates user acoustic profiles and recommendation metadata into scientific research queries.
    """

    @staticmethod
    def construct_query_from_profile(
        user_profile: Dict[str, Any],
        recommended_track: Optional[Dict[str, Any]] = None,
        target_topic: Optional[str] = None
    ) -> str:
        """
        Construct an academic research query based on user profile and recommended track features.

        Args:
            user_profile: Quantitative user profile dictionary (from build_user_music_profile).
            recommended_track: Optional recommended track series/dict containing audio features.
            target_topic: Optional specific scientific topic override (e.g. 'anxiety', 'stress', 'autonomic').

        Returns:
            Formatted research query string.
        """
        query_parts = []

        # 1. Extract target topic or default focus
        if target_topic:
            query_parts.append(target_topic)
        else:
            query_parts.append("music interventions emotion regulation stress reduction")

        # 2. Extract acoustic feature traits from track or user profile
        energy = None
        tempo = None
        acousticness = None

        if recommended_track:
            energy = recommended_track.get("energy")
            tempo = recommended_track.get("tempo")
            acousticness = recommended_track.get("acousticness")

        # Fallback to audio profile summary if track features not provided
        if energy is None and "audio_feature_summary" in user_profile:
            energy = user_profile["audio_feature_summary"].get("energy_mean")
            tempo = user_profile["audio_feature_summary"].get("tempo_mean")
            acousticness = user_profile["audio_feature_summary"].get("acousticness_mean")

        # 3. Translate numerical features into academic acoustic terms
        acoustic_terms = []
        if energy is not None:
            if energy <= 0.4:
                acoustic_terms.append("low energy soothing acoustics")
            elif energy >= 0.7:
                acoustic_terms.append("high energy uplifting arousal")
            else:
                acoustic_terms.append("moderate energy acoustic structure")

        if tempo is not None:
            if tempo < 100:
                acoustic_terms.append("slow tempo rhythmic stability")
            elif tempo > 120:
                acoustic_terms.append("fast tempo auditory stimulation")

        if acousticness is not None and acousticness >= 0.5:
            acoustic_terms.append("high acousticness instrumental timbre")

        if acoustic_terms:
            query_parts.append(" ".join(acoustic_terms))

        # 4. Add core domain context terms
        query_parts.append("physiological arousal stress recovery individual differences")

        # Combine into clean query string
        full_query = " ".join(query_parts).strip()
        return full_query
