"""
Candidate Retrieval Module

WHAT: Filters and selects a candidate pool of tracks from the music catalog for recommendation ranking.

WHY: Production recommendation architectures decouple candidate retrieval (selecting potentially relevant tracks)
from candidate ranking (scoring and ordering candidates). This multi-stage pipeline scales efficiently.

FIRST-PRINCIPLES INTUITION:
Instead of running expensive distance calculations across millions of global tracks, a candidate generator
quickly retrieves a manageable subset (e.g. 500 catalog tracks, or genre/context-filtered subsets) for ranking.
"""

import pandas as pd
from typing import Dict, Any, Optional


def retrieve_candidates(
    catalog_df: pd.DataFrame,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Retrieve candidate tracks from the catalog based on optional contextual filters.

    WHAT: Accepts a catalog DataFrame and returns candidate tracks matching criteria (e.g. genre, explicit, tempo bounds).

    WHY: Decouples candidate selection from similarity scoring, allowing context-aware pre-filtering.
    """
    if catalog_df.empty:
        return catalog_df.copy()

    candidates = catalog_df.copy()

    if filter_criteria:
        # Genre filter
        if "genres" in filter_criteria and filter_criteria["genres"]:
            candidates = candidates[candidates["genre"].isin(filter_criteria["genres"])]

        # Explicit content filter
        if "exclude_explicit" in filter_criteria and filter_criteria["exclude_explicit"]:
            if "explicit" in candidates.columns:
                candidates = candidates[candidates["explicit"] == False]

        # Tempo range filter
        if "min_tempo" in filter_criteria:
            candidates = candidates[candidates["tempo"] >= filter_criteria["min_tempo"]]
        if "max_tempo" in filter_criteria:
            candidates = candidates[candidates["tempo"] <= filter_criteria["max_tempo"]]

    return candidates.reset_index(drop=True)
