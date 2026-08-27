# Spotify Brain Architectural Analysis & Adaptation Guide

> **Reference Repository**: [keremburakyilmaz/spotify-brain](https://github.com/keremburakyilmaz/spotify-brain) (MIT License, Copyright © 2025 Kerem Burak Yılmaz)
> **Purpose**: Document how our **Music Brain Wellbeing Intelligence System** selectively adapts scalable engineering concepts from `spotify-brain` without copying code, introducing live API dependencies, or overclaiming clinical diagnoses.

---

## 1. Executive Summary

`spotify-brain` is an open-source machine learning system that ingests personal Spotify listening streams, partitions tracks into sessions, clusters tracks into mood groups using K-Means, and predicts next-track mood clusters and session start times.

We use `spotify-brain` as an **architectural reference** to build our Phase 1 **Music Data + Listening Intelligence Foundation**. We adapt its feature engineering, session gap thresholding, tempo normalization, silhouette-guided K-Means, and cyclical temporal encodings into our existing project.

---

## 2. Adaptation Matrix

| Architectural Concept | What `spotify-brain` Does | Why It Is Useful | How We Adapt It | Our Differences & Safeguards |
| :--- | :--- | :--- | :--- | :--- |
| **Track Audio Features** | Ingests 8 Spotify audio descriptors (`valence`, `energy`, `danceability`, `acousticness`, `instrumentalness`, `tempo`, etc.) | Provides standardized numerical vector representations for audio similarity. | We implement `src/features/music_features.py` with range checks ($[0,1]$). | We keep raw data untouched and maintain strict schema validation without live OAuth. |
| **Session Boundary** | Applies a 30-minute inactivity gap rule ($\Delta t > 30\text{ mins}$) | Grouping tracks into continuous behavioral episodes (e.g. commute, focus). | We implement `src/features/sessions.py` with configurable `gap_minutes = 30`. | We document sessionization as a behavioral heuristic, not a psychological boundary. |
| **Tempo Normalization** | Computes `tempo_norm = tempo / 250.0` | Prevents raw BPM ($40-220$) from dominating $[0,1]$ audio features. | Combined with `StandardScaler` in `src/features/music_features.py`. | Integrated into a reusable scikit-learn standard scaling pipeline. |
| **K-Means Clustering** | Fits K-Means on 6 audio features, testing $K \in [3, 15]$ using Silhouette & Elbow metrics | Discovers natural acoustic groupings in track catalogs without manual genre reliance. | We implement `src/features/clustering.py` testing $K \in [2, 8]$. | **Crucial Distinction**: We label clusters as **"Acoustic Profiles"**, strictly rejecting clinical/mood diagnoses. |
| **Distance Assignment** | Assigns new tracks to nearest centroid via Euclidean distance | Enables fast incremental track labeling without re-fitting K-Means. | Implemented as `assign_nearest_cluster()` in `clustering.py`. | Distance calculations operate on standardized feature arrays. |
| **Cyclical Encoding** | Transforms hour and day of week into sine/cosine pairs | Eliminates boundary discontinuities ($23:00 \leftrightarrow 00:00$ and $\text{Sun} \leftrightarrow \text{Mon}$). | We implement `src/features/temporal.py` using unit circle equations. | Preserves continuous distance relationships across midnight. |
| **User Music Profiling** | Single track history sequence | Aggregates listening events into user-level vectors. | We implement `src/features/user_profile.py`. | Computes comprehensive means, stds, peak hours, and cluster shares. |

---

## 3. What We Deliberately Omitted / Rejected

1. **ReccoBeats API Fallback**: `spotify-brain` falls back to an external 3rd party API for missing features. We omit this to prevent external API point-of-failure vulnerabilities.
2. **Clinical Mood Labels**: `spotify-brain` calls clusters "sad", "happy", or "anxious". We strictly reject this to prevent scientific overreach.
3. **Live Spotify OAuth Ingestion**: We avoid requiring live Spotify user authentication tokens for development and unit testing, ensuring full offline testability.

---

## 4. Architectural Layer Integration

```text
[Existing MXMH Survey ML Foundation]
  └── Self-Reported Anxiety Prediction Model (max_depth=3 Decision Tree, R² = 0.0636)
           │
           │ (Decoupled Layer Integration)
           ▼
[Music Data & Listening Intelligence Foundation] (THIS PHASE)
  ├── Track Catalog Ingestion (data/raw/spotify/tracks.csv)
  ├── Synthetic Listening Log (data/processed/music/listening_history.csv)
  ├── Audio Feature Validation & Scaling (src/features/music_features.py)
  ├── 30-Min Session Partitioning (src/features/sessions.py)
  ├── Cyclical Temporal Encodings (src/features/temporal.py)
  ├── K-Means Acoustic Clustering (src/features/clustering.py)
  └── Quantitative User Profiler (src/features/user_profile.py)
           │
           ▼
[Future Phases]
  ├── Phase 2: Personalized Recommendation Engine
  ├── Phase 3: Research-Grounded RAG Literature Layer
  ├── Phase 4: Non-Clinical LLM Explanation Layer
  └── Phase 5: FastAPI Serving Backend & Dashboard UI
```

---

## 5. Licensing & Attribution Notice

The engineering patterns adapted from `spotify-brain` are licensed under the **MIT License**:
> Copyright (c) 2025 Kerem Burak Yılmaz  
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files...

Our implementation is an independent, clean-room python package built specifically for the **Music, Brain & Wellbeing** system architecture.
