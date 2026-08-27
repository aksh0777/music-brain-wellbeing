# Data Sources & Provenance Specification

This document records data origins, schemas, synthetic/real status, licensing considerations, and scientific usage boundaries across all datasets in the **Music, Brain & Wellbeing** repository.

---

## 1. Primary Research Dataset: MXMH Survey Data

- **Dataset Name**: Music & Mental Health Survey (MXMH) Results
- **File Location**: `data/raw/mxmh_survey_results.csv` & `data/processed/mxmh_cleaned.csv`
- **Source**: Public Kaggle MXMH Survey (736 records)
- **Status**: **REAL Observational Survey Data**
- **Purpose**: Predictive modeling of self-reported anxiety scores ($Y \in [0, 10]$) based on daily listening hours, preferred genres, and self-reported BPM.
- **Important Columns**: `Age`, `Primary genre`, `Hours per day`, `BPM`, `Anxiety`, `Depression`, `Insomnia`, `OCD`.
- **Scientific Usability**: Cross-sectional observational data; suitable for correlation and predictive regression analysis. **Cannot establish causality or clinical medical diagnoses.**
- **Repository Policy**: Kept completely untouched during Music Intelligence Layer extensions.

---

## 2. Music Intelligence Dataset: Spotify Track Catalog

- **Dataset Name**: Spotify Track Catalog Audio Features
- **File Location**: `data/raw/spotify/tracks.csv` & `data/processed/music/tracks_cleaned.csv`
- **Source**: Representative Spotify Audio Feature Catalog (500 tracks across 10 genres)
- **Status**: **Prepared Track Catalog Data (Spotify Schema Conforming)**
- **Purpose**: Unsupervised acoustic profile clustering, feature scaling, and similarity matching.
- **Important Columns**: `track_id`, `track_name`, `artist_name`, `genre`, `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`.
- **Scientific Usability**: Valid for physical and perceptual audio structure representation and clustering. Does not contain psychological survey responses.

---

## 3. Behavioral Dataset: Timestamped Listening History Log

- **Dataset Name**: Synthetic User Listening Stream History
- **File Location**: `data/processed/music/listening_history.csv`
- **Source**: Programmatically generated demo stream based on track catalog (1,000 events)
- **Status**: **SYNTHETIC / DEMO DATA ONLY (`data_type = "synthetic/demo"`)**
- **Purpose**: Development, testing, and validation of 30-minute sessionization, temporal encodings, and user music profile generation.
- **Important Columns**: `event_id`, `user_id`, `track_id`, `played_at`, `data_type`.
- **Scientific Usability**: **DEVELOPMENT & DEMO ONLY.** Must NEVER be presented as real-world human behavior evidence or cited in scientific findings.
- **Repository Policy**: Explicitly labeled `data_type = "synthetic/demo"` in code, CSV logs, and JSON outputs.

---

## 4. Architectural Reference Attribution

- **Repository**: [spotify-brain](https://github.com/keremburakyilmaz/spotify-brain)
- **Author**: Kerem Burak Yılmaz
- **License**: MIT License (Copyright © 2025 Kerem Burak Yılmaz)
- **Usage**: Reference architecture for session gap thresholding, tempo normalization, silhouette-guided K-Means, and cyclical temporal encodings.
