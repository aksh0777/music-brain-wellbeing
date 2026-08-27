# Chapter 14 — Spotify API Integration Layer & Adapter Architecture

## 1. Executive Summary & Why Spotify Integration Exists
This document details the first principles, OAuth 2.0 security design, data mapping adapters, and pipeline coordination of the **Spotify API Integration Layer** (Phase 3). 

While Phases 1 and 2 established our offline Music Intelligence and Recommendation Engine layers using synthetic/demo data, Phase 3 connects our system to **real-world external Spotify Web API streams**. By translating external API payloads into our internal application schemas, we prove that our completed machine learning and recommendation pipelines can consume real user music streams without modifying downstream code.

---

## 2. OAuth 2.0 Authentication from First Principles

### 1. What Problem Does It Solve?
Private user Spotify data (recently played tracks, user profiles) requires explicit user authorization before an application can access it.

### 2. First-Principles Intuition
OAuth 2.0 acts like a hotel keycard: instead of sharing the user's Spotify master password with our application, the user grants a temporary, scope-restricted keycard (Bearer Access Token) to our app.

### 3. Data Flow Architecture

```text
User ──(1. Sign-In Request)──> Application ──(2. Redirect with Scope)──> Spotify Authorization Page
                                                                                  │
User Grants Permission <──────────────────────────────────────────────────────────┘
       │
       └──(3. Auth Code Redirect)──> Application ──(4. Exchange Code)──> Spotify Token Server
                                                                                  │
Application <──(5. Returns Access & Refresh Tokens)───────────────────────────────┘
       │
       └──(6. API Request with Bearer Token)──> Spotify Web API Endpoints
```

---

## 3. Package Architecture & Data Mapping (`src/spotify/`)

The integration package strictly isolates external network calls and API schemas:
- `spotify_auth.py`: Manages credentials (`SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`), authorization URL construction, code exchange, access token caching, and automated token refreshes.
- `spotify_client.py`: Encapsulates HTTP GET requests (`/v1/me`, `/v1/me/player/recently-played`, `/v1/tracks`), enforcing exponential backoff retries for rate limits (HTTP 429 `Retry-After`) and automatic 401 token refreshes.
- `spotify_mapper.py`: Translates raw nested Spotify JSON into internal DataFrames matching `REQUIRED_HISTORY_COLUMNS` (`event_id`, `user_id`, `track_id`, `played_at`, `data_type`, `source`). Also provides the `AudioFeatureProvider` fallback adapter.
- `spotify_pipeline.py`: Coordinates `REAL` vs `DEMO` execution modes, linking mapped streams to Phase 1 user profiling and Phase 2 recommendation ranking.

---

## 4. Internal Data Schemas & Provenance Explicit Tagging

| Metric | Real Spotify API Data | Synthetic / Demo Data |
| :--- | :--- | :--- |
| **`source`** | `"spotify_api"` | `"synthetic"` |
| **`data_type`** | `"real"` | `"synthetic/demo"` |
| **`user_id`** | `"USR_SPOTIFY_LIVE"` | `"USR001"` |
| **`played_at`** | ISO 8601 UTC timestamp from API | Simulated timestamp |

> [!IMPORTANT]
> **Data Provenance Rule**: Real API data and synthetic demo logs are never mixed implicitly. Provenance fields (`source` and `data_type`) explicitly tag data lineage across DataFrames, CSV exports, and user profiles.

---

## 5. Spotify API Access Limitations

1. **Observed Window Constraint**: The GET `/v1/me/player/recently-played` endpoint retrieves only up to 50 recent play events ($\approx$ 2–3 days of active listening). The system treats retrieved stream data as an **observed time window**, not an unlimited lifetime archive.
2. **Audio-Features Access Policy**: Spotify API policies restrict bulk audio feature endpoints for non-extended commercial developer apps. The `AudioFeatureProvider` adapter cleanly isolates this boundary, allowing reference catalogs or local audio feature datasets to enrich stream logs without crashing downstream pipelines.

---

## 6. Security & Credential Governance

- **Zero Hardcoded Secrets**: Credentials are read exclusively from environment variables (`os.getenv`).
- **Git Safety**: `.env` and `*.env` patterns are added to `.gitignore`. Only `.env.example` containing placeholder variable names is tracked in version control.
- **Server-Side Token Storage**: Refresh tokens are kept in secure environment variables or repository Secrets, never logged or printed.

---

## 7. Error Handling & Resilience Matrix

| Error Scenario | HTTP Status / Exception | Recovery Strategy |
| :--- | :--- | :--- |
| **Expired Access Token** | HTTP 401 Unauthorized | `SpotifyClient` catches 401 and calls `SpotifyAuth.refresh_access_token()` automatically. |
| **Invalid Refresh Token** | HTTP `invalid_grant` / 400 | Raises `SpotifyReauthorizationRequired`, halting pipeline and requesting user sign-in. |
| **Rate Limit Exceeded** | HTTP 429 Too Many Requests | Inspects `Retry-After` header and sleeps for requested backoff duration before retrying. |
| **Server Failure** | HTTP 500, 502, 503, 504 | Performs exponential backoff retries ($2^{\text{attempt}}$ seconds). |
| **Empty Stream Log** | 204 No Content or `{}` | `map_recently_played_to_internal` returns clean empty DataFrame without crashing. |

---

## 8. Connecting Phase 3 to Phase 1 & Phase 2

```text
Spotify API (Live / Mock JSON)
       ↓
src/spotify/spotify_mapper.py (Internal Schema Conversion)
       ↓
src/features/sessions.py (30-min Inactivity Gap Partitioning)
       ↓
src/features/temporal.py (Cyclical Sin/Cos Hour Encodings)
       ↓
src/features/user_profile.py (Quantitative User Profile Vector)
       ↓
src/recommendation/recommender.py (Top-N Standardized Euclidean Recommendation Engine)
       ↓
Top-N Recommendations + Deterministic Explanations
```

---

## 9. Future Production Architecture & Placements

### A. FastAPI Backend Placement (Phase 5)
FastAPI will wrap our Python modules into REST endpoints:
- `POST /auth/login`: Redirects user to Spotify OAuth sign-in.
- `GET /auth/callback`: Handles authorization code exchange.
- `GET /user/profile`: Returns quantitative user music profile JSON.
- `POST /recommend`: Serves top-N personalized recommendations.

### B. PostgreSQL Memory Placement (Phase 5)
PostgreSQL will serve as persistent application memory:
- `users`: User profiles and OAuth refresh tokens.
- `listening_events`: Historical stream logs appended across API ingestion runs.
- `track_catalog`: Permanent track catalog and acoustic feature embeddings.
