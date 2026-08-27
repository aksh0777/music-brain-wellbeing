"""
Spotify Web API Client Module

WHAT: Encapsulates HTTP GET requests to Spotify Web API endpoints (`/v1/me`, `/v1/me/player/recently-played`,
`/v1/me/player/currently-playing`, `/v1/tracks`).

WHY: Isolates network requests, HTTP status handling, rate limit retries (429), and automatic token refreshes (401)
from data mapping and recommendation logic.
"""

import time
import requests
from typing import Dict, Any, Optional
from .spotify_auth import SpotifyAuth, SpotifyAuthError


class SpotifyAPIError(Exception):
    """Base exception for Spotify Web API HTTP failures."""
    pass


class SpotifyClient:
    """
    Spotify Web API HTTP Client.

    WHAT: Executes authenticated requests to Spotify endpoints with exponential backoff and rate-limit handling.

    WHY: Provides clean application functions for retrieving real user listening data.
    """

    BASE_URL = "https://api.spotify.com/v1"

    def __init__(self, auth: Optional[SpotifyAuth] = None):
        self.auth = auth or SpotifyAuth()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Execute an authenticated GET request with rate-limit and 401 retry handling.

        WHAT: Adds Authorization Bearer header, checks HTTP status, handles 429 Retry-After, and returns JSON.

        WHY: Protects application pipelines against transient network errors and API rate limits.
        """
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        token = self.auth.get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise SpotifyAPIError(f"Network error communicating with Spotify API: {str(e)}")
                time.sleep(2 ** attempt)
                continue

            # Token expired (401) -> refresh token and retry
            if response.status_code == 401:
                try:
                    token = self.auth.refresh_access_token()
                    headers["Authorization"] = f"Bearer {token}"
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                except SpotifyAuthError as ae:
                    raise SpotifyAPIError(f"Authentication refresh failed during API call: {str(ae)}")

            # Rate limited (429) or Server error (5xx)
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            wait_time = int(retry_after)
                    time.sleep(wait_time)
                    continue

            # 204 No Content (e.g. currently playing when player is idle)
            if response.status_code == 204:
                return {}

            if response.status_code != 200:
                raise SpotifyAPIError(
                    f"Spotify API request to {endpoint} failed (HTTP {response.status_code}): {response.text}"
                )

            return response.json()

        raise SpotifyAPIError(f"Exceeded max retries for Spotify API endpoint {endpoint}")

    def get_user_profile(self) -> Dict[str, Any]:
        """Fetch the current user's profile metadata (/v1/me)."""
        return self._make_request("me")

    def get_recently_played(
        self,
        limit: int = 50,
        after_timestamp_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch the current user's recently played tracks (/v1/me/player/recently-played).

        WHAT: Retrieves up to 50 recently played tracks.

        WHY: Serves as the primary real user listening history source for sessionization and user profiling.
        """
        params = {"limit": min(limit, 50)}
        if after_timestamp_ms is not None:
            params["after"] = after_timestamp_ms

        return self._make_request("me/player/recently-played", params=params)

    def get_currently_playing(self) -> Optional[Dict[str, Any]]:
        """Fetch the current user's currently playing track (/v1/me/player/currently-playing)."""
        res = self._make_request("me/player/currently-playing")
        return res if res else None

    def get_track(self, track_id: str) -> Dict[str, Any]:
        """Fetch single track metadata (/v1/tracks/{id})."""
        return self._make_request(f"tracks/{track_id}")
