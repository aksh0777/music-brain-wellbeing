"""
Spotify OAuth 2.0 Authentication Module

WHAT: Manages Spotify OAuth 2.0 Authorization Code flow, token exchange, access token caching,
and refresh token handling.

WHY: Spotify user data (recently played tracks, user profile) is private. Accessing private user data requires
an authorized OAuth 2.0 flow. Credentials and tokens must be kept secure in environment variables.

OAUTH 2.0 FIRST-PRINCIPLES FLOW:
    User -> Application -> Spotify Auth URL -> User Grants Scope Permission
    -> Spotify redirects with Auth Code -> Application exchanges Auth Code for Access & Refresh Tokens
    -> Application calls API endpoints with Bearer Access Token.
"""

import os
import time
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlencode


class SpotifyAuthError(Exception):
    """Base exception for Spotify authentication errors."""
    pass


class SpotifyReauthorizationRequired(SpotifyAuthError):
    """Raised when a Spotify refresh token is expired, revoked, or rejected with invalid_grant."""
    pass


class SpotifyAuth:
    """
    Handles Spotify OAuth 2.0 authentication and token management.

    WHAT: Manages client credentials, authorization URL generation, token exchange, and token refresh.

    WHY: Decouples OAuth token mechanics from API data fetching and recommendation logic.
    """

    TOKEN_URL = "https://accounts.spotify.com/api/token"
    AUTH_URL = "https://accounts.spotify.com/authorize"
    TOKEN_BUFFER_SECONDS = 60

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        refresh_token: Optional[str] = None
    ):
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
        self.refresh_token = refresh_token or os.getenv("SPOTIFY_REFRESH_TOKEN")

        self.access_token: Optional[str] = None
        self.token_expires_at: float = 0.0

    def validate_credentials(self) -> bool:
        """
        Validate that required client credentials are standard non-empty strings.

        WHAT: Checks client_id and client_secret presence.

        WHY: Prevents launching OAuth requests with missing or placeholder environment variables.
        """
        if not self.client_id or "your_spotify" in str(self.client_id):
            raise SpotifyAuthError("SPOTIFY_CLIENT_ID missing or unconfigured in environment variables.")
        if not self.client_secret or "your_spotify" in str(self.client_secret):
            raise SpotifyAuthError("SPOTIFY_CLIENT_SECRET missing or unconfigured in environment variables.")
        return True

    def get_authorization_url(self, scope: str = "user-read-recently-played user-read-currently-playing") -> str:
        """
        Generate the Spotify OAuth 2.0 authorization URL.

        WHAT: Constructs user sign-in URL with requested scope permissions.

        WHY: User must be redirected to Spotify's authorization server to grant application permissions.
        """
        self.validate_credentials()
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "show_dialog": "true"
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    def exchange_code_for_tokens(self, auth_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token and refresh token.

        WHAT: Sends POST request to Spotify token endpoint.

        WHY: The authorization code is short-lived; it must be exchanged for a Bearer access token.
        """
        self.validate_credentials()
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri
        }

        try:
            response = requests.post(
                self.TOKEN_URL,
                auth=(self.client_id, self.client_secret),
                data=data,
                timeout=30
            )
        except requests.RequestException as e:
            raise SpotifyAuthError(f"Network error during code exchange: {str(e)}")

        if response.status_code != 200:
            raise SpotifyAuthError(f"Token exchange failed (HTTP {response.status_code}): {response.text}")

        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token", self.refresh_token)
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = time.time() + expires_in - self.TOKEN_BUFFER_SECONDS

        return token_data

    def refresh_access_token(self) -> str:
        """
        Refresh access token using stored refresh token.

        WHAT: Sends POST request with refresh_token to obtain a new access_token.

        WHY: Access tokens expire after 1 hour. Refresh tokens allow automated token renewal.
        """
        self.validate_credentials()
        if not self.refresh_token:
            raise SpotifyAuthError("No refresh token available to perform access token refresh.")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }

        try:
            response = requests.post(
                self.TOKEN_URL,
                auth=(self.client_id, self.client_secret),
                data=data,
                timeout=30
            )
        except requests.RequestException as e:
            raise SpotifyAuthError(f"Network error during token refresh: {str(e)}")

        if response.status_code != 200:
            error_code = None
            try:
                error_data = response.json()
                error_code = error_data.get("error")
            except Exception:
                pass

            if error_code == "invalid_grant" or response.status_code == 400:
                raise SpotifyReauthorizationRequired(
                    "Spotify rejected refresh token (invalid_grant). The token is expired or revoked."
                )
            raise SpotifyAuthError(f"Failed to refresh access token (HTTP {response.status_code}): {response.text}")

        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token", self.refresh_token)
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = time.time() + expires_in - self.TOKEN_BUFFER_SECONDS

        return self.access_token

    def get_access_token(self) -> str:
        """
        Retrieve a valid access token, performing automatic refresh if expired.

        WHAT: Returns cached token or triggers refresh if expired.

        WHY: Ensures downstream client requests always use an unexpired Bearer token.
        """
        if not self.access_token or time.time() >= self.token_expires_at:
            if self.refresh_token:
                self.refresh_access_token()
            else:
                raise SpotifyAuthError("No valid access token or refresh token set.")
        return self.access_token
