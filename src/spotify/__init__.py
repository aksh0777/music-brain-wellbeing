"""
Spotify API Integration Package

WHAT: Secure OAuth authentication, Web API client, data mapping adapter, and pipeline coordinator
for integrating Spotify user listening streams into the Music Brain Wellbeing System.

WHY: Isolates external Spotify Web API authentication, rate limiting, error handling, and JSON response mapping
from downstream machine learning and recommendation engine layers.
"""

from .spotify_auth import SpotifyAuth, SpotifyAuthError, SpotifyReauthorizationRequired
from .spotify_client import SpotifyClient, SpotifyAPIError
from .spotify_mapper import map_recently_played_to_internal, AudioFeatureProvider
from .spotify_pipeline import run_spotify_pipeline

__all__ = [
    "SpotifyAuth",
    "SpotifyAuthError",
    "SpotifyReauthorizationRequired",
    "SpotifyClient",
    "SpotifyAPIError",
    "map_recently_played_to_internal",
    "AudioFeatureProvider",
    "run_spotify_pipeline"
]
