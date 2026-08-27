import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.spotify.spotify_auth import SpotifyAuth, SpotifyAuthError, SpotifyReauthorizationRequired
from src.spotify.spotify_client import SpotifyClient, SpotifyAPIError
from src.spotify.spotify_mapper import map_recently_played_to_internal, AudioFeatureProvider
from src.spotify.spotify_pipeline import run_spotify_pipeline, generate_mock_spotify_recently_played


class TestSpotifyIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_spotify_json = {
            "items": [
                {
                    "track": {
                        "id": "TRK0001",
                        "name": "Ambient Soundscape",
                        "artists": [{"name": "Acoustic Artist"}],
                        "album": {"name": "Mindful Moments"},
                        "duration_ms": 210000
                    },
                    "played_at": "2026-08-27T10:00:00Z"
                },
                {
                    "track": {
                        "id": "TRK0002",
                        "name": "Upbeat Pulse",
                        "artists": [{"name": "Rhythmic Beats"}],
                        "album": {"name": "Energy Boost"},
                        "duration_ms": 180000
                    },
                    "played_at": "2026-08-27T10:05:00Z"
                }
            ]
        }

        self.catalog_df = pd.DataFrame({
            "track_id": ["TRK0001", "TRK0002"],
            "track_name": ["Ambient Soundscape", "Upbeat Pulse"],
            "artist_name": ["Acoustic Artist", "Rhythmic Beats"],
            "album_name": ["Mindful Moments", "Energy Boost"],
            "genre": ["Ambient", "Pop"],
            "energy": [0.2, 0.8],
            "valence": [0.3, 0.7],
            "danceability": [0.3, 0.8],
            "acousticness": [0.8, 0.1],
            "instrumentalness": [0.9, 0.0],
            "speechiness": [0.05, 0.08],
            "liveness": [0.1, 0.15],
            "tempo": [80.0, 125.0],
            "duration_ms": [210000, 180000]
        })

    def test_auth_validate_credentials_missing(self):
        auth = SpotifyAuth(client_id=None, client_secret=None)
        with self.assertRaises(SpotifyAuthError):
            auth.validate_credentials()

    def test_auth_get_authorization_url(self):
        auth = SpotifyAuth(client_id="test_id", client_secret="test_secret", redirect_uri="http://localhost/callback")
        url = auth.get_authorization_url()
        self.assertIn("https://accounts.spotify.com/authorize", url)
        self.assertIn("client_id=test_id", url)
        self.assertIn("response_type=code", url)

    def test_map_recently_played_to_internal_schema(self):
        df = map_recently_played_to_internal(
            self.mock_spotify_json,
            user_id="USR_SPOTIFY_TEST",
            source="spotify_api",
            data_type="real"
        )
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns[:5]), ["event_id", "user_id", "track_id", "played_at", "track_name"])
        self.assertEqual(df["data_type"].iloc[0], "real")
        self.assertEqual(df["source"].iloc[0], "spotify_api")

    def test_map_empty_response(self):
        df = map_recently_played_to_internal({}, user_id="USR_EMPTY")
        self.assertTrue(df.empty)

    def test_audio_feature_provider_enrichment(self):
        mapped_df = map_recently_played_to_internal(self.mock_spotify_json, user_id="USR_TEST")
        enriched = AudioFeatureProvider.enrich_tracks_with_features(mapped_df, self.catalog_df)

        self.assertIn("energy", enriched.columns)
        self.assertEqual(enriched["energy"].iloc[0], 0.2)
        self.assertEqual(enriched["energy"].iloc[1], 0.8)

    @patch("src.spotify.spotify_client.requests.get")
    def test_client_rate_limit_retry_429(self, mock_get):
        # Mock 429 rate limit then 200 success
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429
        mock_resp_429.headers = {"Retry-After": "0"}

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {"id": "test_user"}

        mock_get.side_effect = [mock_resp_429, mock_resp_200]

        auth = SpotifyAuth(client_id="id", client_secret="secret")
        auth.access_token = "valid_token"
        auth.token_expires_at = 9999999999.0

        client = SpotifyClient(auth=auth)
        res = client.get_user_profile()

        self.assertEqual(res, {"id": "test_user"})
        self.assertEqual(mock_get.call_count, 2)

    def test_end_to_end_spotify_pipeline_demo_mode(self):
        # Scratch catalog setup
        os.makedirs("scratch", exist_ok=True)
        cat_path = "scratch/temp_spotify_pipeline_catalog.csv"
        self.catalog_df.to_csv(cat_path, index=False)

        try:
            history_df, user_profile, recs = run_spotify_pipeline(
                mode="DEMO",
                catalog_path=cat_path,
                mock_response=self.mock_spotify_json,
                top_n_recs=2
            )

            self.assertFalse(history_df.empty)
            self.assertEqual(user_profile["user_id"], "USR_SPOTIFY_001")
            self.assertIn("audio_feature_summary", user_profile)
            self.assertEqual(len(recs), 2)
            self.assertIn("recommendation_reason", recs.columns)
        finally:
            if os.path.exists(cat_path):
                os.remove(cat_path)


if __name__ == "__main__":
    unittest.main()
