import unittest
import pandas as pd
from src.features.user_profile import build_user_music_profile


class TestUserProfile(unittest.TestCase):

    def test_build_user_music_profile(self):
        history_df = pd.DataFrame({
            "event_id": ["E1", "E2", "E3"],
            "user_id": ["USR001", "USR001", "USR001"],
            "track_id": ["T1", "T2", "T1"],
            "played_at": pd.to_datetime([
                "2026-08-01T08:00:00Z",
                "2026-08-01T08:04:00Z",
                "2026-08-01T15:00:00Z"
            ]),
            "data_type": ["synthetic/demo", "synthetic/demo", "synthetic/demo"]
        })

        catalog_df = pd.DataFrame({
            "track_id": ["T1", "T2"],
            "energy": [0.4, 0.8],
            "valence": [0.3, 0.7],
            "danceability": [0.5, 0.6],
            "acousticness": [0.7, 0.2],
            "instrumentalness": [0.8, 0.1],
            "tempo": [90.0, 130.0]
        })

        profile = build_user_music_profile(history_df, catalog_df)

        self.assertEqual(profile["user_id"], "USR001")
        self.assertEqual(profile["total_tracks_listened"], 3)
        self.assertEqual(profile["total_sessions"], 2)
        self.assertIn("audio_feature_summary", profile)
        self.assertEqual(profile["data_provenance"], "synthetic/demo")


if __name__ == "__main__":
    unittest.main()
