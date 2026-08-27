import os
import unittest
import pandas as pd
from src.data.music_loader import load_music_catalog, load_listening_history


class TestMusicLoader(unittest.TestCase):

    def test_load_music_catalog_valid(self):
        catalog_file = "scratch/temp_tracks.csv"
        df = pd.DataFrame({
            "track_id": ["T1", "T2"],
            "track_name": ["Track 1", "Track 2"],
            "artist_name": ["Artist 1", "Artist 2"],
            "genre": ["Pop", "Rock"],
            "danceability": [0.7, 0.5],
            "energy": [0.8, 0.9],
            "valence": [0.6, 0.4],
            "tempo": [120.0, 140.0],
            "acousticness": [0.2, 0.1],
            "instrumentalness": [0.0, 0.2],
            "speechiness": [0.05, 0.08],
            "liveness": [0.1, 0.15]
        })
        df.to_csv(catalog_file, index=False)

        try:
            loaded = load_music_catalog(catalog_file)
            self.assertEqual(len(loaded), 2)
            self.assertIn("track_id", loaded.columns)
        finally:
            if os.path.exists(catalog_file):
                os.remove(catalog_file)

    def test_load_music_catalog_missing_column(self):
        catalog_file = "scratch/temp_invalid_tracks.csv"
        df = pd.DataFrame({
            "track_id": ["T1"],
            "track_name": ["Track 1"]
        })
        df.to_csv(catalog_file, index=False)

        try:
            with self.assertRaises(ValueError):
                load_music_catalog(catalog_file)
        finally:
            if os.path.exists(catalog_file):
                os.remove(catalog_file)

    def test_load_listening_history_valid(self):
        history_file = "scratch/temp_history.csv"
        df = pd.DataFrame({
            "event_id": ["E1", "E2"],
            "user_id": ["U1", "U1"],
            "track_id": ["T1", "T2"],
            "played_at": ["2026-08-01T10:00:00Z", "2026-08-01T10:05:00Z"],
            "data_type": ["synthetic/demo", "synthetic/demo"]
        })
        df.to_csv(history_file, index=False)

        try:
            loaded = load_listening_history(history_file)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["played_at"]))
        finally:
            if os.path.exists(history_file):
                os.remove(history_file)


if __name__ == "__main__":
    unittest.main()
