import unittest
import numpy as np
import pandas as pd
from src.features.music_features import (
    validate_feature_ranges,
    normalize_tempo,
    extract_feature_matrix,
    scale_feature_matrix
)


class TestMusicFeatures(unittest.TestCase):

    def test_validate_feature_ranges(self):
        df = pd.DataFrame({
            "danceability": [-0.5, 1.5],
            "energy": [0.5, 0.8],
            "tempo": [10.0, 300.0]
        })
        validated = validate_feature_ranges(df)

        self.assertEqual(validated["danceability"].min(), 0.0)
        self.assertEqual(validated["danceability"].max(), 1.0)
        self.assertEqual(validated["tempo"].min(), 30.0)
        self.assertEqual(validated["tempo"].max(), 250.0)

    def test_normalize_tempo(self):
        df = pd.DataFrame({"tempo": [125.0, 250.0, 300.0]})
        normed = normalize_tempo(df)

        self.assertIn("tempo_norm", normed.columns)
        self.assertEqual(list(normed["tempo_norm"]), [0.5, 1.0, 1.0])

    def test_extract_and_scale_feature_matrix(self):
        df = pd.DataFrame({
            "valence": [0.2, 0.8],
            "energy": [0.3, 0.9],
            "danceability": [0.4, 0.6],
            "acousticness": [0.9, 0.1],
            "instrumentalness": [0.8, 0.0],
            "tempo": [80.0, 140.0]
        })
        df_norm = normalize_tempo(df)

        X, features = extract_feature_matrix(df_norm)
        self.assertEqual(X.shape, (2, 6))

        X_scaled, scaler = scale_feature_matrix(X)
        self.assertEqual(X_scaled.shape, (2, 6))
        self.assertTrue(np.isclose(X_scaled.mean(axis=0), 0.0).all())


if __name__ == "__main__":
    unittest.main()
