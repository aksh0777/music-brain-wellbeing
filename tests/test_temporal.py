import unittest
import numpy as np
import pandas as pd
from src.features.temporal import cyclical_encode, generate_temporal_features


class TestTemporal(unittest.TestCase):

    def test_cyclical_encode(self):
        sin_val, cos_val = cyclical_encode(0, 24)
        self.assertTrue(np.isclose(sin_val, 0.0))
        self.assertTrue(np.isclose(cos_val, 1.0))

        sin_6, cos_6 = cyclical_encode(6, 24)
        self.assertTrue(np.isclose(sin_6, 1.0))
        self.assertTrue(np.isclose(cos_6, 0.0))

    def test_generate_temporal_features(self):
        df = pd.DataFrame({
            "played_at": pd.to_datetime([
                "2026-08-01T00:00:00Z", # Saturday (Day 5), 00:00
                "2026-08-03T12:00:00Z"  # Monday (Day 0), 12:00
            ])
        })

        temporal_df = generate_temporal_features(df)

        self.assertEqual(list(temporal_df["hour"]), [0, 12])
        self.assertEqual(list(temporal_df["day_of_week"]), [5, 0])
        self.assertEqual(list(temporal_df["is_weekend"]), [1, 0])

        self.assertIn("hour_sin", temporal_df.columns)
        self.assertIn("hour_cos", temporal_df.columns)
        self.assertIn("day_sin", temporal_df.columns)
        self.assertIn("day_cos", temporal_df.columns)


if __name__ == "__main__":
    unittest.main()
