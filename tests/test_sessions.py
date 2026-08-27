import unittest
import pandas as pd
from src.features.sessions import assign_listening_sessions, compute_session_summary


class TestSessions(unittest.TestCase):

    def test_assign_listening_sessions_gap_detection(self):
        df = pd.DataFrame({
            "event_id": ["E1", "E2", "E3", "E4"],
            "user_id": ["U1", "U1", "U1", "U1"],
            "track_id": ["T1", "T2", "T3", "T4"],
            "played_at": pd.to_datetime([
                "2026-08-01T10:00:00Z",
                "2026-08-01T10:05:00Z", # +5 min (Same session)
                "2026-08-01T10:45:00Z", # +40 min (>30 min, New session)
                "2026-08-01T10:50:00Z"  # +5 min (Same session)
            ])
        })

        sessions_df = assign_listening_sessions(df, gap_minutes=30)

        self.assertIn("session_id", sessions_df.columns)
        unique_sessions = sessions_df["session_id"].nunique()
        self.assertEqual(unique_sessions, 2)

        positions = list(sessions_df["session_position"])
        self.assertEqual(positions, [1, 2, 1, 2])

    def test_compute_session_summary(self):
        history_df = pd.DataFrame({
            "event_id": ["E1", "E2"],
            "user_id": ["U1", "U1"],
            "track_id": ["T1", "T2"],
            "played_at": pd.to_datetime([
                "2026-08-01T10:00:00Z",
                "2026-08-01T10:05:00Z"
            ])
        })

        catalog_df = pd.DataFrame({
            "track_id": ["T1", "T2"],
            "energy": [0.4, 0.8],
            "valence": [0.5, 0.7],
            "tempo": [100.0, 120.0]
        })

        summary = compute_session_summary(history_df, catalog_df)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary["track_count"].iloc[0], 2)
        self.assertAlmostEqual(summary["avg_energy"].iloc[0], 0.6)


if __name__ == "__main__":
    unittest.main()
