import unittest
import numpy as np
import pandas as pd
from src.features.clustering import (
    evaluate_k_candidates,
    train_kmeans_clustering,
    generate_cluster_labels,
    assign_nearest_cluster
)


class TestClustering(unittest.TestCase):

    def test_evaluate_k_candidates(self):
        np.random.seed(42)
        X = np.random.randn(30, 4)
        k_vals, inertias, s_scores = evaluate_k_candidates(X, k_min=2, k_max=4)

        self.assertEqual(k_vals, [2, 3, 4])
        self.assertEqual(len(inertias), 3)
        self.assertEqual(len(s_scores), 3)

    def test_train_kmeans_clustering_and_labeling(self):
        np.random.seed(42)
        X = np.random.randn(20, 5)
        model, labels = train_kmeans_clustering(X, n_clusters=3)

        self.assertEqual(len(labels), 20)
        self.assertEqual(len(np.unique(labels)), 3)

        feature_names = ["valence", "energy", "danceability", "acousticness", "instrumentalness"]
        centroids = model.cluster_centers_
        cluster_names = generate_cluster_labels(centroids, feature_names)

        self.assertEqual(len(cluster_names), 3)
        self.assertTrue(all("Cluster" in c for c in cluster_names))

    def test_assign_nearest_cluster(self):
        np.random.seed(42)
        X_train = np.random.randn(20, 3)
        model, _ = train_kmeans_clustering(X_train, n_clusters=2)

        X_new = np.random.randn(5, 3)
        assigned = assign_nearest_cluster(X_new, model)

        self.assertEqual(len(assigned), 5)
        self.assertTrue(set(assigned).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
