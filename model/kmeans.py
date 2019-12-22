import numpy as np

from model.point import Point


class KMeans:

    def __init__(self, k, seed=None):
        """KMeans constructor

        :param k: total of cluster
            :type: int

        :param seed: random seed to initialize centroids
            :required: optional
            :type: int
        """

        if seed:
            np.random.seed(seed)

        assert k > 0

        self.k = k
        self.seed = seed
        self.cluster_labels = []
        self.centroids = []

    def inti_centroids(self, data):
        """Get random centroids based on number of point in data set and self.k

        :param data: data with coordinates
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :return: random centroids
            :type: list
            :example:
                [3, 1, 6]
        """

        centroids_index = np.random.permutation(len(data))[:self.k]
        centroids_points = np.array(data)[centroids_index]
        return centroids_points

    @staticmethod
    def pairwise_distances(points, centroids):
        """Create pairwise distance between points and the centroids

        :param points:
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :param centroids:
            :type: list, tuple, np.array
            :example:
                [[0, 0], [3, 3]]
        :return: distance
            :type: list
            :example:
                [[14.866068747318506, 1.0, 0.0],
                 [14.212670403551895, 1.4142135623730951, 1.0],
                 [14.142135623730951, 0.0, 1.0], ... ]
        """

        distances = list()
        for index, point in enumerate(points):
            distances.append([])
            point = Point(point)
            for centroid in centroids:
                centroid = Point(centroid)
                distance = centroid.distance(point)
                distances[index].append(distance)
        return distances

    def assign_clusters(self, data):
        """Assign cluster of data based on self.cluster_labels

        :param data: data with coordinates
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :return: cluster labels
        """

        distances_to_centroids = self.pairwise_distances(data, self.centroids)
        cluster_labels = np.argmin(distances_to_centroids, axis=1)
        return cluster_labels

    def update_centroids(self, data):
        """Update centroids

        :param data: data with coordinates
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :return: new_centroids
        """

        new_centroids = np.array([data[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, x, epochs):
        """Train model

        :param x: data with coordinates
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :param epochs: number of epochs to train the model
            :type: int
        :return: history_data with all self.centroids and self.cluster_labels during fitting
            :type: dict
        """

        self.centroids = self.inti_centroids(x)
        for epoch in range(epochs):
            self.cluster_labels = self.assign_clusters(x)
            self.centroids = self.update_centroids(x)
        return self

    def predict(self, x):
        """Predict x based on fitted self.clusters_labels

        :param x: data with coordinates
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :return:
        """

        return self.assign_clusters(x)

    def __repr__(self):
        return f"""KMeans(k={self.k}, 
            seed={self.seed}, 
            fitted={True if len(self.cluster_labels) != 0 else False}, 
            centroids={self.centroids})"""
