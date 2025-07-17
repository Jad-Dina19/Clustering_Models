import numpy as np

class K_Means:
    def __init__(self, k, tol=1e-3, max_iter=100):
        self.k = k
        self.centers = None
        self.tol = tol
        self.max_iter = max_iter

    def euclidean_dist(self, x, c):
        return np.linalg.norm(np.array(x) - np.array(c)) ** 2

    def initialize_centroid(self, X):
        n_samples = X.shape[0]
        centers = [X[np.random.randint(n_samples)]]

        for _ in range(self.k - 1):  # only need to pick k-1 more centers
            distances = []
            for x in X:
                min_dist = min(self.euclidean_dist(x, c) for c in centers)
                distances.append(min_dist)
            distances = np.array(distances)
            prob = distances / distances.sum()
            cum_prob = np.cumsum(prob)
            r = np.random.rand()
            index = np.searchsorted(cum_prob, r, side = 'right')
            centers.append(X[index])

        return np.array(centers)

    def fit(self, X):
        self.centers = self.initialize_centroid(X)

    def get_centroid(self):
        return self.centers

    def k_means(self, X):
       
        for _ in range(self.max_iter):
            labels = []
            for x in X:
                x_dist = [self.euclidean_dist(x, c) for c in self.centers]
                labels.append(np.argmin(x_dist))
            labels = np.array(labels)

            new_centers = []
            for j in range(self.k):
                if np.any(labels == j):
                    new_centers.append(X[labels == j].mean(axis=0))
                else:
                    new_centers.append(self.centers[j])  # no points, keep old center

            new_centers = np.array(new_centers)

            if np.linalg.norm(self.centers - new_centers) < self.tol:
                break

            self.centers = new_centers

        return labels

    def inertia(self, X):
        labels = self.k_means(X)
        return sum(np.sum((X[labels == j] - self.centers[j]) ** 2) for j in range(len(self.centers)))
