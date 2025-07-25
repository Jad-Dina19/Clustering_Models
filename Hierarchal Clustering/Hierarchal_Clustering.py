import numpy as np

class HierarchalClustering:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.clusters = None

    def ward_dist(self, cluster_a, cluster_b):
        
        mu_a = np.mean(cluster_a, axis=0)
        mu_b = np.mean(cluster_b, axis=0)

        n_a =  len(cluster_a)
        n_b = len(cluster_b)

        return (n_a * n_b)/(n_a + n_b) * np.sum((mu_a - mu_b)**2)

    def get_clusters(self):
        return self.clusters
    
    def fit(self, X):
        
        clusters = [[i] for i in range(X.shape[0])]
        
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            to_merge = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    
                    cluster_i = X[clusters[i]]
                    cluster_j = X[clusters[j]]

                    dist = self.ward_dist(cluster_i, cluster_j)
                
                    if(dist < min_dist):
                        min_dist = dist
                        to_merge = (i, j)
            
            clusters[to_merge[0]] = clusters[to_merge[0]] + clusters[to_merge[1]]
            del clusters[to_merge[1]]

        self.clusters = clusters
    
    def predict(self, X):
        labels = np.empty(X.shape[0], dtype= int)
        for cluster_idx, cluster in enumerate(self.clusters):
            for idx in cluster:
                labels[idx] = cluster_idx
        
        return labels

    

