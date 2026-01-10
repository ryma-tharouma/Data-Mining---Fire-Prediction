import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class MyCLARANS:
    def __init__(self, n_clusters=3, num_local=5, max_neighbor=50,
                 metric='euclidean', random_state=None, verbose=True):
        """
        n_clusters   : number of clusters (k)
        num_local    : number of random restarts
        max_neighbor : max number of non-improving neighbors
        metric       : distance metric
        """
        self.n_clusters = n_clusters
        self.num_local = num_local
        self.max_neighbor = max_neighbor
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        self.medoids_ = None
        self.labels_ = None
        self.best_cost_ = np.inf

        if random_state is not None:
            np.random.seed(random_state)

    # --------------------------------------------------
    # Utility functions
    # --------------------------------------------------
    def _compute_cost(self, X, medoid_indices):
        medoids = X[medoid_indices]
        distances = cdist(X, medoids, metric=self.metric)
        return np.sum(np.min(distances, axis=1))

    def _assign_labels(self, X, medoid_indices):
        medoids = X[medoid_indices]
        distances = cdist(X, medoids, metric=self.metric)
        return np.argmin(distances, axis=1)

    # --------------------------------------------------
    # Core CLARANS algorithm
    # --------------------------------------------------
    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        best_global_medoids = None
        best_global_cost = np.inf

        for local_iter in range(self.num_local):
            # 1️⃣ Random initialization
            current_medoids = np.random.choice(
                n_samples, self.n_clusters, replace=False
            )
            current_cost = self._compute_cost(X, current_medoids)

            if self.verbose:
                print(f"[Local {local_iter+1}] Initial cost = {current_cost:.4f}")

            neighbor_count = 0

            # 2️⃣ Randomized local search
            while neighbor_count < self.max_neighbor:
                medoid_idx = np.random.choice(current_medoids)
                non_medoids = list(set(range(n_samples)) - set(current_medoids))
                candidate = np.random.choice(non_medoids)

                new_medoids = current_medoids.copy()
                new_medoids[new_medoids == medoid_idx] = candidate

                new_cost = self._compute_cost(X, new_medoids)

                if new_cost < current_cost:
                    current_medoids = new_medoids
                    current_cost = new_cost
                    neighbor_count = 0

                    if self.verbose:
                        print(f"  ↓ Improved cost = {new_cost:.4f}")
                else:
                    neighbor_count += 1

            # 3️⃣ Keep best global solution
            if current_cost < best_global_cost:
                best_global_cost = current_cost
                best_global_medoids = current_medoids

        # Save results
        self.medoids_ = best_global_medoids
        self.best_cost_ = best_global_cost
        self.labels_ = self._assign_labels(X, self.medoids_)

        return self

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    def silhouette(self, X):
        if self.labels_ is None:
            raise ValueError("Fit the model first.")
        if len(np.unique(self.labels_)) < 2:
            return np.nan
        return silhouette_score(X, self.labels_, metric=self.metric)

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    def plot_clusters(self, X, reducer='pca', **kwargs):
        X = np.array(X)
        labels = self.labels_

        if X.shape[1] > 2:
            if reducer == 'pca':
                X_2d = PCA(n_components=2, **kwargs).fit_transform(X)
            elif reducer == 'tsne':
                X_2d = TSNE(n_components=2, **kwargs).fit_transform(X)
            else:
                raise ValueError("reducer must be 'pca' or 'tsne'")
        else:
            X_2d = X

        plt.figure(figsize=(8,6))
        for k in np.unique(labels):
            plt.scatter(
                X_2d[labels == k, 0],
                X_2d[labels == k, 1],
                label=f'Cluster {k}',
                alpha=0.7
            )

        # Plot medoids
        medoid_points = X_2d[self.medoids_]
        plt.scatter(
            medoid_points[:,0], medoid_points[:,1],
            c='black', marker='X', s=120, label='Medoids'
        )

        plt.title('CLARANS Clustering')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
