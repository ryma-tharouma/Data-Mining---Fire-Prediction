import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='kmeans++', 
                 random_state=None, verbose=True, use_class_label=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.use_class_label = use_class_label
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        if self.init == 'kmeans++':
            centroids = [X[rng.integers(n_samples)]]
            for _ in range(1, self.n_clusters):
                dist_sq = np.min(np.array([np.sum((X - c)**2, axis=1) for c in centroids]), axis=0)
                probs = dist_sq / dist_sq.sum()
                cumulative_probs = np.cumsum(probs)
                r = rng.random()
                idx = np.searchsorted(cumulative_probs, r)
                centroids.append(X[idx])
            return np.array(centroids)
        else:  # random
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[indices]

    def fit(self, X, class_labels=None):
        X = np.array(X)
        # Optionally append class label as a feature
        if self.use_class_label and class_labels is not None:
            X = np.hstack([X, np.array(class_labels).reshape(-1,1)])

        n_samples, n_features = X.shape
        self.centroids = self._initialize_centroids(X)
        
        for i in range(self.max_iter):
            dist = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(dist, axis=1)
            new_centroids = np.array([
                X[labels==k].mean(axis=0) if np.any(labels==k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                if self.verbose:
                    print(f'Converged at iteration {i+1}, shift={shift:.6f}')
                break
        self.labels_ = labels
        self.inertia_ = np.sum((X - self.centroids[labels])**2)
        return self

    def fit_k_range(self, X, k_list, class_labels=None):
        results = {}
        for k in k_list:
            km = MyKMeans(n_clusters=k, max_iter=self.max_iter, tol=self.tol,
                          init=self.init, random_state=self.random_state,
                          verbose=False, use_class_label=self.use_class_label)
            km.fit(X, class_labels=class_labels)
            sil = silhouette_score(np.array(X), km.labels_) if k>1 else np.nan
            results[k] = {'inertia': km.inertia_, 'silhouette': sil, 
                          'labels': km.labels_, 'centroids': km.centroids}
            if self.verbose:
                print(f'k={k} → inertia={km.inertia_:.4f}, silhouette={sil:.4f}')
        return results

    def plot_clusters(self, X, class_labels=None, reducer='pca', **kwargs):
        """
        Plot clusters in 2D for any-dimensional data.
        reducer: 'pca' or 'tsne'
        class_labels: optional true labels to plot instead of cluster labels
        kwargs: passed to PCA/TSNE
        """
        X = np.array(X)
        labels = self.labels_ if self.labels_ is not None else np.zeros(X.shape[0])
        if class_labels is not None:
            labels = np.array(class_labels)

        if X.shape[1] > 2:
            if reducer == 'pca':
                X_2d = PCA(n_components=2, **kwargs).fit_transform(X)
                if self.centroids is not None:
                    cent_2d = PCA(n_components=2, **kwargs).fit_transform(self.centroids)
            elif reducer == 'tsne':
                X_2d = TSNE(n_components=2, **kwargs).fit_transform(X)
                if self.centroids is not None:
                    cent_2d = TSNE(n_components=2, **kwargs).fit_transform(self.centroids)
            else:
                raise ValueError("reducer must be 'pca' or 'tsne'")
        else:
            X_2d = X
            cent_2d = self.centroids if self.centroids is not None else None

        plt.figure(figsize=(8,6))
        for k in np.unique(labels):
            plt.scatter(X_2d[labels==k,0], X_2d[labels==k,1], label=f'Cluster {k}', alpha=0.7)
        if self.centroids is not None:
            plt.scatter(cent_2d[:,0], cent_2d[:,1], c='black', s=100, marker='x', label='Centroids')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.title('Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def evaluate_k_range(self, X, k_list, class_labels=None, plot=True):
        """
        Compute inertia and silhouette for multiple k values.
        Returns a dict: {k: {'inertia': ..., 'silhouette': ...}}
        """
        results = self.fit_k_range(X, k_list, class_labels=class_labels)

        if plot:
            import matplotlib.pyplot as plt
            ks = sorted(results.keys())
            inertias = [results[k]['inertia'] for k in ks]
            silhouettes = [results[k]['silhouette'] for k in ks]

            fig, ax1 = plt.subplots(figsize=(8,5))
            color = 'tab:blue'
            ax1.set_xlabel('k')
            ax1.set_ylabel('Inertia', color=color)
            ax1.plot(ks, inertias, '-o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Silhouette', color=color)
            ax2.plot(ks, silhouettes, '-s', color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('KMeans Evaluation: Inertia & Silhouette')
            plt.show()

        # Also print table
        print("\n k |   Inertia   | Silhouette")
        print("---|-------------|-----------")
        for k in ks:
            sil = results[k]['silhouette']
            print(f"{k:2d} | {results[k]['inertia']:11.4f} | {sil if not np.isnan(sil) else '  nan':>9}")
        
        return results
