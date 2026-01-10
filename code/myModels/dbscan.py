import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class MyDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', verbose=True, use_class_label=False):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.verbose = verbose
        self.use_class_label = use_class_label
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None

    def fit(self, X, class_labels=None):
        X = np.array(X)
        if self.use_class_label and class_labels is not None:
            X = np.hstack([X, np.array(class_labels).reshape(-1,1)])
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        db.fit(X)
        self.labels_ = db.labels_
        self.core_sample_indices_ = db.core_sample_indices_
        # Number of clusters, ignoring noise (-1 label)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        return self

    def fit_param_range(self, X, param_grid, class_labels=None):
        """
        param_grid: iterable of (eps, min_samples) pairs
        Returns: dict keyed by (eps, min_samples)
        """
        results = {}
        for eps, min_samples in param_grid:
            db = MyDBSCAN(eps=eps, min_samples=min_samples, metric=self.metric,
                          verbose=False, use_class_label=self.use_class_label)
            db.fit(X, class_labels=class_labels)
            labels = db.labels_
            # Only compute silhouette if there are >1 labels and at least one cluster
            valid = (len(np.unique(labels)) > 1 and db.n_clusters_ > 0 and 
                     np.sum(labels != -1) > 1)
            sil = silhouette_score(np.array(X)[labels != -1], labels[labels != -1]) if valid else np.nan
            results[(eps, min_samples)] = {
                'n_clusters': db.n_clusters_,
                'silhouette': sil,
                'labels': db.labels_,
                'core_sample_indices': db.core_sample_indices_
            }
            if self.verbose:
                print(f'eps={eps}, min_samples={min_samples} → clusters={db.n_clusters_}, silhouette={sil:.4f}')
        return results

    def plot_clusters(self, X, class_labels=None, reducer='pca', **kwargs):
        X = np.array(X)
        labels = self.labels_ if self.labels_ is not None else np.zeros(X.shape[0])
        if class_labels is not None:
            labels = np.array(class_labels)
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
            if k == -1:  # Noise
                plt.scatter(X_2d[labels==k,0], X_2d[labels==k,1], label='Noise', alpha=0.5, c='gray')
            else:
                plt.scatter(X_2d[labels==k,0], X_2d[labels==k,1], label=f'Cluster {k}', alpha=0.7)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.title('DBSCAN Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def evaluate_param_grid(self, X, param_grid, class_labels=None, plot=True):
        """
        Computes number of clusters and silhouette for param grid.
        param_grid: iterable of (eps, min_samples) pairs
        Returns a dict: {(eps, min_samples): {...}}
        """
        results = self.fit_param_range(X, param_grid, class_labels=class_labels)
        if plot:
            import matplotlib.pyplot as plt
            eps_vals = [pair[0] for pair in param_grid]
            min_samples_vals = [pair[1] for pair in param_grid]
            n_clusters = [results[pair]['n_clusters'] for pair in param_grid]
            silhouettes = [results[pair]['silhouette'] for pair in param_grid]

            fig, ax1 = plt.subplots(figsize=(8,5))
            color = 'tab:blue'
            ax1.set_xlabel('eps')
            ax1.set_ylabel('Number of clusters', color=color)
            ax1.plot(eps_vals, n_clusters, '-o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Silhouette', color=color)
            ax2.plot(eps_vals, silhouettes, '-s', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            plt.title('DBSCAN Evaluation: Number of Clusters & Silhouette')
            plt.show()

        # Also print table
        print("\n eps | min_samples | n_clusters | Silhouette")
        print("-----|-------------|------------|-----------")
        for eps, min_samples in param_grid:
            sil = results[(eps, min_samples)]['silhouette']
            nc = results[(eps, min_samples)]['n_clusters']
            print(f"{eps:3.2f} | {min_samples:11d} | {nc:10d} | {sil if not np.isnan(sil) else '  nan':>9}")
        
        return results