import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def euclidean_distance(X, Y):
    a2 = np.sum(X ** 2, axis=1)[:, None]
    b2 = np.sum(Y ** 2, axis=1)[None, :]
    ab = X @ Y.T
    return np.sqrt(np.maximum(a2 + b2 - 2 * ab, 0))

def manhattan_distance(X, Y):
    return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)

DIST_FUNCS = {
    "euclidean": euclidean_distance,
    "l2": euclidean_distance,
    "manhattan": manhattan_distance,
    "l1": manhattan_distance,
}

class MyKnn:
    def __init__(self, k=[5], distance='euclidean', weighted=False, batch_size=1000):
        # Flatten and deduplicate k at any depth
        def flatten_k(seq):
            for item in seq:
                if isinstance(item, (list, tuple, np.ndarray)):
                    yield from flatten_k(item)
                else:
                    yield int(item)

        if isinstance(k, int):
            self.k_list = [k]
        elif isinstance(k, (list, tuple, np.ndarray)):
            # Robustly flatten nested k and deduplicate
            self.k_list = sorted(set(flatten_k(k)))
        else:
            raise ValueError("k must be int or list/tuple/array of ints.")
        self.maxK = max(self.k_list)
        if distance not in DIST_FUNCS:
            raise ValueError(f"Unknown distance metric: {distance}")
        self.dist_fn = DIST_FUNCS[distance]
        self.weighted = weighted
        self.batch_size = batch_size
        self._topk_dists = None
        self._topk_classes = None
        self.fitted = False

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.fitted = True
        self._topk_dists = None
        self._topk_classes = None
        return self

    def compute_topk_neighbors(self, X_test, verbose=1):
        assert self.fitted, "Call fit() before compute_topk_neighbors"
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]
        maxK = self.maxK

        topk_dists = np.full((n_test, maxK), np.inf)
        topk_classes = np.full((n_test, maxK), -1, dtype=self.y_train.dtype)

        for start in range(0, n_train, self.batch_size):
            end = min(start + self.batch_size, n_train)
            X_batch = self.X_train[start:end]
            y_batch = self.y_train[start:end]

            dists = self.dist_fn(X_test, X_batch)  # (n_test, batch_size)
            all_dists = np.concatenate([topk_dists, dists], axis=1)
            all_classes = np.concatenate([topk_classes, np.tile(y_batch, (n_test, 1))], axis=1)
            idx = np.argsort(all_dists, axis=1)[:, :maxK]
            row_idx = np.arange(n_test)[:, None]
            topk_dists = all_dists[row_idx, idx]
            topk_classes = all_classes[row_idx, idx]

            if verbose:
                print(f"Processed train batch {start}:{end} of {n_train}; memory: {topk_dists.nbytes//1024} KB")

        self._topk_dists = topk_dists
        self._topk_classes = topk_classes

    def predict(self, X_test=None, k=None, verbose=1):
        # Flatten and deduplicate k at any depth
        def flatten(seq):
            for item in seq:
                if isinstance(item, (list, tuple, np.ndarray)):
                    yield from flatten(item)
                else:
                    yield int(item)
        if k is None:
            k_list = self.k_list
        elif isinstance(k, int):
            k_list = [k]
        elif isinstance(k, (list, tuple, np.ndarray)):
            k_flat = [int(x) for x in flatten(k) if isinstance(x, (int, np.integer))]
            if not k_flat:
                raise ValueError("k must contain at least one int.")
            k_list = sorted(set(k_flat))
        else:
            raise ValueError("k must be int, or (possibly nested) list/tuple/array of ints, or None.")

        for k_ in k_list:
            if k_ > self.maxK:
                raise ValueError(f"k={k_} requested but maxK={self.maxK}; initialize with higher k if needed.")

        # If X_test is given, or no cache, compute neighbors
        # Always check X_test shape to ensure match!
        needs_compute = False
        if X_test is not None:
            needs_compute = True
        elif self._topk_classes is None:
            needs_compute = True
        elif hasattr(self, "_last_Xtest_shape") and self._topk_classes.shape[0] != self._last_Xtest_shape:
            needs_compute = True
        if needs_compute:
            if X_test is None:
                raise ValueError("X_test must be provided on first prediction or after fit.")
            self.compute_topk_neighbors(X_test, verbose=verbose)
            self._last_Xtest_shape = self._topk_classes.shape[0]

        results = {}
        for k_ in k_list:
            preds = []
            for i in range(self._topk_classes.shape[0]):
                neighbor_labels = self._topk_classes[i, :k_]
                neighbor_distances = self._topk_dists[i, :k_]
                if self.weighted:
                    weights = 1/(neighbor_distances + 1e-8)
                    weights = np.clip(weights, 0, 1e6)
                    label_score = {}
                    for label, w in zip(neighbor_labels, weights):
                        label_score[label] = label_score.get(label, 0) + w
                    preds.append(max(label_score.items(), key=lambda x: x[1])[0])
                else:
                    preds.append(Counter(neighbor_labels).most_common(1)[0][0])
            results[k_] = np.array(preds)
        if len(k_list) == 1:
            return results[k_list[0]]
        return results

    def plot_elbow_curve(self, y_true, y_pred_dict, show=True, savepath=None, verbose=1):
        metrics = []
        ks = sorted(y_pred_dict.keys())
        for k in ks:
            macro_f1 = f1_score(y_true, y_pred_dict[k], average="macro")
            metrics.append(macro_f1)
            if verbose:
                print(f"k={k}: Macro-F1={macro_f1:.4f}")
        plt.figure()
        plt.plot(ks, metrics, 'o-m')
        plt.xlabel("k")
        plt.ylabel("Macro F1-score")
        plt.title("KNN Macro F1-score vs k (Elbow Curve)")
        plt.grid(True)
        if savepath:
            plt.savefig(savepath)
            if verbose:
                print(f"Elbow plot saved to: {savepath}")
        if show:
            plt.show()
        return dict(zip(ks, metrics))