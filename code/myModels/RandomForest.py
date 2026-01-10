import numpy as np
import random
from collections import Counter, defaultdict
from DecisionTree import MyDecisionTree

class MyRandomForest:
    def __init__(
        self, 
        n_estimators=20, 
        max_depth=10, 
        min_samples_split=2, 
        n_bins=100, 
        chunk_size=1000, 
        max_features="sqrt", 
        bootstrap=True,
        verbose=True,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_bins = n_bins
        self.chunk_size = chunk_size
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.trees = []
        self.features_per_tree = []
        self.random_state = random_state

    def _get_max_features(self, n_features):
        if isinstance(self.max_features, int):
            return self.max_features
        elif self.max_features is None or self.max_features == 'all':
            return n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            return n_features

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []
        self.features_per_tree = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True) if self.bootstrap else np.arange(n_samples)
            X_sample = X[indices]
            y_sample = y[indices]
            # Random feature subset
            max_feats = self._get_max_features(n_features)
            feature_indices = np.random.choice(n_features, max_feats, replace=False)
            self.features_per_tree.append(feature_indices)
            if self.verbose:
                print(f"\nFitting tree {i+1}/{self.n_estimators}: features {feature_indices}")
            tree = MyDecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                n_bins=self.n_bins,
                chunk_size=self.chunk_size,
                verbose=False
            )
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.zeros((X.shape[0], len(self.trees)), dtype=object)
        for i, (tree, feats) in enumerate(zip(self.trees, self.features_per_tree)):
            preds[:, i] = tree.predict(X[:, feats])
        # Majority vote per row
        maj = []
        for row in preds:
            vc = Counter(row)
            maj.append(vc.most_common(1)[0][0])
        return np.array(maj)

    def predict_proba(self, X):
        # P(class | x) = fraction of trees predicting that class for x
        n_classes = None
        all_preds = []
        for i, (tree, feats) in enumerate(zip(self.trees, self.features_per_tree)):
            preds = tree.predict(X[:, feats])
            all_preds.append(preds)
        all_preds = np.array(all_preds)  # [n_estimators, n_samples]
        n_samples = X.shape[0]
        class_labels = np.unique(all_preds)
        n_classes = len(class_labels)
        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            votes = Counter(all_preds[:, i])
            for j, label in enumerate(class_labels):
                proba[i, j] = votes[label] / len(self.trees)
        return proba

    def feature_importances(self, n_features=None):
        """Average feature importances across trees, mapping back to the original features."""
        if not self.trees:
            return {}
        # Need to aggregate even when only a subset of features used per tree
        n_features = n_features or max([max(f) for f in self.features_per_tree]) + 1
        all_importances = np.zeros(n_features)
        for tree, feat_idx in zip(self.trees, self.features_per_tree):
            importances = tree.feature_importances()
            for i, idx in enumerate(feat_idx):
                all_importances[idx] += importances.get(i, 0)
        total = all_importances.sum()
        if total > 0:
            all_importances /= total
        return dict(enumerate(all_importances))

    def plot_feature_importances(self, feature_names=None, top_k=10):
        importances = self.feature_importances()
        if not importances:
            print("No feature importances (forest not fitted yet)")
            return

        # Sort descending (largest first)
        features = sorted(importances, key=lambda f: importances[f], reverse=True)
        features = features[:top_k]
        values = [importances[f] for f in features]

        # Print the top-k importances
        print("\nTop Feature Importances:")
        for f, v in zip(features, values):
            if feature_names:
                print(f"  {feature_names[f]} → {v:.4f}")
            else:
                print(f"  X[{f}] → {v:.4f}")

        # Labels
        if feature_names is None:
            labels = [f"X[{f}]" for f in features]
        else:
            labels = [feature_names[f] for f in features]

        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(labels, values, color='#3F51B5', edgecolor='#283593', linewidth=2)
        ax.invert_yaxis()  # largest on top

        # Better numeric labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            xpos = val + (0.01 * max(values))  # shift text to the right
            ax.text(xpos, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va='center', fontsize=10, weight='bold')

        ax.set_xlabel('Importance', fontsize=12, weight='bold')
        ax.set_ylabel('Feature', fontsize=12, weight='bold')
        ax.set_title(f'Top {top_k} Feature Importances (Random Forest)',
                     fontsize=14, weight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

